"""Greedy cost-minimization + incremental local search for 5-colourings of K_n without C4."""
import time
import random
import numpy as np

_START_TIME = time.time()
DEADLINE = 118.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

class _C4Tracker:
    """Incremental C4 violation tracker using per-color common-neighbor counts.
    
    For each color c, we track for every pair (u,v): how many common neighbors
    they have in color c. A C4 violation exists when any pair has >= 2 common
    neighbors in the same color.
    
    We maintain:
    - adj[c][u]: bitmask of neighbors of u in color c
    - common[c][u][v]: count of common neighbors of u,v in color c (for u<v)
    - total_violations: number of (c,u,v) with common[c][u][v] >= 2
    """
    def __init__(self, n):
        self.n = n
        self.adj = [[0] * n for _ in range(5)]
        # common[c] is a list of n lists; common[c][u][v] for u < v
        self.common = [[ [0] * n for _ in range(n) ] for _ in range(5)]
        self.total_violations = 0
        
    def _add_edge(self, c, u, v):
        """Add edge (u,v) to color c. Returns change in total_violations."""
        delta = 0
        # For each w != u, v: if w is neighbor of u in color c, then (w,v) common count changes
        # If w is neighbor of v in color c, then (u,w) common count changes
        adj_u = self.adj[c][u]
        adj_v = self.adj[c][v]
        
        # Common neighbors of u and v in color c (excluding each other since not yet added)
        # After adding, common neighbors of (u,v) = popcount(adj_u & adj_v) (they're not in each other's adj yet)
        # But we need to update common counts for pairs involving u and pairs involving v.
        
        # For each w in adj_u (w != v): common[c][min(u,w)][max(u,w)] and common[c][min(v,w)][max(v,w)] change
        # Actually: adding v to u's adj means:
        #   For each w in adj_u (w != v): u and w now have one more common neighbor (v)
        #   For each w in adj_v (w != u): v and w now have one more common neighbor (u)
        #   u and v now have popcount(adj_u & adj_v) common neighbors
        
        # Update pairs (u, w) for w in adj_u, w != v
        w = adj_u
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            w ^= lb
            if wi == v:
                continue
            a, b = (u, wi) if u < wi else (wi, u)
            old = self.common[c][a][b]
            new = old + 1
            self.common[c][a][b] = new
            if old < 2 and new >= 2:
                delta += 1
            elif old >= 2 and new < 2:
                delta -= 1
        
        # Update pairs (v, w) for w in adj_v, w != u
        w = adj_v
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            w ^= lb
            if wi == u:
                continue
            a, b = (v, wi) if v < wi else (wi, v)
            old = self.common[c][a][b]
            new = old + 1
            self.common[c][a][b] = new
            if old < 2 and new >= 2:
                delta += 1
            elif old >= 2 and new < 2:
                delta -= 1
        
        # Set common for (u, v)
        a, b = (u, v) if u < v else (v, u)
        cn = (adj_u & adj_v).bit_count()
        old = self.common[c][a][b]
        self.common[c][a][b] = cn
        if old < 2 and cn >= 2:
            delta += 1
        elif old >= 2 and cn < 2:
            delta -= 1
        
        # Update adj
        self.adj[c][u] |= (1 << v)
        self.adj[c][v] |= (1 << u)
        
        self.total_violations += delta
        return delta
    
    def _remove_edge(self, c, u, v):
        """Remove edge (u,v) from color c."""
        delta = 0
        adj_u = self.adj[c][u]
        adj_v = self.adj[c][v]
        
        # First remove from adj
        self.adj[c][u] &= ~(1 << v)
        self.adj[c][v] &= ~(1 << u)
        
        # Reset common for (u,v)
        a, b = (u, v) if u < v else (v, u)
        old = self.common[c][a][b]
        self.common[c][a][b] = 0
        if old >= 2:
            delta -= 1
        
        # For each w in adj_u (after removal, so w != v): u and w lose common neighbor v
        w = adj_u
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            w ^= lb
            aa, bb = (u, wi) if u < wi else (wi, u)
            old_c = self.common[c][aa][bb]
            new_c = old_c - 1
            self.common[c][aa][bb] = new_c
            if old_c >= 2 and new_c < 2:
                delta -= 1
            elif old_c < 2 and new_c >= 2:
                delta += 1
        
        # For each w in adj_v (after removal, so w != u): v and w lose common neighbor u
        w = adj_v
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            w ^= lb
            aa, bb = (v, wi) if v < wi else (wi, v)
            old_c = self.common[c][aa][bb]
            new_c = old_c - 1
            self.common[c][aa][bb] = new_c
            if old_c >= 2 and new_c < 2:
                delta -= 1
            elif old_c < 2 and new_c >= 2:
                delta += 1
        
        self.total_violations += delta
        return delta

def _greedy_construct(n, seed):
    """Greedy construction: assign edges one by one, choosing color with min C4 risk."""
    rng = random.Random(seed)
    tracker = _C4Tracker(n)
    colors = [0] * (n * (n - 1) // 2)
    
    # Edge list in lexicographic order
    edges = []
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            edges.append((a, b, idx))
            idx += 1
    
    # Process edges in a specific order: by vertex pair
    # For each edge, evaluate all 5 colors' incremental cost
    for u, v, eidx in edges:
        best_cost = float('inf')
        best_color = 0
        for c in range(5):
            # Compute cost: how many new violations would be created
            # Cost = number of pairs (u,w) and (v,w) that go from 1 common neighbor to 2
            # Plus the (u,v) pair itself if it would have >= 2 common neighbors
            cost = 0
            adj_u = tracker.adj[c][u]
            adj_v = tracker.adj[c][v]
            
            # Pairs (u, w) for w in adj_u, w != v: these would gain a common neighbor
            w = adj_u
            while w:
                lb = w & (-w)
                wi = lb.bit_length() - 1
                w ^= lb
                if wi == v:
                    continue
                a2, b2 = (u, wi) if u < wi else (wi, u)
                if tracker.common[c][a2][b2] == 1:
                    cost += 1  # would become 2 -> new violation
            
            # Pairs (v, w) for w in adj_v, w != u
            w = adj_v
            while w:
                lb = w & (-w)
                wi = lb.bit_length() - 1
                w ^= lb
                if wi == u:
                    continue
                a2, b2 = (v, wi) if v < wi else (wi, v)
                if tracker.common[c][a2][b2] == 1:
                    cost += 1
            
            # The (u,v) pair itself
            cn = (adj_u & adj_v).bit_count()
            if cn >= 2:
                cost += 1
            
            if cost < best_cost:
                best_cost = cost
                best_color = c
        
        colors[eidx] = best_color
        tracker._add_edge(best_color, u, v)
        
        if _time_left() < 5.0:
            break
    
    return colors, tracker.total_violations, tracker

def _local_search(n, colors, tracker, timeout=3.0):
    """Local search: flip edges to reduce violations."""
    start_t = time.time()
    edges = []
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            edges.append((a, b, idx))
            idx += 1
    
    current_viol = tracker.total_violations
    if current_viol == 0:
        return colors, tracker, 0
    
    best_viol = current_viol
    best_colors = colors[:]
    best_adj = [[row[:] for row in tracker.adj]]
    best_common = [[row[:] for row in tracker.common[c]] for c in range(5)]
    
    # Collect edges involved in violations for targeted search
    # Also do random edge flips
    for it in range(200000):
        if time.time() - start_t > timeout:
            break
        if _time_left() < 5.0:
            break
        
        # Pick random edge
        u, v, eidx = edges[random.randint(0, len(edges) - 1)]
        old_color = colors[eidx]
        new_color = (old_color + random.randint(1, 4)) % 5
        
        # Try the flip
        tracker._remove_edge(old_color, u, v)
        tracker._add_edge(new_color, u, v)
        colors[eidx] = new_color
        
        if tracker.total_violations <= current_viol:
            current_viol = tracker.total_violations
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
                best_adj = [[row[:] for row in tracker.adj]]
                best_common = [[row[:] for row in tracker.common[c]] for c in range(5)]
                if best_viol == 0:
                    break
        else:
            # Revert
            tracker._remove_edge(new_color, u, v)
            tracker._add_edge(old_color, u, v)
            colors[eidx] = old_color
    
    # Restore best
    if best_viol < current_viol:
        colors = best_colors
        tracker.adj = best_adj
        tracker.common = best_common
        tracker.total_violations = best_viol
    
    return colors, tracker, tracker.total_violations

def _try_n(n, time_budget):
    """Try multiple greedy constructions + local search for given n."""
    best_viol = float('inf')
    best_colors = None
    
    seed = 0
    while _time_left() > time_budget:
        seed += 1
        colors, viol, tracker = _greedy_construct(n, seed)
        
        if viol < best_viol:
            best_viol = viol
            best_colors = colors[:]
        
        if best_viol == 0:
            return best_colors
        
        # Local search to improve
        if _time_left() > time_budget + 1.0:
            colors, tracker, viol = _local_search(n, colors, tracker, timeout=2.0)
            if viol < best_viol:
                best_viol = viol
                best_colors = colors[:]
            if best_viol == 0:
                return best_colors
    
    if best_viol == 0:
        return best_colors
    return None

def _validate(n, colors):
    """Validate that no color class contains a C4."""
    neighbors = [[0] * n for _ in range(5)]
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            c = colors[idx]
            neighbors[c][a] |= (1 << b)
            neighbors[c][b] |= (1 << a)
            idx += 1
    
    for c in range(5):
        for u in range(n):
            nu = neighbors[c][u]
            for v in range(u + 1, n):
                nv = neighbors[c][v]
                if (nu & nv).bit_count() >= 2:
                    return False
    return True

def construct():
    random.seed(42)
    np.random.seed(42)
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try from high n down, spending more time on higher n
    for n in [29, 28, 27, 26]:
        if _time_left() < 8.0:
            break
        
        # Allocate time: more for smaller n (easier), but try bigger first
        if n >= 28:
            budget = min(_time_left() - 10.0, 15.0)
        elif n == 27:
            budget = min(_time_left() - 8.0, 20.0)
        else:
            budget = min(_time_left() - 5.0, 30.0)
        
        if budget < 3.0:
            continue
        
        result = _try_n(n, budget)
        if result is not None:
            # Validate
            if _validate(n, result):
                best_n = n
                best_colors = result
                break  # Found valid coloring for this n, no need to try smaller
    
    # If we haven't found anything for n >= 26, try a quick greedy for n=26
    if best_n < 26 and _time_left() > 10.0:
        result = _try_n(26, _time_left() - 5.0)
        if result is not None and _validate(26, result):
            best_n = 26
            best_colors = result
    
    return {'n': best_n, 'colors': best_colors}
