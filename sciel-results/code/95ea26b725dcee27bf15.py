import time
import random

def _gf5_add(a, b):
    return (a + b) % 5

def _gf5_mul(a, b):
    return (a * b) % 5

def _gf5_inv(a):
    if a == 0:
        return 0
    for x in range(1, 5):
        if _gf5_mul(a, x) == 1:
            return x
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Construct PG(2,5) polarity graph
    # Points are 1D subspaces of GF(5)^3
    # Represent each point by a canonical vector
    
    def canonicalize(v):
        # Make first nonzero component 1
        for i in range(3):
            if v[i] != 0:
                inv = _gf5_inv(v[i])
                return tuple(_gf5_mul(v[j], inv) for j in range(3))
        return (0, 0, 0)
    
    points = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                pt = canonicalize((a, b, c))
                if pt not in points:
                    points.append(pt)
    
    n_pts = len(points)  # should be 31
    
    # Polarity: B(u,v) = u1*v2 + u2*v3 + u3*v1
    def bilinear(u, v):
        return (_gf5_add(_gf5_add(_gf5_mul(u[0], v[1]), _gf5_mul(u[1], v[2])), _gf5_mul(u[2], v[0]))) % 5
    
    # Build adjacency
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if bilinear(points[i], points[j]) == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    
    # Verify C4-free and alpha < 11
    def has_c4(adj, n):
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    common = adj[i] & adj[j]
                    if len(common) >= 2:
                        return True
        return False
    
    def max_independent_set(adj, n):
        # Simple greedy + backtracking for small n
        # Use bitmasks for n <= 64
        if n > 64:
            return n
        # Branch and bound
        best = [0]
        
        def backtrack(remaining, current_size):
            if current_size + len(remaining) <= best[0]:
                return
            if not remaining:
                best[0] = max(best[0], current_size)
                return
            # Choose vertex with fewest connections in remaining
            v = min(remaining, key=lambda x: len(adj[x] & remaining))
            # Branch 1: include v
            new_remaining = remaining - {v} - adj[v]
            backtrack(new_remaining, current_size + 1)
            # Branch 2: exclude v
            backtrack(remaining - {v}, current_size)
        
        backtrack(set(range(n)), 0)
        return best[0]
    
    if has_c4(adj, n_pts):
        # Should not happen for polarity graph
        pass
    
    alpha = max_independent_set(adj, n_pts)
    
    if alpha >= 11:
        # Fallback to seed
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Try to extend
    current_n = n_pts
    current_adj = [set(adj[i]) for i in range(n_pts)]
    current_edges = list(edges)
    
    # Try adding vertices
    for new_vertex_id in range(current_n, 44):
        if time.time() > deadline:
            break
        
        # Try to find a valid set of edges for the new vertex
        # The new vertex must not create C4 and must not create independent 11-set
        # For simplicity, try adding vertex connected to a subset of existing vertices
        # that maintains C4-free
        
        best_subset = None
        # Try subsets of size around the average degree
        avg_deg = sum(len(s) for s in current_adj) / current_n if current_n > 0 else 0
        
        # For C4-free: for any two neighbors u,v of new vertex, u and v must not share another common neighbor
        # Actually, new vertex w: for any u,v in N(w), the number of common neighbors of u and v in the old graph must be 0
        # because w is a common neighbor, and if there's another, that's C4
        
        # So N(w) must be a set where no two vertices share a common neighbor in the old graph
        # i.e., N(w) is a "C4-free" set in the complement sense
        
        # Try to find a large such set
        # Greedy: pick vertices one by one that don't conflict
        
        candidate_neighbors = list(range(current_n))
        valid_neighbors = []
        
        # For each pair in candidate, check if they share a common neighbor
        # We want a set S such that for all u,v in S, |N(u) ∩ N(v)| = 0 in old graph
        
        # This is equivalent to S being an independent set in the "2-shadow" graph
        # where u~v if |N(u)∩N(v)| >= 1
        
        # Build conflict graph
        conflict = [set() for _ in range(current_n)]
        for i in range(current_n):
            for j in range(i+1, current_n):
                if len(current_adj[i] & current_adj[j]) > 0:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # Find large independent set in conflict graph = valid neighbor set
        # Use greedy
        remaining = set(range(current_n))
        chosen = set()
        while remaining:
            if time.time() > deadline:
                break
            # Pick vertex with min degree in conflict restricted to remaining
            v = min(remaining, key=lambda x: len(conflict[x] & remaining))
            chosen.add(v)
            remaining -= {v}
            remaining -= conflict[v]
        
        if len(chosen) < 1:
            continue
        
        # Check alpha after adding
        new_adj = [set(s) for s in current_adj]
        new_adj.append(set(chosen))
        for v in chosen:
            new_adj[v].add(current_n)
        
        # Quick C4 check for new vertex
        c4_ok = True
        for u in chosen:
            for v in chosen:
                if u < v:
                    if len(new_adj[u] & new_adj[v]) >= 2:
                        # One is the new vertex, check if another exists
                        common = new_adj[u] & new_adj[v]
                        if len(common) >= 2:
                            c4_ok = False
                            break
            if not c4_ok:
                break
        
        if not c4_ok:
            continue
        
        # Check alpha
        if current_n + 1 <= 64:
            alpha_new = max_independent_set(new_adj, current_n + 1)
            if alpha_new >= 11:
                continue
        
        # Accept
        current_n += 1
        current_adj = new_adj
        for v in chosen:
            current_edges.append([v, current_n - 1])
    
    # Final verification
    final_adj = [set() for _ in range(current_n)]
    for e in current_edges:
        u, v = e
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    if has_c4(final_adj, current_n):
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    alpha_final = max_independent_set(final_adj, current_n)
    if alpha_final >= 11:
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return {'n': current_n, 'edges': current_edges}
