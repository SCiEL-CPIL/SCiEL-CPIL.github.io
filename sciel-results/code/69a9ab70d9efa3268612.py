import time
import random
import sys
import math

_START_TIME = time.time()
_DEADLINE = 130.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for a given coloring."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles using bitsets."""
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _find_one_mono_triangle(adj, n):
    """Find one monochromatic triangle, return (i, j, k, c) or None."""
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                if common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    return (i, j, k, c)
    return None

def _greedy_init(n):
    """
    Greedy initialization: build vertex by vertex, choosing edge colors
    to minimize immediate monochromatic triangle creation.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    
    return colors

def _sa_search(n, time_budget):
    """
    Simulated annealing to find a valid 4-coloring of K_n.
    Returns colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Precompute index-to-edge mapping for speed
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1
    
    best_result = None
    best_tri = float('inf')
    
    for restart in range(300):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with greedy coloring
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        # Simulated annealing parameters
        T = 3.0
        T_min = 0.01
        cooling = 0.9998
        accepted = 0
        total_steps = 0
        
        while True:
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random edge
            ri = random.randint(0, m - 1)
            old_c = colors[ri]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            u, v = idx_to_edge[ri]
            
            # Compute delta in triangle count
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = added - removed
            
            total_steps += 1
            
            if delta <= 0:
                colors[ri] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += delta
                accepted += 1
                
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_result = colors[:]
                    if tri_count == 0:
                        return colors
            else:
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        colors[ri] = new_c
                        adj[old_c][u] &= ~(1 << v)
                        adj[old_c][v] &= ~(1 << u)
                        adj[new_c][u] |= (1 << v)
                        adj[new_c][v] |= (1 << u)
                        tri_count += delta
                        accepted += 1
            
            T *= cooling
            
            # Occasionally restart if stuck
            if total_steps > 2000 and accepted < total_steps * 0.005:
                break
        
    return best_result if best_tri == 0 else None

def _local_search(n, time_budget):
    """
    Local search: iteratively recolor edges in monochromatic triangles
    to reduce triangle count.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    best_result = None
    best_tri = float('inf')
    
    for restart in range(150):
        if time.time() - start_time > time_budget:
            break
        
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        no_improve = 0
        max_no_improve = 100
        
        while no_improve < max_no_improve:
            if time.time() - start_time > time_budget:
                break
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                return colors
            
            i, j, k, c = tri
            
            # Try all 3 edges and 3 new colors each
            edges = [(i, j), (i, k), (j, k)]
            best_delta = float('inf')
            best_edge = None
            best_new_c = -1
            
            for u, v in edges:
                for new_c in range(4):
                    if new_c == c:
                        continue
                    removed = (adj[c][u] & adj[c][v]).bit_count()
                    added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    delta = added - removed
                    if delta < best_delta:
                        best_delta = delta
                        best_edge = (u, v)
                        best_new_c = new_c
            
            if best_edge is None or best_delta >= 0:
                no_improve += 1
                continue
            
            u, v = best_edge
            idx = _compute_index(n, u, v)
            
            removed = (adj[c][u] & adj[c][v]).bit_count()
            added = (adj[best_new_c][u] & adj[best_new_c][v]).bit_count()
            tri_count += added - removed
            
            colors[idx] = best_new_c
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            adj[best_new_c][u] |= (1 << v)
            adj[best_new_c][v] |= (1 << u)
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_result = colors[:]
                no_improve = 0
            else:
                no_improve += 1
    
    return best_result if best_tri == 0 else None

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
        
        # Try to find valid colorings for larger n
        targets = [50, 48, 46, 44, 42, 40, 38, 36, 34]
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 3:
                break
            
            # Allocate budget based on target
            if target_n >= 48:
                budget = min(remaining - 2, 45)
            elif target_n >= 44:
                budget = min(remaining - 2, 30)
            elif target_n >= 40:
                budget = min(remaining - 2, 20)
            else:
                budget = min(remaining - 2, 12)
            
            if budget < 2:
                continue
            
            # Try simulated annealing
            result = _sa_search(target_n, budget * 0.65)
            if result is not None:
                return {'n': target_n, 'colors': result}
            
            # Try local search with remaining time
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining > 3:
                result = _local_search(target_n, min(remaining - 2, budget * 0.35))
                if result is not None:
                    return {'n': target_n, 'colors': result}
        
        return best
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
