"""Greedy vertex-by-vertex construction + local search for R(3,3,3,3) colourings."""
import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin under the 120 s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32 (guaranteed valid floor)."""
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
    return colors

def _compute_index(n, i, j):
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
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

def _greedy_vertex_extension(n_target, time_budget, seed=None):
    """
    Build a coloring incrementally by adding vertices one at a time.
    For each new vertex, choose colors for edges to existing vertices
    to minimize the number of new monochromatic triangles created.
    """
    if seed is not None:
        random.seed(seed)
    
    m_target = n_target * (n_target - 1) // 2
    colors = [0] * m_target
    
    # Start with a valid K32 coloring
    colors_32 = _seed_32()
    m_32 = 32 * 31 // 2
    
    # Copy K32 coloring into the larger array
    for i in range(32):
        for j in range(i + 1, 32):
            colors[_compute_index(n_target, i, j)] = colors_32[i * 31 - i * (i + 1) // 2 + (j - i - 1)]
    
    # Build adjacency for K32
    adj = [[0] * n_target for _ in range(4)]
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c = colors_32[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    start_time = time.time()
    
    # Add vertices 32 to n_target-1
    for v in range(32, n_target):
        if time.time() - start_time > time_budget or _time_left() < 2.0:
            break
        
        # For each existing vertex u < v, choose a color for edge (u, v)
        # to minimize new monochromatic triangles
        assignment = [0] * v
        
        # Greedy: for each u, pick the color that creates fewest new triangles
        for u in range(v):
            best_c = 0
            best_cost = float('inf')
            
            for c in range(4):
                # If we color (u, v) with color c, how many new mono-triangles?
                # A triangle (w, u, v) is mono-color c if (w, u) is color c AND (w, v) is color c
                # But (w, v) hasn't been assigned yet for w < u
                # For w > u, (w, v) will be assigned later
                # So we only count triangles where all three edges are already determined
                # Actually, we need to think about this differently
                
                # When we assign color c to (u, v), we create a new mono-triangle
                # for each w < u such that (w, u) is color c AND (w, v) is color c
                # But (w, v) for w < u hasn't been assigned yet in this loop
                # So we can't directly count this
                
                # Alternative: count potential conflicts
                # For each w < u, if (w, u) is color c, then when we later assign (w, v),
                # we'll need to avoid color c to not create a triangle
                # This creates a constraint on future assignments
                
                # Simpler heuristic: count how many neighbors of u in color c
                # This is a proxy for "how constrained will this be"
                neighbors_in_c = (adj[c][u] & ((1 << u) - 1)).bit_count()
                
                # Also consider: how many of these neighbors already have edges to v?
                # No, v is new, so no edges to v exist yet
                
                # Use a simpler cost: number of neighbors in color c
                # Lower is better (fewer constraints)
                cost = neighbors_in_c
                
                if cost < best_cost:
                    best_cost = cost
                    best_c = c
            
            assignment[u] = best_c
        
        # Apply the assignment
        for u in range(v):
            c = assignment[u]
            idx = _compute_index(n_target, u, v)
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
    
    return colors

def _local_search_refine(n, colors, time_budget):
    """Refine a coloring using local search to eliminate remaining monochromatic triangles."""
    m = n * (n - 1) // 2
    start_time = time.time()
    
    idx_to_edge = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    adj = _build_adj(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    
    if tri_count == 0:
        return colors, 0
    
    best_tri = tri_count
    best_colors = colors[:]
    
    # Simulated annealing
    T = 3.0
    T_min = 0.1
    cool_rate = 0.999
    
    for _ in range(50000):
        if time.time() - start_time > time_budget or _time_left() < 1.0:
            break
        
        ri = random.randint(0, m - 1)
        old_c = colors[ri]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
        
        u, v = idx_to_edge[ri]
        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
        delta = added - removed
        
        if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
            colors[ri] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            tri_count += delta
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                if tri_count == 0:
                    return colors, 0
        T *= cool_rate
    
    return best_colors, best_tri

def construct():
    # Guaranteed floor: K32 valid coloring.
    floor_colors = _seed_32()
    floor_n = 32
    
    # Try greedy vertex-by-vertex extension for larger n
    # The known record is n=50, so try up to 50
    
    for n_target in (35, 40, 45, 50):
        if _time_left() < 15.0:
            break
        budget = min(12.0, _time_left() - 5.0)
        if budget <= 0:
            break
        
        # Try multiple seeds for diversity
        for seed in (42, 123, 456):
            if _time_left() < 5.0:
                break
            
            colors = _greedy_vertex_extension(n_target, time_budget=budget / 3, seed=seed)
            
            # Verify and refine
            adj = _build_adj(colors, n_target)
            tri = _count_mono_triangles(adj, n_target)
            
            if tri == 0:
                return {'n': n_target, 'colors': colors}
            
            # If not perfect, try local search to fix
            if tri < 50:  # Only refine if close
                refined, refined_tri = _local_search_refine(n_target, colors, time_budget=5.0)
                if refined_tri == 0:
                    return {'n': n_target, 'colors': refined}
    
    # Fallback: try direct local search from random init
    for n in (35, 40, 45):
        if _time_left() < 8.0:
            break
        budget = min(6.0, _time_left() - 3.0)
        if budget <= 0:
            break
        
        m = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        refined, tri = _local_search_refine(n, colors, time_budget=budget)
        
        if tri == 0:
            return {'n': n, 'colors': refined}
    
    return {'n': floor_n, 'colors': floor_colors}
