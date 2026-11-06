import time
import random
import sys
import math

_START_TIME = time.time()
_DEADLINE = 145.0

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

def _index_to_edge(n, idx):
    """Convert lexicographic edge index to (i, j) pair."""
    i = 0
    remaining = idx
    while i < n - 1:
        edges_from_i = n - 1 - i
        if remaining < edges_from_i:
            j = i + 1 + remaining
            return (i, j)
        remaining -= edges_from_i
        i += 1
    return (n - 2, n - 1)

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

def _deterministic_search(n, time_budget, seed=42):
    """
    Deterministic local search with fixed seed and optimized delta calculation.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    rng = random.Random(seed)
    
    best_tri_count = float('inf')
    best_colors = None
    
    # Try multiple restarts
    max_restarts = 1000
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with random coloring
        colors = [rng.randint(0, 3) for _ in range(m)]
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri_count:
            best_tri_count = tri_count
            best_colors = colors[:]
        
        if tri_count == 0:
            return colors
        
        # Local search loop
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while True:
            if time.time() - start_time > time_budget:
                break
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                return colors
            
            i, j, k, c = tri
            
            # Choose an edge to recolor
            edges = [(i, j), (i, k), (j, k)]
            best_edge = None
            best_new_c = -1
            best_delta = float('inf')
            
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
            
            if best_edge is None:
                break
            
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
            
            if tri_count < best_tri_count:
                best_tri_count = tri_count
                best_colors = colors[:]
            
            T *= cooling
    
    return best_colors if best_tri_count == 0 else None

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        # Try deterministic search for n=50
        if _time_left() > 10:
            result = _deterministic_search(50, _time_left() - 5, seed=42)
            if result is not None:
                return {'n': 50, 'colors': result}
        
        # Try n=45 if time remains
        if _time_left() > 5:
            result = _deterministic_search(45, _time_left() - 2, seed=123)
            if result is not None:
                return {'n': 45, 'colors': result}
        
        return best
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
