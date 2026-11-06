import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try n=51 first, then 50
    for n in [51, 50]:
        if time.time() > deadline - 10:
            break
            
        # Initialize random coloring
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Helper to get edge index
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
            
        # Precompute neighbors for each vertex and color?
        # Or just iterate.
        
        # Count mono triangles
        def count_mono_triangles(colors):
            count = 0
            # Build adjacency bitmasks for each color
            # adj[c][u] is a bitmask of v such that edge (u,v) has color c
            adj = [[0] * n for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_idx(i, j)]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            
            for c in range(4):
                # For each vertex u, count pairs (v, w) in adj[c][u]
                # such that v < w and edge (v, w) has color c
                # This is equivalent to counting triangles in the graph G_c
                # We can do this by iterating u, and for each v in adj[c][u] with v > u,
                # count common neighbors w > v?
                # Or simply: for each u, for each v in adj[c][u] with v > u,
                # common = adj[c][u] & adj[c][v]
                # count += popcount(common)
                # But this counts each triangle 3 times (once for each vertex).
                # So divide by 3 at the end? Or just count and accept factor.
                pass
            
            # Faster method:
            # For each color c, the number of mono triangles is the number of triangles in G_c.
            # We can compute this using matrix multiplication or simple loops.
            # Given n=50, simple loops are fine.
            total = 0
            for c in range(4):
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[c][u] >> v) & 1:
                            # Check common neighbors w > v
                            common = adj[c][u] & adj[c][v]
                            # Mask out bits <= v
                            mask = (1 << (n - v - 1)) - 1
                            common &= (mask << (v + 1))
                            total += bin(common).count("1")
            return total
            
        # This count is O(4 * n^3) which is 4 * 125000 = 500k ops.
        # Doing this every step is too slow.
        
        # Better: Local update.
        # When edge (u,v) changes color from c1 to c2:
        # Triangles affected are those containing (u,v).
        # These are triangles (u,v,w) for w in 0..n-1, w!=u,v.
        # For each w:
        #   Old state: edges (u,v), (u,w), (v,w).
        #   New state: edges (u,v) changed.
        
        # Let's implement the local search with local delta.
        
        # Initial count
        T = count_mono_triangles(colors)
        
        # Precompute edge indices for speed?
        # We'll use the edge_idx function.
        
        # Simulated Annealing
        T_current = T
        T_best = T
        colors_best = colors[:]
        
        temp = 1.0
        cooling = 0.99
        
        while time.time() < deadline - 5:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            idx = edge_idx(u, v)
            c_old = colors[idx]
            
            # Try all other colors
            best_delta = float('inf')
            best_new_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                
                # Calculate delta
                delta = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    # Get colors of (u,w) and (v,w)
                    idx_uw = edge_idx(u, w)
                    idx_vw = edge_idx(v, w)
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    
                    # Old triangle (u,v,w)
                    # Edges: c_old, c_uw, c_vw
                    if c_old == c_uw == c_vw:
                        delta -= 1 # Removed a mono triangle
                    
                    # New triangle (u,v,w)
                    # Edges: c_new, c_uw, c_vw
                    if c_new == c_uw == c_vw:
                        delta += 1 # Added a mono triangle
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c_new
            
            # Acceptance
            if best_delta < 0:
                accept = True
            else:
                if temp > 0:
                    accept = random.random() < math.exp(-best_delta / temp)
                else:
                    accept = False
            
            if accept and best_new_c != -1:
                colors[idx] = best_new_c
                T_current += best_delta
                
                if T_current < T_best:
                    T_best = T_current
                    colors_best = colors[:]
                    if T_best == 0:
                        return {'n': n, 'colors': colors_best}
            
            temp *= cooling
            
        if T_best == 0:
            return {'n': n, 'colors': colors_best}
    
    # If 50 or 51 failed, return the seed 32-vertex solution
    # Or try 32 again?
    # The seed is guaranteed to work.
    
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
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
