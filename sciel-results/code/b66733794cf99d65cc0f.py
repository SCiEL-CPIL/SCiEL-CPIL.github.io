import sys
sys.setrecursionlimit(10000)

def construct():
    N = 50
    E = N * (N - 1) // 2
    
    # Precompute edge indices
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            idx += 1
            
    # Precompute triangles
    # triangle[e] = list of (e1, e2) such that (e, e1, e2) form a triangle
    # This is expensive to precompute fully, but we can check on the fly.
    
    # Backtracking
    colors = [-1] * E
    
    # Order edges by some heuristic? 
    # Let's just iterate in lexicographic order.
    # This is likely to fail for N=50.
    
    # Better: Use a known deterministic construction.
    # The construction for n=50 is based on the Paley graph of order 53.
    # Vertices 0..49.
    # Color c(u,v) = 0 if (u-v)^2 is a quadratic residue mod 53
    # This gives a 2-coloring.
    # To get 4 colors, we can use the Paley graph of order 16? No.
    
    # Actually, there is a 4-coloring for n=50 using the following:
    # c(u,v) = ( (u-v)^2 % 53 ) % 4 ? No.
    
    # I will use a deterministic version of MinConflicts (which is a local search, not backtracking).
    # But the directive says "do not reuse the current designs".
    # MinConflicts is a specific design.
    # Simulated Annealing is different.
    # I will implement Simulated Annealing.
    
    import random
    import numpy as np
    
    random.seed(123)
    
    # Initialize
    colors = np.random.randint(0, 4, size=E)
    
    # Precompute us, vs
    us = np.zeros(E, dtype=int)
    vs = np.zeros(E, dtype=int)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            us[idx] = i
            vs[idx] = j
            idx += 1
            
    # Precompute edge_idx matrix
    edge_idx_mat = np.full((N, N), -1, dtype=int)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx_mat[i, j] = idx
            edge_idx_mat[j, i] = idx
            idx += 1
            
    # Calculate initial conflicts
    conflict_count = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            e1 = edge_idx_mat[i, j]
            c1 = colors[e1]
            count = 0
            for k in range(N):
                if k == i or k == j: continue
                e2 = edge_idx_mat[i, k]
                e3 = edge_idx_mat[j, k]
                if colors[e2] == c1 and colors[e3] == c1:
                    count += 1
            conflict_count[e1] = count
            
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    for step in range(50000):
        if np.sum(conflict_count) == 0:
            break
            
        e = np.random.randint(E)
        if conflict_count[e] == 0:
            continue
            
        u = us[e]
        v = vs[e]
        c_old = colors[e]
        
        # Pick a new color
        c_new = np.random.randint(0, 4)
        if c_new == c_old:
            continue
            
        # Calculate delta
        delta = 0
        for w in range(N):
            if w == u or w == v:
                continue
            e1 = edge_idx_mat[u, w]
            e2 = edge_idx_mat[v, w]
            c1 = colors[e1]
            c2 = colors[e2]
            
            was_mono = (c1 == c_old and c2 == c_old)
            is_mono = (c1 == c_new and c2 == c_new)
            
            if was_mono and not is_mono:
                delta -= 3
            elif not was_mono and is_mono:
                delta += 3
                
        # Accept if delta < 0 or with probability exp(-delta/T)
        if delta < 0:
            accept = True
        else:
            accept = (random.random() < np.exp(-delta / T))
            
        if accept:
            colors[e] = c_new
            for w in range(N):
                if w == u or w == v:
                    continue
                e1 = edge_idx_mat[u, w]
                e2 = edge_idx_mat[v, w]
                c1 = colors[e1]
                c2 = colors[e2]
                
                was_mono = (c1 == c_old and c2 == c_old)
                is_mono = (c1 == c_new and c2 == c_new)
                
                if was_mono and not is_mono:
                    conflict_count[e] -= 1
                    conflict_count[e1] -= 1
                    conflict_count[e2] -= 1
                elif not was_mono and is_mono:
                    conflict_count[e] += 1
                    conflict_count[e1] += 1
                    conflict_count[e2] += 1
                    
        T *= cooling
        
    if np.sum(conflict_count) == 0:
        return {'n': N, 'colors': [int(c) for c in colors]}
    else:
        # Fallback to 32
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
