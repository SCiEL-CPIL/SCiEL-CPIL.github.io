import numpy as np
import math

def construct():
    N = 50
    E = N * (N - 1) // 2
    
    # Precompute edge indices and endpoints
    edge_idx = np.full((N, N), -1, dtype=int)
    us = np.zeros(E, dtype=int)
    vs = np.zeros(E, dtype=int)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i, j] = idx
            edge_idx[j, i] = idx
            us[idx] = i
            vs[idx] = j
            idx += 1
            
    # Initialize colors
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 4, size=E)
    
    # Calculate initial conflict counts
    # conflict_count[e] is number of w such that (u,v,w) is monochromatic with color c(e)
    conflict_count = np.zeros(E, dtype=int)
    
    # Efficient initial calculation
    # For each triangle (i,j,k), check if mono.
    # If mono with color c, increment conflict_count for edges (i,j), (i,k), (j,k).
    # We can iterate over all triangles.
    # Number of triangles is N*(N-1)*(N-2)/6 = 50*49*48/6 = 19600.
    # This is very fast.
    
    # To speed up, we can use the fact that if c(i,j)==c(i,k)==c(j,k), then all 3 edges gain 1 conflict.
    # We can iterate i, j, k.
    # But Python loops are slow.
    # Let's use the vectorized approach or just the O(N^3) loop which is ~20k iterations.
    
    for i in range(N):
        for j in range(i + 1, N):
            e1 = edge_idx[i, j]
            c1 = colors[e1]
            for k in range(j + 1, N):
                e2 = edge_idx[i, k]
                e3 = edge_idx[j, k]
                if colors[e2] == c1 and colors[e3] == c1:
                    conflict_count[e1] += 1
                    conflict_count[e2] += 1
                    conflict_count[e3] += 1
                # Note: we only check if all 3 match c1.
                # What if they match a different color?
                # If c(i,j)=0, c(i,k)=1, c(j,k)=1. Not mono.
                # If c(i,j)=0, c(i,k)=0, c(j,k)=1. Not mono.
                # If c(i,j)=0, c(i,k)=0, c(j,k)=0. Mono with 0.
                # So checking against c1 is sufficient to find *any* mono triangle?
                # No. If c(i,j)=0, c(i,k)=1, c(j,k)=1, it's not mono.
                # If c(i,j)=1, c(i,k)=1, c(j,k)=1, it's mono with 1.
                # In my loop, I pick e1=(i,j). I check if e2, e3 match c1.
                # If they do, it's a mono triangle.
                # This covers all mono triangles exactly once?
                # Yes, for each triangle, exactly one edge is "first" in lex order?
                # No, we iterate i<j<k. e1=(i,j), e2=(i,k), e3=(j,k).
                # We check if c(e1)==c(e2)==c(e3).
                # This identifies all mono triangles.
                
    total_conflicts = np.sum(conflict_count)
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    max_iters = 200000
    # We want to finish in < 120s.
    # Each iter is ~O(N) = 50 ops.
    # 200k * 50 = 10M ops. Should be fast in Python/NumPy mix.
    
    for step in range(max_iters):
        if total_conflicts == 0:
            break
            
        # Pick a random edge
        e = rng.integers(E)
        u = us[e]
        v = vs[e]
        c_old = colors[e]
        
        # Pick a new color
        c_new = rng.integers(0, 4)
        if c_new == c_old:
            continue
            
        # Calculate delta in total conflicts
        # Only triangles containing e are affected.
        # Triangles are (u, v, w) for w != u, v.
        delta = 0
        for w in range(N):
            if w == u or w == v:
                continue
            e1 = edge_idx[u, w]
            e2 = edge_idx[v, w]
            c1 = colors[e1]
            c2 = colors[e2]
            
            # Current status
            was_mono = (c1 == c_old and c2 == c_old)
            # New status
            is_mono = (c1 == c_new and c2 == c_new)
            
            if was_mono and not is_mono:
                delta -= 3
            elif not was_mono and is_mono:
                delta += 3
                
        # Acceptance criterion
        if delta <= 0:
            # Accept improvement
            colors[e] = c_new
            total_conflicts += delta
            # Update conflict counts
            for w in range(N):
                if w == u or w == v:
                    continue
                e1 = edge_idx[u, w]
                e2 = edge_idx[v, w]
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
        else:
            # Accept worse move with probability exp(-delta / T)
            if T > T_min:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    colors[e] = c_new
                    total_conflicts += delta
                    for w in range(N):
                        if w == u or w == v:
                            continue
                        e1 = edge_idx[u, w]
                        e2 = edge_idx[v, w]
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
            # Cool down
            T *= alpha
            
    if total_conflicts == 0:
        return {'n': N, 'colors': [int(c) for c in colors]}
    else:
        # Fallback to 32-vertex solution if SA fails
        # This ensures we don't score 0.
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
