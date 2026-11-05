import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140

    # GF(25) construction
    # Elements: (a, b) representing a + b*omega, where omega^2 = 2 mod 5
    # Tr(a + b*omega) = 2*a mod 5
    
    def gf25_sub(x, y):
        a1, b1 = x
        a2, b2 = y
        return ((a1 - a2) % 5, (b1 - b2) % 5)
    
    def gf25_square(x):
        a, b = x
        # (a + b*omega)^2 = a^2 + 2ab*omega + b^2*omega^2 = a^2 + 2b^2 + 2ab*omega
        real = (a * a + 2 * b * b) % 5
        imag = (2 * a * b) % 5
        return (real, imag)
    
    def gf25_trace(x):
        a, b = x
        return (2 * a) % 5
    
    # Build all 25 elements of GF(25)
    gf25_elements = [(a, b) for a in range(5) for b in range(5)]
    
    # Vertex 0..24: elements of GF(25), vertex 25: infinity
    n = 26
    colors = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if i < 25 and j < 25:
                # Both in GF(25)
                diff = gf25_sub(gf25_elements[j], gf25_elements[i])
                sq = gf25_square(diff)
                c = gf25_trace(sq)
            else:
                # One is infinity (vertex 25)
                # Color based on the GF(25) element
                elem = gf25_elements[i if i == 25 else j]
                c = elem[0]  # Use the 'a' coefficient
            colors.append(c)
    
    # Verify
    def count_violations(n, colors):
        m = n * (n - 1) // 2
        total = 0
        for c in range(5):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i] & adj[j]) >= 2:
                        total += 1
        return total
    
    viol = count_violations(n, colors)
    
    if viol == 0:
        return {'n': n, 'colors': colors}
    
    # If the algebraic construction doesn't work, fall back to local search
    def count_violations_fast(n, colors):
        m = n * (n - 1) // 2
        if m == 0:
            return 0
        colors_arr = np.array(colors, dtype=np.int8)
        total = 0
        adj_all = np.zeros((5, n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors_arr[idx]
                adj_all[c, i, j] = 1
                adj_all[c, j, i] = 1
                idx += 1
        for c in range(5):
            common = adj_all[c] @ adj_all[c]
            np.fill_diagonal(common, 0)
            iu = np.triu_indices(n, 1)
            total += int(np.sum(common[iu] >= 2))
        return total

    def fast_local_search(n, colors, max_iter=5000, seed=None, deadline=None):
        rng = random.Random(seed)
        colors = list(colors)
        m = n * (n - 1) // 2
        best = colors[:]
        best_viol = count_violations_fast(n, best)
        
        for it in range(max_iter):
            if best_viol == 0:
                break
            if deadline and time.time() > deadline - 5:
                break
            idx = rng.randint(0, m - 1)
            old = colors[idx]
            new = rng.randint(0, 4)
            if new == old:
                continue
            colors[idx] = new
            v = count_violations_fast(n, colors)
            if v < best_viol:
                best = colors[:]
                best_viol = v
            else:
                colors[idx] = old
        return best, best_viol

    best_n = 1
    best_colors = []

    # Try to fix the algebraic construction with local search
    colors_lw = colors[:]
    colors_lw, viol = fast_local_search(26, colors_lw, max_iter=10000, seed=42, deadline=deadline)
    if viol == 0:
        return {'n': 26, 'colors': colors_lw}
    
    # Try smaller n
    for target_n in range(25, 10, -1):
        if time.time() > deadline - 10:
            break
        # Use truncated algebraic construction
        colors_sub = colors[:target_n*(target_n-1)//2]
        colors_sub, viol = fast_local_search(target_n, colors_sub, max_iter=5000, seed=42, deadline=deadline)
        if viol == 0:
            return {'n': target_n, 'colors': colors_sub}
    
    # Fallback: structured init + local search
    def structured_init(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % 5
                colors.append(d)
        return colors

    for target_n in range(20, 10, -1):
        if time.time() > deadline - 10:
            break
        colors_init = structured_init(target_n)
        colors_init, viol = fast_local_search(target_n, colors_init, max_iter=5000, seed=42, deadline=deadline)
        if viol == 0:
            return {'n': target_n, 'colors': colors_init}
    
    # Last resort
    for target_n in range(10, 1, -1):
        colors_init = structured_init(target_n)
        viol = count_violations_fast(target_n, colors_init)
        if viol == 0:
            return {'n': target_n, 'colors': colors_init}
    
    return {'n': 1, 'colors': []}
