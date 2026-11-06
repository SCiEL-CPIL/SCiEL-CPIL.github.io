import numpy as np
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate the seed n=32 solution
    def get_seed_solution():
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
        return 32, colors

    n_seed, seed_colors = get_seed_solution()
    
    # Fallback solution
    fallback = {'n': n_seed, 'colors': seed_colors}
    
    # 2. Setup n=50
    N = 50
    # Initialize color matrix
    # C[i, j] is the color of edge (i, j)
    C = np.zeros((N, N), dtype=np.int8)
    
    # Fill seed part
    # The seed colors are in lexicographic order for i < j
    idx = 0
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            C[i, j] = seed_colors[idx]
            C[j, i] = seed_colors[idx]
            idx += 1
            
    # Fill rest randomly
    for i in range(N):
        for j in range(i + 1, N):
            if i < n_seed and j < n_seed:
                continue
            c = random.randint(0, 3)
            C[i, j] = c
            C[j, i] = c
            
    # 3. Simulated Annealing
    # Energy: number of monochromatic triangles
    # Calculate initial energy
    # E = sum over all triangles (i, j, k) of I(c_ij == c_jk == c_ik)
    # We can calculate this by summing over edges and checking triangles?
    # Or just iterate all triples. N=50, triples = 19600.
    # This is fast.
    
    def calc_energy(C):
        E = 0
        for i in range(N):
            for j in range(i + 1, N):
                c_ij = C[i, j]
                for k in range(j + 1, N):
                    if C[j, k] == c_ij and C[i, k] == c_ij:
                        E += 1
        return E

    E = calc_energy(C)
    
    if E == 0:
        # Convert to list format
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                colors.append(int(C[i, j]))
        return {'n': N, 'colors': colors}

    T = 1.0
    T_min = 1e-5
    alpha = 0.999
    
    # Precompute neighbors for speed? No, vectorized delta is better.
    
    while time.time() < deadline:
        # Pick random edge
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        c_old = C[u, v]
        c_new = random.randint(0, 3)
        if c_new == c_old:
            continue
            
        # Calculate delta E
        # Triangles (u, v, w)
        # w != u, w != v
        c_uw = C[u, :]
        c_vw = C[v, :]
        
        # Masks for w
        # We need to check all w
        # was_mono: c_old == c_uw[w] == c_vw[w]
        # is_mono: c_new == c_uw[w] == c_vw[w]
        
        # Vectorized
        was_mono = (c_uw == c_old) & (c_vw == c_old)
        is_mono = (c_uw == c_new) & (c_vw == c_new)
        
        # Exclude u and v from w
        was_mono[u] = False
        was_mono[v] = False
        is_mono[u] = False
        is_mono[v] = False
        
        dE = np.sum(is_mono) - np.sum(was_mono)
        
        if dE <= 0 or random.random() < np.exp(-dE / T):
            C[u, v] = c_new
            C[v, u] = c_new
            E += dE
            
            if E == 0:
                colors = []
                for i in range(N):
                    for j in range(i + 1, N):
                        colors.append(int(C[i, j]))
                return {'n': N, 'colors': colors}
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart? Or just keep going.
            # If stuck, maybe randomize some edges?
            # For now, just continue.
            
    # If not found, return fallback
    return fallback
