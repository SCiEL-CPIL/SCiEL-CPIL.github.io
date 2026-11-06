import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # 40s margin for safety and overhead
    
    # Target n
    n = 50
    
    # Precompute edge indices for fast access?
    # Or just use a 2D array for colors
    # colors[i, j] is the color of edge (i, j)
    
    def run_local_search():
        # Initialize random coloring
        c = np.random.randint(0, 4, size=(n, n), dtype=np.int8)
        # Make symmetric
        c = np.triu(c, 1) + np.triu(c, 1).T
        # Diagonal is 0, we don't care about it
        
        # Initialize N[u, v, k]
        # N[u, v, k] = count of w != u, v such that c[u, w] == k and c[v, w] == k
        N = np.zeros((n, n, 4), dtype=np.int32)
        
        # Compute initial N
        # Vectorized approach might be complex due to dependencies, loop is fine for n=50
        for u in range(n):
            for v in range(u + 1, n):
                # For each k, count w
                # c[u, :] and c[v, :]
                # mask = (c[u, :] == k) & (c[v, :] == k)
                # Exclude u and v
                for k in range(4):
                    count = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if c[u, w] == k and c[v, w] == k:
                            count += 1
                    N[u, v, k] = count
                    N[v, u, k] = count
        
        # Calculate initial T (total monochromatic triangles)
        # T = 1/3 * sum_{u<v} N[u, v, c[u, v]]
        T = 0
        for u in range(n):
            for v in range(u + 1, n):
                T += N[u, v, c[u, v]]
        T //= 3
        
        if T == 0:
            return c
        
        # Local Search
        # We will use a simple greedy descent with occasional random moves (simulated annealing-like)
        # But since T is integer and we want T=0, greedy is strong.
        
        steps = 0
        max_steps_per_iter = 20000
        stuck_count = 0
        
        while time.time() < deadline:
            for _ in range(max_steps_per_iter):
                if time.time() > deadline:
                    return c
                
                if T == 0:
                    return c
                
                # Pick random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                while v == u:
                    v = random.randint(0, n - 1)
                
                c_old = c[u, v]
                
                # Find best new color
                # delta_T for color k: (N[u, v, k] - N[u, v, c_old]) / 3
                # We want to minimize T, so minimize N[u, v, k]
                min_N = N[u, v, :].min()
                
                if min_N < N[u, v, c_old]:
                    # Improvement available
                    candidates = np.where(N[u, v, :] == min_N)[0]
                    c_new = random.choice(candidates)
                else:
                    # No improvement, try random move to escape local minimum
                    # Accept with low probability or just skip
                    if random.random() < 0.05:
                        c_new = random.randint(0, 3)
                        if c_new == c_old:
                            continue
                    else:
                        continue
                
                if c_new == c_old:
                    continue
                
                # Update T
                # T += (N[u, v, c_new] - N[u, v, c_old]) // 3
                # Note: (N_new - N_old) is divisible by 3?
                # Yes, because T is integer.
                T += (N[u, v, c_new] - N[u, v, c_old]) // 3
                
                # Update N
                # For each w != u, v:
                # N[u, w, k] depends on c[u, z] and c[w, z].
                # If z=v, c[u, v] changed.
                # So N[u, w, k] changes if c[w, v] == k.
                # Specifically:
                # If c_old == c[w, v], N[u, w, c_old] -= 1
                # If c_new == c[w, v], N[u, w, c_new] += 1
                # Similarly for N[v, w, k] based on c[u, w]
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    c_wv = c[w, v]
                    c_uw = c[u, w]
                    
                    # Update N[u, w]
                    if c_old == c_wv:
                        N[u, w, c_old] -= 1
                        N[w, u, c_old] -= 1
                    if c_new == c_wv:
                        N[u, w, c_new] += 1
                        N[w, u, c_new] += 1
                        
                    # Update N[v, w]
                    if c_old == c_uw:
                        N[v, w, c_old] -= 1
                        N[w, v, c_old] -= 1
                    if c_new == c_uw:
                        N[v, w, c_new] += 1
                        N[w, v, c_new] += 1
                        
                # Update color
                c[u, v] = c_new
                c[v, u] = c_new
                
                steps += 1
                
            # If we didn't find T=0 in this batch, maybe restart?
            # For now, just keep going.
            
        return c

    best_c = None
    # Run multiple restarts to increase probability of finding T=0
    while time.time() < deadline:
        c = run_local_search()
        # Verify if T=0
        # Quick check
        is_valid = True
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if c[i, j] == c[j, k] == c[i, k]:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
        
        if is_valid:
            # Convert to required format
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(int(c[i, j]))
            return {'n': n, 'colors': colors}
            
    # If no valid coloring found for n=50, return n=32 seed
    # (Though n=50 should be found)
    
    # Fallback: Seed 32
    def get_seed_32():
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
        
    return get_seed_32()
