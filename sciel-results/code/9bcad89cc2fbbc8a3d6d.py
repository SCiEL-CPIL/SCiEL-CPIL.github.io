import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 115
    
    # Grid for local search
    M = 15000
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    Z = np.exp(1j * thetas[:, None] * np.arange(N)[None, :])
    
    best_c = None
    best_score = np.inf
    best_raw = np.inf  # raw max |P(z)|, score = raw / sqrt(24)
    
    def calculate_raw_score(c):
        vals = Z @ c
        return np.max(np.abs(vals))
    
    def local_search(c_init, max_iters=200):
        c = c_init.astype(float).copy()
        vals = Z @ c
        current_raw = np.max(np.abs(vals))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping each bit, in a fixed order
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = c[i]
                new_val = -old_val
                c[i] = new_val
                
                vals_new = vals + 2 * new_val * Z[:, i]
                new_raw = np.max(np.abs(vals_new))
                
                if new_raw < current_raw - 1e-12:
                    current_raw = new_raw
                    vals = vals_new
                    improved = True
                else:
                    c[i] = old_val
            
            if not improved:
                break
                
        return c, current_raw

    # Structured seeds
    seeds = []
    
    # Rudin-Shapiro
    r = [1, 1]
    for i in range(2, N):
        if i % 2 == 0:
            r.append(r[i // 2])
        else:
            r.append(-r[(i - 1) // 2])
    seeds.append(np.array(r, dtype=float))
    
    # Thue-Morse
    tm = np.array([(-1) ** bin(i).count('1') for i in range(N)], dtype=float)
    seeds.append(tm)
    
    # Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float)
    seeds.append(alt)
    
    # All ones
    seeds.append(np.ones(N, dtype=float))
    
    # All minus
    seeds.append(-np.ones(N, dtype=float))
    
    # Block patterns: +++---+++---+++---
    block = np.array([1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1], dtype=float)
    seeds.append(block)
    
    # Reverse patterns
    for s in seeds[:]:
        seeds.append(s[::-1].copy())
    
    # Add random seeds
    np.random.seed(42)
    for _ in range(30):
        if time.time() > deadline:
            break
        seeds.append(np.random.choice([-1, 1], N).astype(float))
    
    # Process all seeds
    for seed in seeds:
        if time.time() > deadline:
            break
        
        c_res, raw_score = local_search(seed)
        
        if raw_score < best_raw:
            best_raw = raw_score
            best_c = c_res.copy()
    
    # If we have time, do more random restarts
    if time.time() < deadline - 10:
        extra_restarts = 0
        while time.time() < deadline - 5:
            c_rand = np.random.choice([-1, 1], N).astype(float)
            c_res, raw_score = local_search(c_rand, max_iters=100)
            if raw_score < best_raw:
                best_raw = raw_score
                best_c = c_res.copy()
            extra_restarts += 1
            if extra_restarts > 200:
                break
    
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
