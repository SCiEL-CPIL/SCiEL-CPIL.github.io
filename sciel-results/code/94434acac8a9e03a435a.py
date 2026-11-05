import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def compute_val(h_vec):
        s = np.sum(h_vec)
        if abs(s) < 1e-10:
            return np.inf
        g = np.convolve(h_vec, h_vec)
        return abs(2 * N * np.max(g) / (s ** 2))

    def refine(h_init, max_iters=5000, step_size=0.01):
        h = h_init.copy()
        for i in range(max_iters):
            if time.time() > deadline:
                break
            g = np.convolve(h, h)
            max_val = np.max(g)
            k_star = np.argmax(g)
            
            v = np.zeros(N)
            start_idx = max(0, k_star - N + 1)
            end_idx = min(N, k_star + 1)
            if end_idx > start_idx:
                v[start_idx:end_idx] = 2 * h[k_star : k_star - (end_idx - start_idx) - 1 : -1]
            
            h_new = h - step_size * v
            sum_h = np.sum(h_new)
            if abs(sum_h) < 1e-10:
                h_new = h
                sum_h = 1.0
            h = h_new / sum_h
            
        return h

    best_h = np.ones(N)
    best_val = compute_val(best_h)
    
    # Initial refinement from seed
    h_ref = refine(best_h, max_iters=2000, step_size=0.05)
    val = compute_val(h_ref)
    if val < best_val:
        best_val = val
        best_h = h_ref.copy()
        
    # Random restarts
    rng = np.random.default_rng(123)
    
    # Allocate remaining time for restarts
    # Each restart takes ~0.5s for 2000 iters?
    # Let's do 10 restarts with 1000 iters each.
    
    for r in range(20):
        if time.time() > deadline - 10.0:
            break
            
        h_rand = rng.normal(0, 1, N)
        # Ensure sum is not zero
        if abs(np.sum(h_rand)) < 1e-6:
            h_rand[0] += 1.0
        h_rand = h_rand / np.sum(h_rand)
        
        # Coarse refinement
        h_cur = refine(h_rand, max_iters=1000, step_size=0.05)
        val = compute_val(h_cur)
        if val < best_val:
            best_val = val
            best_h = h_cur.copy()
            # Fine polish
            if time.time() < deadline - 5.0:
                h_pol = refine(best_h, max_iters=2000, step_size=0.01)
                val_pol = compute_val(h_pol)
                if val_pol < best_val:
                    best_val = val_pol
                    best_h = h_pol.copy()

    return best_h
