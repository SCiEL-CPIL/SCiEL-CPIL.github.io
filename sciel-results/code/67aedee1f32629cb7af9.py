import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute the ratio: (L2 norm squared) / (L1 norm * sup norm of autoconvolution)
    for a vector of nonnegative heights."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    # Compute autoconvolution
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)  # This is 1.0
    
    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Start with a single spike as baseline
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = 1.0
    
    # Try simple 2-point configurations
    for i in range(N):
        for j in range(i+1, min(i+5, N)):
            for a in np.linspace(0.01, 0.99, 100):
                h = np.zeros(N)
                h[i] = a
                h[j] = 1 - a
                sc = compute_ratio(h)
                update_best(h, sc)
    
    # Try 3-point configurations
    for i in range(N):
        for j in range(i+1, min(i+5, N)):
            for k in range(j+1, min(j+5, N)):
                for a in np.linspace(0.01, 0.6, 50):
                    for b in np.linspace(0.01, 0.6, 50):
                        if a + b < 0.99:
                            c = 1 - a - b
                            h = np.zeros(N)
                            h[i] = a
                            h[j] = b
                            h[k] = c
                            sc = compute_ratio(h)
                            update_best(h, sc)
    
    # Local search on sparse configurations
    def local_search(h_start, max_iter=500):
        h = h_start.copy()
        current_score = compute_ratio(h)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try swapping weight between two positions
            nonzero_indices = np.where(h > 1e-10)[0]
            if len(nonzero_indices) < 2:
                break
                
            # Randomly pick two positions to adjust
            idx1, idx2 = np.random.choice(len(nonzero_indices), 2, replace=False)
            i, j = nonzero_indices[idx1], nonzero_indices[idx2]
            
            # Try moving some weight from i to j
            for frac in np.linspace(0.01, 0.5, 10):
                move = h[i] * frac
                if h[j] + move > 0 and h[i] - move > 0:
                    h_new = h.copy()
                    h_new[i] -= move
                    h_new[j] += move
                    sc = compute_ratio(h_new)
                    if sc > current_score + 1e-10:
                        h = h_new
                        current_score = sc
                        improved = True
                        break
            
            # Try adding weight to a new position
            if not improved:
                zero_indices = np.where(h < 1e-10)[0]
                if len(zero_indices) > 0:
                    new_idx = np.random.choice(zero_indices)
                    src_idx = nonzero_indices[np.random.randint(len(nonzero_indices))]
                    for frac in np.linspace(0.01, 0.3, 10):
                        move = h[src_idx] * frac
                        h_new = h.copy()
                        h_new[src_idx] -= move
                        h_new[new_idx] += move
                        sc = compute_ratio(h_new)
                        if sc > current_score + 1e-10:
                            h = h_new
                            current_score = sc
                            improved = True
                            break
            
            if not improved:
                break
        
        return h, current_score
    
    # Run local search from best found so far
    h_opt, sc_opt = local_search(best_h, 1000)
    update_best(h_opt, sc_opt)
    
    # Run from some random sparse starts
    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        # Random sparse 3-4 point start
        n_points = np.random.randint(3, 5)
        positions = np.random.choice(N, n_points, replace=False)
        h_start = np.zeros(N)
        h_start[positions] = np.random.dirichlet(np.ones(n_points))
        h_opt, sc_opt = local_search(h_start, 300)
        update_best(h_opt, sc_opt)
    
    return list(best_h)
