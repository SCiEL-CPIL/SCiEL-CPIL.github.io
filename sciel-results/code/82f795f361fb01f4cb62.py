"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    # Seed solution
    h_seed = np.ones(N)
    best_h = h_seed.copy()
    best_val = 2.0
    
    def eval_val(h):
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2 * N * np.max(conv) / (s ** 2))

    # Check seed
    v_seed = eval_val(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed

    # Try some simple structured vectors
    # 1. Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    if np.sum(h_alt) != 0:
        v_alt = eval_val(h_alt)
        if v_alt < best_val:
            best_val = v_alt
            best_h = h_alt.copy()

    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        # Gaussian
        h_rand = rng.standard_normal(N)
        if np.sum(h_rand) != 0:
            v_rand = eval_val(h_rand)
            if v_rand < best_val:
                best_val = v_rand
                best_h = h_rand.copy()
        
        # Sparse
        h_sparse = rng.standard_normal(N)
        mask = rng.random(N) < 0.5
        h_sparse[~mask] = 0
        if np.sum(h_sparse) != 0:
            v_sp = eval_val(h_sparse)
            if v_sp < best_val:
                best_val = v_sp
                best_h = h_sparse.copy()

    # Local refinement using gradient descent on the value
    # Value is non-differentiable due to max and abs, but we can use subgradients or smooth approximations
    # Given the theoretical bound of 2, and seed achieving 2, 
    # any perturbation likely increases the value.
    # However, we perform a few steps of coordinate descent just in case.
    
    h = best_h.copy()
    lr = 0.01
    for i in range(1000):
        if time.time() > deadline:
            break
        for j in range(N):
            if time.time() > deadline:
                break
            old_h = h[j]
            # Try small perturbations
            for delta in [-0.1, 0.1, -0.01, 0.01]:
                h[j] = old_h + delta
                v = eval_val(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
                else:
                    h[j] = old_h
            h[j] = old_h

    return best_h
# EVOLVE-BLOCK-END
