"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _exact_c3(h):
    """Compute exact C3 upper bound: max|conv| / (sum|h|)^2"""
    conv = np.convolve(h, h, mode='full')
    max_abs = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    if sum_abs < 1e-12:
        return np.inf
    return max_abs / (sum_abs ** 2)


def _discrete_local_search(h, max_rounds=50, time_limit=None):
    """
    Local search by flipping individual signs and block flips to reduce C3.
    h is a signed step function with values in {-1, 1} (or general signed values).
    """
    h = h.copy()
    current_c3 = _exact_c3(h)
    
    for round_idx in range(max_rounds):
        if time_limit is not None and time.time() > time_limit:
            break
        
        improved = False
        
        # Try flipping individual signs
        for i in range(N):
            if time_limit is not None and time.time() > time_limit:
                break
            
            h[i] = -h[i]
            new_c3 = _exact_c3(h)
            if new_c3 < current_c3:
                current_c3 = new_c3
                improved = True
            else:
                h[i] = -h[i]  # revert
        
        # Try block flips (flip contiguous blocks of size 1-20)
        block_sizes = [1, 2, 3, 5, 8, 12, 16, 20]
        for bs in block_sizes:
            if time_limit is not None and time.time() > time_limit:
                break
            for start in range(0, N - bs + 1, max(1, (N - bs) // 10)):
                if time_limit is not None and time.time() > time_limit:
                    break
                
                # Flip block [start, start+bs)
                h[start:start+bs] = -h[start:start+bs]
                new_c3 = _exact_c3(h)
                if new_c3 < current_c3:
                    current_c3 = new_c3
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]  # revert
        
        if not improved:
            break
    
    return h


def _structured_initializations():
    """Generate diverse structured initializations."""
    inits = []
    
    # All ones
    inits.append(np.ones(N))
    
    # All negative ones
    inits.append(-np.ones(N))
    
    # Alternating signs
    alt = np.ones(N)
    alt[1::2] = -1
    inits.append(alt)
    
    # Block patterns: various block sizes
    for block_size in [2, 4, 8, 10, 20, 40, 100]:
        h = np.ones(N)
        for start in range(0, N, block_size * 2):
            h[start:start+block_size] = -1
        inits.append(h.copy())
    
    # Random initializations
    np.random.seed(42)
    for _ in range(5):
        h = np.random.choice([-1, 1], size=N)
        inits.append(h)
    
    # Sparse positive pattern: mostly -1 with some +1
    for frac in [0.1, 0.25, 0.5]:
        h = np.ones(N)
        h[::2] = -1  # start with alternating
        # Randomly flip some to create structure
        mask = np.random.random(N) < frac
        h[mask] = 1
        inits.append(h)
    
    return inits


def _c3_upper_bound(initial_h):
    """
    Find a good signed step function by discrete local search from multiple initializations.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = initial_h.copy()
    best_c3 = _exact_c3(best_h)
    
    # Generate structured initializations
    inits = _structured_initializations()
    
    # Add the provided initial
    inits.append(initial_h)
    
    # Also add perturbed versions of the best found so far
    np.random.seed(42)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        h_work = init.copy()
        # Ensure we're working with signed values (normalize to ±1 if needed)
        abs_h = np.abs(h_work)
        if np.min(abs_h) > 1e-6:
            h_work = np.sign(h_work)
        
        # Run discrete local search
        h_refined = _discrete_local_search(h_work, max_rounds=30, time_limit=deadline)
        c3_val = _exact_c3(h_refined)
        
        if c3_val < best_c3:
            best_c3 = c3_val
            best_h = h_refined.copy()
    
    # Additional random restarts from best solution
    for _ in range(10):
        if time.time() > deadline:
            break
        
        # Perturb best solution
        h_pert = best_h.copy()
        # Flip a random fraction of entries
        n_flips = int(N * np.random.uniform(0.05, 0.3))
        flip_idx = np.random.choice(N, size=n_flips, replace=False)
        h_pert[flip_idx] = -h_pert[flip_idx]
        
        h_refined = _discrete_local_search(h_pert, max_rounds=20, time_limit=deadline)
        c3_val = _exact_c3(h_refined)
        
        if c3_val < best_c3:
            best_c3 = c3_val
            best_h = h_refined.copy()
    
    return best_h


def construct():
    # Start with all ones as initial
    h_init = np.ones(N)
    return _c3_upper_bound(h_init)
# EVOLVE-BLOCK-END
