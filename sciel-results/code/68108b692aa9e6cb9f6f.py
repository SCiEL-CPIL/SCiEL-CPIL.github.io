import numpy as np
import time

N = 100

def _max_overlap(h):
    """
    Compute the maximum overlap C_k for linear shifts k=1..N-1.
    C_k = sum_{i=0}^{N-1-k} h_i * (1 - h_{i+k})
    """
    max_c = 0.0
    # Use vectorized operations for speed
    # We can compute all C_k efficiently?
    # C_k is the cross-correlation of h and 1-h, but truncated.
    # For N=100, a loop is fine.
    for k in range(1, N):
        # h_i for i in 0..N-1-k
        # h_{i+k} for i in 0..N-1-k
        c_k = np.dot(h[:N-k], 1 - h[k:])
        if c_k > max_c:
            max_c = c_k
    return max_c

def _project(h):
    """
    Project h to satisfy sum(h) = N/2 and 0 <= h_i <= 1.
    Simple projection: normalize sum, then clip.
    This might not preserve sum exactly after clipping.
    """
    h = np.clip(h, 0, 1)
    # Adjust to satisfy sum constraint
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-9:
        # Distribute difference uniformly or to elements not at bounds
        # Simple approach: add diff/N to all
        h += diff / N
        h = np.clip(h, 0, 1)
        # Re-normalize if clipping changed sum
        # Iterate a few times
        for _ in range(10):
            diff = N / 2.0 - np.sum(h)
            if abs(diff) < 1e-9:
                break
            # Add to elements not at 0 or 1
            mask = (h > 0) & (h < 1)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
            else:
                # If all at bounds, we are stuck or infeasible
                # But sum=N/2 is always feasible for N even
                break
            h = np.clip(h, 0, 1)
    return h

def construct():
    start_time = time.time()
    deadline = 110.0  # seconds
    
    # Initialize with uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    
    # Simple local search
    # Step size
    step_size = 0.1
    n_iter = 0
    
    while time.time() - start_time < deadline:
        # Generate a random perturbation
        # Perturb a few random elements
        n_perturb = 5
        idx = np.random.choice(N, n_perturb, replace=False)
        h_new = best_h.copy()
        for i in idx:
            h_new[i] += np.random.uniform(-step_size, step_size)
        
        h_new = _project(h_new)
        
        score_new = _max_overlap(h_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            # Reduce step size slowly
            step_size *= 0.99
        else:
            # If no improvement, maybe reduce step size more
            step_size *= 0.95
            
        n_iter += 1
        
        # Check time
        if time.time() - start_time > deadline - 5:
            break
            
    return best_h
