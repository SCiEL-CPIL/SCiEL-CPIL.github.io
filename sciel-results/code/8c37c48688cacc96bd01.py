import numpy as np
import time

N = 400

def _get_cost(h):
    """
    Compute the upper bound on C3: max(|autocorr_sidelobe|)
    """
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)[:N]
    # C[0] is the DC component
    # We want max absolute value of sidelobes C[1:]
    if N > 1:
        max_side = np.max(np.abs(C[1:]))
    else:
        max_side = 0.0
    return max_side

def _fix_sum(h):
    """
    Ensure the sum of h is not zero.
    """
    s = np.sum(h)
    if s == 0:
        # Flip a random index to change sum by +2 or -2
        idx = np.random.randint(N)
        h[idx] = -h[idx]
    return h

def _local_search(h, max_iters=300):
    """
    Hill Climbing with Block Flips.
    """
    current_cost = _get_cost(h)
    best_h = h.copy()
    best_cost = current_cost
    
    for i in range(max_iters):
        # Pick a block size (1 to 5) and start index
        block_size = np.random.randint(1, 6)
        start = np.random.randint(0, N - block_size + 1)
        
        # Save original values
        original_values = h[start:start+block_size].copy()
        
        # Flip the block
        h[start:start+block_size] = -original_values
        
        new_cost = _get_cost(h)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            # Revert
            h[start:start+block_size] = original_values
            
    return best_h, best_cost

def construct():
    deadline = time.time() + 110.0 # Safety margin
    np.random.seed(42)
    
    best_overall_h = None
    best_overall_cost = float('inf')
    
    # Multiple restarts
    while time.time() < deadline:
        # Initialize with random +/- 1
        h = np.random.choice([-1.0, 1.0], size=N)
        
        # Enforce non-zero sum immediately
        h = _fix_sum(h)
        
        # Local Search
        h_opt, cost_opt = _local_search(h, max_iters=300)
        
        # Ensure final result also has non-zero sum
        h_opt = _fix_sum(h_opt)
        # Recalculate cost after fix if we changed it
        final_cost = _get_cost(h_opt)
        
        if final_cost < best_overall_cost:
            best_overall_cost = final_cost
            best_overall_h = h_opt.copy()
            
        # If we have time, do a polish pass
        if time.time() < deadline - 5.0 and best_overall_h is not None:
             h_polish = best_overall_h.copy()
             h_polish = _fix_sum(h_polish)
             h_polish, cost_polish = _local_search(h_polish, max_iters=100)
             h_polish = _fix_sum(h_polish)
             final_polish_cost = _get_cost(h_polish)
             
             if final_polish_cost < best_overall_cost:
                 best_overall_cost = final_polish_cost
                 best_overall_h = h_polish.copy()

    # Final safety check
    if best_overall_h is None:
        best_overall_h = np.random.choice([-1.0, 1.0], size=N)
        best_overall_h = _fix_sum(best_overall_h)
        
    return best_overall_h
