"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_score(h):
    """Compute max over shifts s=1..N-1 of sum_i h_i * (1 - h_{i+s mod N})."""
    h = np.asarray(h)
    n = len(h)
    max_overlap = 0.0
    for s in range(1, n):
        shifted = np.roll(h, s)
        overlap = np.sum(h * (1.0 - shifted))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Start with the known-good constant solution
    best_h = np.full(N, 0.5)
    best_score = _overlap_score(best_h)
    
    # Try a few random perturbations and local optimization
    rng = np.random.default_rng(42)
    
    max_iters = 500
    lr = 0.01
    
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            h = best_h.copy()
        else:
            h = np.clip(rng.uniform(0, 1, N), 0, 1)
            # Project to sum = N/2
            h = h - (h.sum() - N / 2) / N
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Compute gradient of max overlap
            # We'll use a subgradient approach: find the shift with max overlap
            max_overlap = 0.0
            max_shift = 0
            for s in range(1, N):
                shifted = np.roll(h, s)
                overlap = np.sum(h * (1.0 - shifted))
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_shift = s
            
            if max_overlap < best_score - 1e-9:
                best_score = max_overlap
                best_h = h.copy()
            
            # Gradient of sum_i h_i * (1 - h_{i+s}) w.r.t. h_i
            # = (1 - h_{i+s}) - h_{i-s}
            s = max_shift
            grad = (1.0 - np.roll(h, s)) - np.roll(h, -s)
            
            # Update
            h_new = h - lr * grad
            # Project to [0,1] and sum=N/2
            h_new = np.clip(h_new, 0, 1)
            # Adjust to fix sum
            diff = N / 2 - h_new.sum()
            if abs(diff) > 1e-12:
                h_new = h_new + diff / N
                h_new = np.clip(h_new, 0, 1)
            
            h = h_new
        
        # Final check
        final_score = _overlap_score(h)
        if final_score < best_score - 1e-9:
            best_score = final_score
            best_h = h.copy()
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = N / 2 - best_h.sum()
    if abs(diff) > 1e-12:
        # Distribute adjustment
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        # Fine-tune to get exact sum
        remaining = N / 2 - best_h.sum()
        if abs(remaining) > 0:
            # Adjust the largest element
            idx = np.argmax(best_h) if remaining > 0 else np.argmin(best_h)
            best_h[idx] += remaining
    
    return best_h
# EVOLVE-BLOCK-END
