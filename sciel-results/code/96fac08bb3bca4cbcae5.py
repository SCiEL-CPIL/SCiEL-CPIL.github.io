import numpy as np
import time

N = 100

def _compute_overlaps(h):
    """Compute overlap values O_k = sum_i h_i * (1 - h_{(i+k)%N}) for all shifts k."""
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value."""
    return np.max(_compute_overlaps(h))

def construct():
    """
    Construct step heights minimizing max overlap with shifted 1-h.
    
    Returns a sequence of N step heights in [0,1] with sum exactly N/2,
    minimizing the maximum overlap O_k = sum_i h_i * (1 - h_{(i+k)%N}).
    
    The uniform vector h_i = 0.5 achieves the theoretical lower bound N/4
    for the maximum overlap, as proven by the constant autocorrelation property.
    """
    start_time = time.time()
    deadline = 120.0
    
    # The uniform solution is provably optimal: max overlap = N/4
    uniform = np.full(N, 0.5)
    
    # Verify the uniform solution meets all constraints
    assert np.all(uniform >= 0.0) and np.all(uniform <= 1.0)
    assert abs(np.sum(uniform) - N / 2.0) < 1e-12
    
    # Compute the max overlap for verification
    max_ov = _max_overlap(uniform)
    
    # If we have time, try a small perturbation search to confirm optimality
    # and potentially find any numerical edge cases
    if time.time() - start_time < deadline - 10.0:
        best_h = uniform.copy()
        best_score = max_ov
        
        # Try a few random perturbations to verify no improvement exists
        rng = np.random.default_rng(42)
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Create a small perturbation
            h_try = uniform + rng.normal(0, 0.01, N)
            h_try = np.clip(h_try, 0.0, 1.0)
            
            # Adjust to ensure sum is exactly N/2
            diff = N / 2.0 - np.sum(h_try)
            h_try += diff / N
            h_try = np.clip(h_try, 0.0, 1.0)
            
            # Re-adjust after clipping
            diff = N / 2.0 - np.sum(h_try)
            h_try += diff / N
            
            if np.any(h_try < -1e-10) or np.any(h_try > 1.0 + 1e-10):
                continue
            
            score = _max_overlap(h_try)
            if score < best_score - 1e-12:
                best_score = score
                best_h = h_try.copy()
        
        if best_score < max_ov - 1e-12:
            return best_h
    
    return uniform
