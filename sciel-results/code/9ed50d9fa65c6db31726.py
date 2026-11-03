"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    # h is a 1D array of length N representing step heights
    # Overlap at shift s (in units of steps): O(s) = sum_i h[i] * (1 - h[(i+s) % N]) / N
    # = 0.5 - (1/N) * sum_i h[i] * h[(i+s) % N]  ... wait let me recalculate
    
    # Actually: O(s) = (1/N) * sum_i h[i] * (1 - h[(i+s)%N])
    # = (1/N) * (sum_i h[i] - sum_i h[i]*h[(i+s)%N])
    # = 0.5 - (1/N) * autocorr(s)
    
    # So max overlap = 0.5 - min_s [(1/N) * autocorr(s)]
    # To minimize max overlap, we want to maximize min_s autocorr(s)
    
    max_ov = 0.0
    for s in range(1, N):  # skip s=0 since that gives O(0) = 0.5 - mean(h)^2... 
        # Actually at s=0: O(0) = (1/N) sum h[i]*(1-h[i]) = 0.5 - (1/N)sum h[i]^2
        # Let's include all shifts
        shifted = np.roll(h, s)
        ov = np.mean(h * (1 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with a random binary-like sequence with sum = N/2
    np.random.seed(42)
    h = np.zeros(N)
    # Create a sequence with exactly N/2 ones and N/2 zeros, shuffled
    half = N // 2
    base = np.array([1.0]*half + [0.0]*(N-half))
    np.random.shuffle(base)
    h = base.copy()
    
    best_h = h.copy()
    best_score = _compute_max_overlap(best_h)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    steps_per_temp = 50
    
    current_score = best_score
    current_h = h.copy()
    
    while T > T_min:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                return best_h
            
            # Pick a random index and flip or perturb
            idx = np.random.randint(N)
            old_val = current_h[idx]
            
            # Try flipping to the other value, or a random value
            if np.random.rand() < 0.5:
                new_val = 1.0 - old_val
            else:
                new_val = np.random.rand()
            
            # Check if sum constraint is maintained approximately
            # We'll adjust later, but for now just try
            current_h[idx] = new_val
            
            new_score = _compute_max_overlap(current_h)
            
            if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = current_h.copy()
            else:
                current_h[idx] = old_val
        
        T *= cooling
    
    # Enforce exact sum constraint: adjust to make sum exactly N/2
    # The sum should already be close; let's verify
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2) > 1e-10:
        # Adjust by distributing the difference
        diff = N/2 - current_sum
        # Add diff/N to each element, then clip and re-adjust
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        # Re-normalize if clipping changed the sum
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h * (N/2) / s
    
    return best_h
# EVOLVE-BLOCK-END
