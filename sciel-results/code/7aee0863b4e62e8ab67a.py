"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    # Try sinusoidal patterns with different frequencies and amplitudes
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    def compute_overlap(h):
        """Compute max overlap of h with shifted 1-h (cyclic)."""
        max_ov = 0.0
        for s in range(N):
            shifted = np.roll(h, -s)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Evaluate baseline
    base_score = compute_overlap(best_h)
    best_score = base_score
    
    # Try sinusoidal perturbations
    for k in range(1, N // 2 + 1):
        if time.time() > deadline - 5:
            break
        for A in np.linspace(0.0, 0.5, 11):
            if time.time() > deadline - 5:
                break
            h = 0.5 + A * np.sin(2 * np.pi * k * np.arange(N) / N)
            # Clip to [0,1] and renormalize sum
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            if current_sum > 0:
                h = h * (N / 2.0) / current_sum
                h = np.clip(h, 0.0, 1.0)
                # Re-normalize after clip
                diff = N / 2.0 - np.sum(h)
                if abs(diff) < 1e-10:
                    pass
                else:
                    # Distribute difference
                    h += diff / N
                    h = np.clip(h, 0.0, 1.0)
            
            score = compute_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Final check
    final_sum = np.sum(best_h)
    if abs(final_sum - N / 2.0) > 1e-6:
        # Fallback to constant
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
