"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 600

# EVOLVE-BLOCK-START
def _compute_score(h):
    """
    Computes the normalized maximum of the autoconvolution.
    Metric: max(h*h) / (||h||_1 * ||h||_inf)
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return float('inf')
        
    area = np.sum(h)
    max_val = np.max(h)
    
    if area == 0 or max_val == 0:
        return float('inf')
        
    # Use FFT for convolution
    # Zero-pad to length 2*N to avoid circular convolution artifacts
    f = np.zeros(2 * N)
    f[:N] = h
    fft_f = np.fft.fft(f)
    conv = np.fft.ifft(fft_f * fft_f).real
    max_conv = np.max(conv)
    
    return max_conv / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    idx = np.arange(N)
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Parametric Search with Optimal Alpha
    # We search over the width W of the central block of 1s.
    # The rest of the function is alpha.
    # We find the optimal alpha for each W.
    
    # We can optimize the range of W. 
    # Based on inspirations, W around N/2 or similar might be good.
    # But let's scan a good portion.
    
    # To save time, we can scan W from 10 to N-10 in steps of 1 initially.
    # 600 iterations is fine.
    
    for W in range(1, N + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Create mask for central block of size W
        start_idx = (N - W) // 2
        end_idx = start_idx + W
        
        mask = np.zeros(N, dtype=bool)
        mask[start_idx:end_idx] = True
        
        # Function to evaluate score for a given alpha
        def score_alpha(alpha):
            h = np.where(mask, 1.0, alpha)
            return _compute_score(h)
        
        # Find optimal alpha in [0, 1]
        # We can use a coarse grid first to find the basin, then refine?
        # Or just minimize_scalar.
        try:
            res = minimize_scalar(score_alpha, bounds=(0, 1), method='bounded', options={'xatol': 1e-5})
            if res.success:
                alpha_opt = res.x
                h_candidate = np.where(mask, 1.0, alpha_opt)
                score_candidate = res.fun
                
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_h = h_candidate.copy()
        except Exception:
            continue

    if best_h is None:
        # Fallback to constant
        return [1.0] * N

    # Phase 2: Local Search (Hill Climbing)
    # Start from the best parametric candidate.
    
    current_h = best_h
    current_score = best_score
    
    np.random.seed(42)
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Perturb
        noise = np.random.normal(0, step_size, N)
        h_new = current_h + noise
        h_new = np.clip(h_new, 0, 1)
        
        if np.max(h_new) == 0:
            continue
            
        score_new = _compute_score(h_new)
        
        # Acceptance criterion
        if score_new < current_score:
            current_h = h_new
            current_score = score_new
            no_improve_count = 0
            step_size *= 0.98
        else:
            no_improve_count += 1
            step_size *= 0.95
            if no_improve_count > max_no_improve:
                # Try a larger jump
                if np.random.rand() < 0.1:
                    noise = np.random.normal(0, 0.5, N)
                    h_new = current_h + noise
                    h_new = np.clip(h_new, 0, 1)
                    if np.max(h_new) > 0:
                        score_new = _compute_score(h_new)
                        if score_new < current_score * 1.5:
                            current_h = h_new
                            current_score = score_new
                else:
                    no_improve_count = 0
                    step_size = 0.1

    return list(current_h)
# EVOLVE-BLOCK-END
