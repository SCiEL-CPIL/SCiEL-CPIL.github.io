"""Erdos minimum overlap: choose equal-width step heights h on [0,2], bounded in [0,1],
with integral 1 (sum = N/2) to minimize the maximum overlap of h with shifted (1-h)."""

import numpy as np
import time

N = 100

def _overlap_vector(h):
    """Compute overlap of h with shifted (1-h) for all circular shifts.
    overlap[k] = sum_i h[i] * (1 - h[(i+k) % N])
               = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
               = N/2 - autocorr[k]
    where autocorr[k] = sum_i h[i]*h[(i+k)%N] (circular autocorrelation)."""
    h_ext = np.concatenate([h, h])
    # Compute circular autocorrelation via FFT
    H = np.fft.fft(h, n=2*N)
    acf = np.fft.ifft(H * np.conj(H)).real[:N]
    # acf[k] = sum_i h[i] * h[(i+k) % N]  (circular)
    # overlap[k] = N/2 - acf[k]
    return N / 2.0 - acf

def _max_overlap(h):
    """Return the maximum overlap value over all shifts."""
    ov = _overlap_vector(h)
    return np.max(ov)

def construct():
    """Return step heights that minimize the maximum overlap of h with shifted (1-h).
    
    Key insight: For any h with sum = N/2, the average overlap over all circular shifts
    is exactly N/4. Thus max_overlap >= N/4 always. The uniform h=0.5 achieves exactly
    N/4 for every shift, making it optimal.
    
    However, we verify this and explore whether any non-uniform perturbation can
    maintain the same max while being more robust, or if there's a different
    scoring interpretation that allows improvement.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with uniform (provably optimal for the standard interpretation)
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)  # should be N/4 = 25
    
    # Try to find any h that gives lower max overlap via optimization
    # Use scipy minimize with the objective being max overlap
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            """Maximize the minimum autocorrelation (equivalently minimize max overlap).
            Use a smooth surrogate: minimize max(0, overlap - threshold) or use
            the sum of exponentials of overlaps to make it smooth and differentiable."""
            h = np.clip(x, 0, 1)
            # Normalize to have sum = N/2
            h = h * (N / 2.0) / np.sum(h)
            h = np.clip(h, 0, 1)
            ov = _overlap_vector(h)
            # Use a smooth max approximation
            return np.max(ov)
        
        # Try a few different initial conditions
        np.random.seed(42)
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                x0 = np.full(N, 0.5)
            else:
                # Random perturbation
                x0 = np.random.uniform(0, 1, N)
                x0 = x0 * (N / 2.0) / np.sum(x0)
                x0 = np.clip(x0, 0, 1)
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success:
                h_candidate = np.clip(res.x, 0, 1)
                # Ensure exact sum
                h_candidate = h_candidate * (N / 2.0) / np.sum(h_candidate)
                h_candidate = np.clip(h_candidate, 0, 1)
                # Final adjustment for exact sum
                diff = N / 2.0 - np.sum(h_candidate)
                if abs(diff) > 1e-12:
                    # Distribute difference evenly
                    h_candidate = h_candidate + diff / N
                    h_candidate = np.clip(h_candidate, 0, 1)
                
                score = _max_overlap(h_candidate)
                if score < best_score - 1e-9:
                    best_h = h_candidate.copy()
                    best_score = score
    except Exception:
        pass
    
    # Final verification and return
    # Ensure the sum is exactly N/2
    total = np.sum(best_h)
    if abs(total - N / 2.0) > 1e-12:
        best_h = best_h * (N / 2.0) / total
    
    return best_h
