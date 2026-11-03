import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / N  # Width of each interval
X = np.linspace(-0.25, 0.25, N, endpoint=False)

def _compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    # Pad to avoid circular convolution
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)
    # The linear autoconvolution is conv[0:n] (lags 0 to n-1)
    peak = np.max(conv[:n])
    total = np.sum(h)
    if total == 0:
        return float('inf')
    # Normalized maximum: peak / total^2
    return peak / (total * total)

def gaussians_to_h(params):
    """
    Convert parameters to heights.
    params: flat array of length 3*K
    For each Gaussian k: center, width, amplitude
    """
    K = len(params) // 3
    h = np.zeros(N)
    for k in range(K):
        center = params[3*k]
        width = params[3*k+1]
        amp = params[3*k+2]
        # Ensure width is positive
        width = abs(width) + 1e-6
        h += amp * np.exp(-0.5 * ((X - center) / width) ** 2)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = float('inf')
    best_heights = np.ones(N) # Default fallback
    
    # We will try optimizing K Gaussians for K in [1, 2, 3, 4]
    # For each K, we do a few restarts with different initial conditions.
    
    for K in [1, 2, 3, 4]:
        if time.time() > deadline:
            break
            
        # Initial guesses
        # We want to place Gaussians to minimize overlap.
        # For K=1, it's just a single bump.
        # For K>1, place them apart.
        
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize parameters
            # Centers spread out
            centers = np.linspace(-0.2, 0.2, K) + np.random.uniform(-0.02, 0.02, K)
            widths = np.full(K, 0.05 + 0.01 * r)
            amplitudes = np.ones(K)
            
            params0 = np.concatenate([centers, widths, amplitudes])
            
            def objective(params):
                h = gaussians_to_h(params)
                # Clip to non-negative (Gaussians are positive if amp>0, but we might get negative amps? 
                # Better to enforce positivity in the model or just let optimizer handle it.
                # If amp is negative, it creates a hole. The task says "nonnegative step function".
                # So we must ensure h >= 0.
                # Using abs(amplitude) in gaussians_to_h is safer for non-negativity if we treat amp as magnitude.
                # Let's modify gaussians_to_h to use abs(amp) or just clip h.
                # Clipping h is easier and differentiable enough for Nelder-Mead?
                # Nelder-Mead doesn't use gradients. Clipping is fine.
                h = np.maximum(h, 0)
                score = _compute_score(h)
                return score
            
            try:
                # Use Nelder-Mead
                res = opt.minimize(objective, params0, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                
                if res.success or res.fun < best_score:
                    h_opt = gaussians_to_h(res.x)
                    h_opt = np.maximum(h_opt, 0)
                    s = _compute_score(h_opt)
                    if s < best_score:
                        best_score = s
                        best_heights = h_opt
            except Exception:
                pass

    # Fallback if optimization failed or didn't find anything
    if best_score == float('inf'):
        best_heights = np.ones(N)
        
    return best_heights.tolist()
