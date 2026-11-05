import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def _objective(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta=50.0):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(-beta * (x - m)))) / beta

def _grad_soft_max(h, beta=50.0):
    conv = np.convolve(h, h)
    m = np.max(conv)
    w = np.exp(-beta * (conv - m))
    w /= np.sum(w)
    # d(max approx)/d(h[i]) = 2 * sum_j w[j] * h[j-i]  (for j in [0, 2N-2])
    # More precisely: d conv[j] / d h[i] = h[j-i] + h[i-j] (with bounds)
    # Let's just use subgradient: direction is sum_j w[j] * d(conv[j])/d(h)
    # d conv[j] / d h[i] = h[j-i] if 0<=j-i<N else 0, plus h[i-j] if 0<=i-j<N else 0
    # So gradient = sum_j w[j] (h[j-i] + h[i-j]) for valid indices
    # This is essentially 2 * (w * h) circular-convolved-like
    # For simplicity, use a finite difference or just return the weighted autocorrelation
    # Let's use: grad[i] = sum_j w[j] * (h[j-i] if valid) + (h[i-j] if valid)
    # Actually, for L-BFGS-B, we can just use the objective without gradient
    return None

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones(N)
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(12345)
    
    # Strategy: Start from low-frequency signals and refine
    # The key insight: to minimize max autocorrelation / (sum)^2,
    # we want a signal whose autocorrelation is as flat as possible
    # (i.e., close to a delta at lag 0 with small sidelobes)
    # AND has large DC component.
    
    # Approach 1: Spectral shaping
    # If h has flat power spectrum (white noise), its autocorrelation is delta-like
    # But we need sum(h) > 0, so add DC
    
    for trial in range(8):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        # Create a signal with controlled spectral content
        # Mix of low-frequency (for DC/sum) and higher frequency (for flat autocorr)
        
        # Method: random phase at each frequency
        n_freq = N // 2 + 1
        freqs = np.arange(n_freq)
        
        # Amplitude profile: emphasize low frequencies for large sum
        # But also keep some high freq for flat autocorrelation
        # Try different alpha values
        alpha = 0.3 + 0.1 * trial
        
        # Amplitude: |H[k]| = A * (1 + freqs^(-alpha)) for k > 0
        # H[0] = large (DC)
        spec = np.zeros(n_freq, dtype=complex)
        spec[0] = 5.0  # DC component
        
        for k in range(1, n_freq):
            mag = 1.0 / (1.0 + freqs[k] ** alpha)
            phase = rng.uniform(-np.pi, np.pi)
            spec[k] = mag * np.exp(1j * phase)
        
        # Enforce conjugate symmetry for real signal
        # irfft handles this
        h_init = np.fft.irfft(spec, n=N)
        
        # Normalize and shift to ensure positive sum
        h_init = h_init - np.mean(h_init) * 0.5  # Keep some DC but reduce
        h_init = h_init + np.mean(h_init) * 0.5 + 1.0  # Ensure positive bias
        
        # Ensure sum is not zero
        if abs(np.sum(h_init)) < 1e-10:
            h_init += 1.0
        
        val = _objective(h_init)
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
        
        # Local refinement using Nelder-Mead or COBYLA
        try:
            from scipy.optimize import minimize
            
            # Use COBYLA for constraint-free optimization
            result = minimize(
                _objective,
                h_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'fatol': 1e-10, 'xatol': 1e-10}
            )
            
            if np.all(np.isfinite(result.x)):
                res_val = _objective(result.x)
                if res_val < best_val:
                    best_val = res_val
                    best_h = result.x.copy()
        except Exception:
            pass
        
        # Try another local method from the best found so far
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
    
    # Final refinement on best_h
    try:
        from scipy.optimize import minimize
        
        result = minimize(
            _objective,
            best_h,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'fatol': 1e-12, 'xatol': 1e-12}
        )
        
        if np.all(np.isfinite(result.x)):
            res_val = _objective(result.x)
            if res_val < best_val:
                best_val = res_val
                best_h = result.x.copy()
    except Exception:
        pass
    
    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
