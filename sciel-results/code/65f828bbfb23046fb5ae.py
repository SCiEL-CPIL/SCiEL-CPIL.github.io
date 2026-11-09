"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Compute autoconvolution peak ratio
    def objective(h):
        h = np.maximum(h, 0)  # enforce nonnegativity
        total = np.sum(h)
        if total < 1e-12:
            return 1e10
        # Circular autoconvolution via FFT for speed
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:N]
        peak = np.max(conv)
        return peak / (total * total)
    
    # Gradient via finite differences is too slow; use analytical gradient
    def objective_and_grad(h):
        h = np.maximum(h, 0)
        total = np.sum(h)
        if total < 1e-12:
            return 1e10, np.zeros(N)
        
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:N]
        peak = np.max(conv)
        val = peak / (total * total)
        
        # Gradient: d/dh_i [peak / total^2]
        # d(peak)/dh_i = sum_j h[j] * delta(peak_lag - |i-j|) ... complex
        # Use subgradient: for the peak lag k*, d(conv[k*])/dh_i = sum_j h[j] * [k* == |i-j| or k* == N-|i-j|]
        peak_lag = int(np.argmax(conv))
        
        # d(conv[k])/dh_i = h[i+k] + h[i-k] (with wraparound for circular)
        # For linear autoconvolution (not circular), it's more careful
        # Let's use circular for simplicity and speed
        grad_conv = np.zeros(N)
        # conv[k] = sum_i h[i] * h[(i+k) % N]
        # d(conv[k])/dh[i] = h[(i+k)%N] + h[(i-k)%N]
        for i in range(N):
            grad_conv[i] = h[(i + peak_lag) % N] + h[(i - peak_lag) % N]
        
        # val = peak / total^2
        # d(val)/dh_i = (grad_conv[i] * total^2 - peak * 2 * total) / total^4
        #             = (grad_conv[i] * total - 2 * peak) / total^3
        grad = (grad_conv * total - 2 * peak) / (total ** 3)
        
        return val, grad
    
    # Try multiple random initializations
    best_h = np.ones(N)
    best_val = objective(best_h)
    
    n_starts = 10
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if s == 0:
            h0 = np.ones(N)
        elif s == 1:
            # Gaussian-like
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-x**2 * 4)
        elif s == 2:
            # Two peaks
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-(x-0.3)**2 * 10) + np.exp(-(x+0.3)**2 * 10)
        elif s == 3:
            # Cosine bump
            x = np.linspace(-1, 1, N)
            h0 = np.maximum(0, np.cos(np.pi * x / 2))
        elif s == 4:
            # Uniform with slight noise
            h0 = 1.0 + 0.1 * np.random.randn(N)
            h0 = np.maximum(h0, 0)
        else:
            h0 = np.random.rand(N)
        
        h0 = h0 / np.sum(h0) * N  # normalize to same total mass
        
        try:
            res = minimize(objective, h0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 60:
            break
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
