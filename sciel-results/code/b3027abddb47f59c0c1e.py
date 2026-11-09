import numpy as np
import time

N = 600
DEADLINE_SECONDS = 115.0

def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    f = np.ones(N)
    
    # Compute autoconvolution peak (normalized)
    def score(f):
        # Autoconvolution via FFT
        n_fft = 2 * N
        F = np.fft.rfft(f, n=n_fft)
        autoconv = np.fft.irfft(F * np.conj(F), n=n_fft)
        # The autoconvolution on the original grid: indices 0..N-1 correspond to lags
        # f*f at lag k = sum_i f[i] f[i+k] for k>=0
        # Using FFT: autoconv[k] for k in 0..N-1
        peak = np.max(autoconv[:N])
        integral = np.sum(f)
        if integral <= 0:
            return np.inf
        return peak / (integral * integral)
    
    # Gradient descent with softplus parameterization
    # f = softplus(z) = log(1 + exp(z)) ensures nonnegativity
    z = np.zeros(N)  # log-scale parameterization
    
    def f_from_z(z):
        return np.exp(z)  # Use exp for simplicity, ensures positive
    
    def compute_score_and_grad(z):
        f = f_from_z(z)
        # Autoconvolution
        n_fft = 2 * N
        F = np.fft.rfft(f, n=n_fft)
        autoconv = np.fft.irfft(F * np.conj(F), n=n_fft)
        peak = np.max(autoconv[:N])
        integral = np.sum(f)
        score_val = peak / (integral * integral)
        
        # Gradient with respect to z
        # df/dz = f
        # d(autoconv)/df: autoconv[k] = sum_i f[i]f[i+k]
        # d(peak)/df is complex; use numerical or analytical
        # For simplicity, use finite differences on a few coordinates or full gradient
        
        # Actually, let's compute gradient analytically
        # autoconv[k] = sum_{i} f[i] * f[i+k] for k >= 0
        # d(autoconv[k])/df[j] = f[j+k] (if j+k < N) + f[j-k] (if j-k >= 0, for the symmetric part)
        # More carefully: autoconv[k] = sum_i f[i]f[i+k]
        # d/d f[j] = f[j+k] (from i=j term) + f[j-k] (from i=j-k term, if j-k >= 0)
        
        # But we need d(peak)/df, which requires knowing which lag gives the peak
        # This is non-smooth. Use subgradient or smooth approximation.
        
        # Alternative: minimize a smooth surrogate like sum of exp(alpha * autoconv)
        # or just use the peak with a smoothing
        
        # For now, use finite difference gradient (expensive but reliable)
        # Better: use the fact that we can compute gradient of autoconv at peak lag
        peak_lag = np.argmax(autoconv[:N])
        
        # d(autoconv[peak_lag])/df[j] for j in 0..N-1
        grad_autoconv = np.zeros(N)
        k = peak_lag
        for j in range(N):
            val = 0.0
            if j + k < N:
                val += f[j + k]
            if j - k >= 0:
                val += f[j - k]
            grad_autoconv[j] = val
        
        # d(score)/df = d(peak/(integral^2))/df
        # = (grad_autoconv * integral^2 - peak * 2 * integral * 1) / integral^4
        # = grad_autoconv / integral^2 - 2 * peak / integral^3
        grad_score = grad_autoconv / (integral * integral) - 2.0 * peak / (integral ** 3)
        
        # Chain rule: d(score)/dz = d(score)/df * df/dz = grad_score * f
        grad_z = grad_score * f
        
        return score_val, grad_z
    
    # Adaptive gradient descent
    lr = 0.01
    best_score = score(f)
    best_f = f.copy()
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        
        score_val, grad_z = compute_score_and_grad(z)
        
        # Gradient step
        z -= lr * grad_z
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.9
        
        f = f_from_z(z)
        cur_score = score(f)
        
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()
    
    # Ensure nonnegative and return as list
    best_f = np.maximum(best_f, 0)
    return list(best_f)
