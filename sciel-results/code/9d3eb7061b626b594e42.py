"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize, LinearConstraint

N = 600
STEP = 1.0 / N  # step size for intervals on [-1/4, 1/4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of 600 nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of the autoconvolution (upper bound on C1).
    
    Strategy:
    1. Start with flat function (theoretical lower bound by Cauchy-Schwarz).
    2. Attempt local optimization to see if we can improve beyond 1.0.
    3. If optimization fails or doesn't improve, return flat function.
    """
    t0 = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Flat function: achieves Cauchy-Schwarz lower bound
    flat = np.ones(N)
    
    # Compute the objective for flat function to establish baseline
    def autoconv_max_normalized(f):
        """Compute max of autoconvolution normalized by (sum f)^2 * N."""
        f = np.asarray(f, dtype=np.float64)
        if np.any(f < -1e-12):
            return np.inf
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return np.inf
        # Autoconvolution at lag 0 is sum(f^2)
        # But max autoconvolution might be at other lags
        # For efficiency, compute via FFT
        nfft = 2 * N
        F = np.fft.rfft(f, n=nfft)
        ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
        max_ac = np.max(ac)
        return max_ac / (s * s) * N
    
    baseline_val = autoconv_max_normalized(flat)
    best_f = flat.copy()
    best_val = baseline_val
    
    # Try to improve using gradient-based optimization
    # We want to minimize max(f*f) / (sum f)^2 * N
    # This is equivalent to minimizing max(f*f) * N / (sum f)^2
    
    # Use a smooth surrogate: instead of max, use log-sum-exp approximation
    def objective_smooth(x, beta=50.0):
        """Smooth approximation of the objective using log-sum-exp for max."""
        if time.time() - t0 > deadline:
            return best_val  # Return best so far if deadline exceeded
        f = np.maximum(x, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return 1e10
        nfft = 2 * N
        F = np.fft.rfft(f, n=nfft)
        ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
        ac = np.maximum(ac, 1e-15)
        # log-sum-exp approximation of max(ac)
        m = np.max(ac)
        if m < 1e-12:
            return 1e10
        log_sum_exp = m + np.log(np.sum(np.exp((ac - m) * beta)) / beta)
        # Objective: log_sum_exp * N / s^2
        return log_sum_exp * N / (s * s)
    
    def gradient_smooth(x, beta=50.0):
        """Numerical gradient of the smooth objective."""
        if time.time() - t0 > deadline:
            return np.zeros(N)
        f = np.maximum(x, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return np.zeros(N)
        
        nfft = 2 * N
        F = np.fft.rfft(f, n=nfft)
        ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
        ac = np.maximum(ac, 1e-15)
        
        # We need d(max_ac)/df_i approximately
        # Use log-sum-exp gradient
        m = np.max(ac)
        if m < 1e-12:
            return np.zeros(N)
        exp_ac = np.exp((ac - m) * beta)
        weights = exp_ac / np.sum(exp_ac)  # Soft max weights
        
        # Gradient of ac[k] w.r.t. f[i]
        # ac[k] = sum_j f[j] f[j+k]
        # dac[k]/df[i] = f[i+k] + (if i-k >= 0) f[i-k] for the circular convolution
        # Actually for linear autoconvolution: ac[k] = sum_j f[j]*f[j+k]
        # dac[k]/df[i] = f[i+k] if i+k < N, plus f[i-k] if i-k >= 0
        
        grad_f = np.zeros(N)
        for k in range(N):
            w = weights[k]
            if w < 1e-10:
                continue
            # d(ac[k])/df[i] for each i
            # ac[k] = sum_{j=0}^{N-1-k} f[j] f[j+k]
            # d(ac[k])/df[i] = f[i+k] if i <= N-1-k, plus f[i-k] if i >= k
            for i in range(N):
                contrib = 0.0
                if i + k < N:
                    contrib += f[i + k]
                if i - k >= 0:
                    contrib += f[i - k]
                grad_f[i] += w * contrib
        
        # Chain rule: objective = N * log_sum_exp / s^2
        # d(objective)/df[i] = N * (d(log_sum_exp)/df[i] * s^2 - log_sum_exp * 2*s * d(s)/df[i]) / s^4
        # = N * (grad_f[i] * s - 2 * log_sum_exp * 1) / s^3  (since d(s)/df[i] = 1)
        
        log_sum_exp = m + np.log(np.sum(exp_ac) / beta)
        s2 = s * s
        s3 = s2 * s
        obj_grad = N * (grad_f * s - 2.0 * log_sum_exp) / s3
        return obj_grad
    
    # Try gradient descent with projected nonnegative constraint
    lr = 0.01
    f_current = flat.copy()
    for iteration in range(200):
        if time.time() - t0 > deadline:
            break
        grad = gradient_smooth(f_current)
        f_new = f_current - lr * grad
        f_new = np.maximum(f_new, 0.0)
        
        val_new = objective_smooth(f_new)
        if val_new < best_val:
            best_val = val_new
            best_f = f_new.copy()
        f_current = f_new
        
        # Adaptive learning rate
        if iteration % 20 == 0:
            lr *= 0.9
    
    # Final evaluation with exact objective
    final_val = autoconv_max_normalized(best_f)
    if final_val < baseline_val:
        # Only use optimized solution if it actually improves
        return list(best_f)
    
    # Return flat function (theoretical optimum)
    return list(flat)
# EVOLVE-BLOCK-END
