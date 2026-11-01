"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _smooth_abs(x, eps=1e-8):
    """Smooth approximation of absolute value."""
    return np.sqrt(x**2 + eps)

def _smooth_max(x, T=1.0, eps=1e-8):
    """Smooth approximation of max using log-sum-exp."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    max_x = np.max(x)
    # Avoid overflow in exp
    shifted = (x - max_x) / T
    # Handle potential -inf or very small values
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted) + eps)
    return max_x + T * log_sum

def _objective(x):
    """
    Objective function for optimization.
    x is a vector of length N, representing the 'log-odds' or raw values.
    We map x to h in (-1, 1) using tanh to keep values bounded.
    The metric is max_k |h * h[k]| / (sum_i |h_i|)^2
    """
    h = np.tanh(x)
    
    # Compute autoconvolution using FFT for efficiency
    # h has length N. Convolution length is 2N-1.
    # We care about the peak absolute value.
    # Using FFT:
    size = 1
    while size < 2 * N:
        size *= 2
    f = np.zeros(size, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    ac = np.fft.irfft(F * np.conj(F), n=size)[:2*N-1]
    
    # Compute smooth max of absolute values of ac
    abs_ac = _smooth_abs(ac)
    peak = _smooth_max(abs_ac, T=1.0)
    
    # Compute sum of absolute values of h
    sum_abs_h = np.sum(_smooth_abs(h))
    
    if sum_abs_h < 1e-12:
        return 1e10
        
    return peak / (sum_abs_h * sum_abs_h)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_val = float('inf')
    best_h = np.ones(N)  # Default fallback
    
    # Define multiple initializations to explore different regions of the landscape
    # 1. All ones
    # 2. Alternating signs
    # 3. Random
    # 4. Blocky random
    # 5. Sine wave signs
    
    initializations = []
    
    # 1. All positive
    initializations.append(np.full(N, 5.0)) # tanh(5) ~ 1
    
    # 2. Alternating
    alt = np.zeros(N)
    alt[::2] = 5.0
    alt[1::2] = -5.0
    initializations.append(alt)
    
    # 3. Random
    np.random.seed(42)
    initializations.append(np.random.randn(N) * 3.0)
    
    # 4. Blocky random (blocks of 10)
    blocks = np.random.choice([-1, 1], size=N//10)
    blocky = np.repeat(blocks * 5.0, 10)
    initializations.append(blocky)
    
    # 5. Sine wave
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    sine = np.sign(np.sin(t)) * 5.0
    initializations.append(sine)
    
    # 6. Another random
    np.random.seed(123)
    initializations.append(np.random.randn(N) * 3.0)
    
    for x0 in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            h_opt = np.tanh(res.x)
            # Evaluate the true objective (non-smooth) to get the actual score
            # This is important because the smooth objective is an approximation
            ac = np.convolve(h_opt, h_opt, mode='full')
            peak_true = np.max(np.abs(ac))
            sum_abs_true = np.sum(np.abs(h_opt))
            true_val = peak_true / (sum_abs_true * sum_abs_true) if sum_abs_true > 1e-12 else 1e10
            
            if true_val < best_val:
                best_val = true_val
                best_h = h_opt.copy()
                
        except Exception:
            continue
            
    # If we have time, do a few more random restarts
    if time.time() < deadline - 5:
        for i in range(10):
            if time.time() > deadline:
                break
            x0 = np.random.randn(N) * 3.0
            try:
                res = minimize(_objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                h_opt = np.tanh(res.x)
                ac = np.convolve(h_opt, h_opt, mode='full')
                peak_true = np.max(np.abs(ac))
                sum_abs_true = np.sum(np.abs(h_opt))
                true_val = peak_true / (sum_abs_true * sum_abs_true) if sum_abs_true > 1e-12 else 1e10
                
                if true_val < best_val:
                    best_val = true_val
                    best_h = h_opt.copy()
            except Exception:
                continue
                
    return best_h

# EVOLVE-BLOCK-END
