"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

# Deadline guard: kill after 140s to leave safety margin for 150s limit
DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _autoconv_max_abs(x):
    """Compute max absolute value of autoconvolution of x."""
    # Autoconvolution via FFT: conv(x, x)
    # For linear convolution, use FFT with zero-padding
    n = len(x)
    f = fft(x, n=2*n)
    conv = ifft(f * f).real[:n]  # linear autoconvolution, first n samples
    return np.max(np.abs(conv))

def _integral_sq(x):
    """Squared integral (sum squared) of x."""
    return np.sum(x * x)

def _objective(x):
    """Ratio: max |autoconv| / (integral^2). Lower is better."""
    den = _integral_sq(x)
    if den < 1e-12:
        return 1e10
    return _autoconv_max_abs(x) / den

def _soft_max_abs(conv, temperature=1.0):
    """Soft-max of absolute values via log-sum-exp."""
    a = np.abs(conv)
    # Scale to avoid overflow
    m = np.max(a)
    if m < 1e-12:
        return 0.0
    scaled = a / (m * temperature)
    log_sum = np.log(np.sum(np.exp(scaled))) + np.log(m * temperature)
    return np.exp(log_sum)

def _smooth_objective(x, temperature=0.5):
    """Smooth surrogate: soft-max |autoconv| / integral^2."""
    n = len(x)
    f = fft(x, n=2*n)
    conv = ifft(f * f).real[:n]
    smax = _soft_max_abs(conv, temperature)
    den = np.sum(x * x)
    if den < 1e-12:
        return 1e10
    return smax / den

def _analytical_gradient(x, temperature=0.5):
    """
    Analytical gradient of the smooth objective.
    Objective: S(x) = soft_max(|conv(x)|) / (sum x^2)
    We use chain rule. Let C = conv(x), A = |C|, S = soft_max(A)/T... 
    Actually, let's compute gradient numerically for safety, but use FFT-based
    approach for speed. Given time constraints, use numerical gradient with
    a small step, but optimized.
    """
    # For speed, use central difference with a moderate step
    eps = 1e-5
    grad = np.zeros_like(x)
    n = len(x)
    f = fft(x, n=2*n)
    conv = ifft(f * f).real[:n]
    a = np.abs(conv)
    m = np.max(a)
    if m < 1e-12:
        return grad
    scaled = a / (m * temperature)
    exp_scaled = np.exp(scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # d(soft_max)/d(conv[i]) = weights[i] * sign(conv[i]) / temperature
    # conv[i] = sum_j x[j] x[i-j] (with appropriate bounds for linear conv)
    # This is complex; use numerical gradient but with FFT acceleration
    
    # Actually, let's just use numerical gradient with a small step
    # but only for a subset of coordinates if needed. For N=400, 
    # numerical gradient is 400 FFTs which might be slow.
    
    # Alternative: use the fact that conv is quadratic, so gradient is linear
    # in x. Let's derive:
    # conv[k] = sum_{j} x[j] x[k-j] (for k in 0..N-1, with x out of range = 0)
    # d(conv[k])/dx[i] = x[k-i] + x[i-k] (with bounds)
    # d(soft_max)/dx[i] = sum_k weights[k] * sign(conv[k]) * (x[k-i] + x[i-k]) / temperature
    
    # Build the gradient
    grad = np.zeros(n)
    for k in range(n):
        w = weights[k] * np.sign(conv[k]) / temperature
        if w == 0:
            continue
        # d(conv[k])/dx[i] for i in 0..n-1
        # conv[k] = sum_j x[j] x[k-j]
        # d/dx[i] = sum_j [delta(j,i) x[k-j] + x[j] delta(k-j,i)]
        # = x[k-i] (if 0<=k-i<n) + x[k-i] (if 0<=k-i<n) ... wait
        # Actually: d/dx[i] sum_j x[j] x[k-j] = x[k-i] + x[k-i] if both terms valid?
        # No: term 1: j=i gives x[k-i]. Term 2: k-j=i => j=k-i gives x[k-i].
        # So d(conv[k])/dx[i] = x[k-i] + x[k-i] = 2*x[k-i] if 0<=k-i<n, else x[k-i] if only one valid.
        # For linear convolution with zero padding, x[j]=0 for j<0 or j>=n.
        idx = k - i
        if 0 <= idx < n:
            grad[i] += 2 * w * x[idx]
        else:
            # Only one term contributes? No, if idx out of range, both x[k-i] and x[k-i] are 0
            # Wait, if k-i < 0, then x[k-i]=0 and x[k-i]=0. If k-i>=n, same.
            pass
    
    # Now compute full gradient of S = smax / den
    den = np.sum(x * x)
    dsmax_dx = grad  # d(soft_max)/dx
    # d(S)/dx = (dsmax_dx * den - smax * 2*x) / den^2
    smax = _soft_max_abs(conv, temperature)
    grad_full = (dsmax_dx * den - smax * 2 * x) / (den * den)
    return grad_full

def _local_search_signs(x, max_iter=500):
    """
    Local search on sign assignments.
    x is a real vector. We fix magnitudes and flip signs to reduce objective.
    """
    n = len(x)
    magnitudes = np.abs(x)
    signs = np.sign(x)
    signs[signs == 0] = 1
    best_val = _objective(x)
    
    for it in range(max_iter):
        if _elapsed() > DEADLINE - 5:
            break
        improved = False
        # Try flipping each sign
        for i in range(n):
            if _elapsed() > DEADLINE - 5:
                break
            signs[i] *= -1
            x_new = signs * magnitudes
            val = _objective(x_new)
            if val < best_val - 1e-12:
                best_val = val
                improved = True
            else:
                signs[i] *= -1  # revert
        if not improved:
            break
        x = signs * magnitudes
    return x, best_val

def construct():
    global _start_time
    _start_time = time.time()
    
    n = N
    best_x = None
    best_score = np.inf
    
    # Strategy 1: Start with random signs on a fixed magnitude profile
    # Use a smooth magnitude profile, e.g., Gaussian or uniform
    # Try several magnitude profiles
    
    rng = np.random.default_rng(42)
    
    # Profile 1: Uniform magnitude
    mags = np.ones(n)
    
    # Profile 2: Gaussian
    t = np.linspace(-3, 3, n)
    mags2 = np.exp(-t*t/2)
    
    # Profile 3: Triangular
    mags3 = np.abs(1 - 2*np.linspace(0, 1, n))
    
    for mags in [mags, mags2, mags3]:
        if _elapsed() > DEADLINE - 10:
            break
        
        # Random sign initialization
        for trial in range(5):
            if _elapsed() > DEADLINE - 10:
                break
            signs = rng.choice([-1, 1], size=n)
            x = signs * mags
            val = _objective(x)
            
            # Local search on signs
            x, val = _local_search_signs(x, max_iter=200)
            
            if val < best_score:
                best_score = val
                best_x = x.copy()
        
        # Continuous optimization from best sign assignment
        if _elapsed() > DEADLINE - 15:
            break
        
        # Use the best x from sign search as initial point
        x0 = best_x.copy() if best_x is not None else signs * mags
        # Normalize to unit integral
        x0 = x0 / np.sqrt(np.sum(x0*x0))
        
        # Try smooth optimization
        try:
            res = minimize(
                _smooth_objective, x0, args=(0.3,),
                jac=_analytical_gradient,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-5, 'disp': False}
            )
            if res.success or res.fun < best_score:
                x_opt = res.x
                # Snap to signs for final evaluation
                signs_opt = np.sign(x_opt)
                signs_opt[signs_opt == 0] = 1
                x_snapped = signs_opt * np.abs(x_opt)
                val_opt = _objective(x_snapped)
                if val_opt < best_score:
                    best_score = val_opt
                    best_x = x_snapped
        except Exception:
            pass
        
        if _elapsed() > DEADLINE - 10:
            break
    
    # Final local search polish
    if best_x is not None and _elapsed() < DEADLINE - 5:
        best_x, best_score = _local_search_signs(best_x, max_iter=500)
    
    # If no valid result, return default
    if best_x is None:
        return np.ones(n)
    
    return best_x

# EVOLVE-BLOCK-END
