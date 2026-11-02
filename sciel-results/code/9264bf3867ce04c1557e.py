"""Combined program: Spectral initialization + Smooth Gradient Descent + Iterative Local Search."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115  # safety margin before 120s kill
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, beta=50.0):
    """Smooth surrogate using log-sum-exp for max(|ac|) / (sum x)^2, with gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Log-sum-exp for smooth max
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    # Gradient computation
    c = weights * np.sign(ac)
    
    # Term B: x[j] * sum_{k=j}^{j+N-1} c[k]
    c_padded = np.zeros(2 * N - 1)
    c_padded[:len(c)] = c
    c_cumsum = np.cumsum(np.concatenate([[0], c_padded]))
    window_sums = np.zeros(N)
    for j in range(N):
        k_lo = j
        k_hi = min(j + N, 2 * N - 1)
        window_sums[j] = c_cumsum[k_hi] - c_cumsum[k_lo]
    term_B = x * window_sums
    
    # Term A: (c * x)_j = sum_m c[j+m] * x[m]
    term_A = np.zeros(N)
    for j in range(N):
        m_lo = 0
        m_hi = min(N - 1, 2 * N - 2 - j)
        if m_lo <= m_hi:
            ms = np.arange(m_lo, m_hi + 1)
            term_A[j] = np.sum(x[ms] * c[j + ms])
    
    grad_smooth_max = term_A + term_B
    
    # Gradient of val = smooth_max / s^2
    grad = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
    
    return val, grad

def _generate_spectral_signal(params):
    """
    Generate a signal from spectral parameters.
    params[0]: bandwidth (controls rolloff)
    params[1]: tilt (linear phase or amplitude tilt)
    params[2]: modulation depth
    """
    a = np.clip(params[0], 0.5, 5.0)  # Bandwidth parameter
    tilt = np.clip(params[1], -2.0, 2.0)
    mod = np.clip(params[2], 0.0, 1.0)
    
    k = np.arange(N)
    kf = k / N
    
    # Gaussian-like spectrum concentrated at low freq
    # We want |X[k]| to be smooth. 
    # Let's define the amplitude directly.
    # A common good shape is a raised cosine or Gaussian in frequency.
    # |X[k]| = exp(-a * kf^2)
    mag = np.exp(-a * kf**2)
    
    # Apply tilt: this effectively shifts the signal or adds a linear phase component 
    # if we treat it as amplitude, but for real signals we need symmetry.
    # Instead, let's modulate the time domain after IFFT, or use a complex spectrum.
    # Simpler: Just use the Gaussian spectrum, IFFT, and then apply time-domain transformations.
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    
    # Normalize
    if np.max(np.abs(x)) > 1e-10:
        x = x / np.max(np.abs(x))
    
    # Apply time-domain tilt/modulation
    t = np.linspace(0, 1, N)
    if abs(tilt) > 1e-3:
        x = x * np.exp(tilt * t)
    if mod > 1e-3:
        x = x * (1.0 + mod * 0.5 * np.sin(2 * np.pi * t))
        
    # Ensure sum is not zero
    if abs(np.sum(x)) < 1e-10:
        x += 1e-5
        
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x

    # Phase 1: Spectral Initialization
    # We search for a good spectral shape parameterization
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, -0.5, 0.0],
        [1.5, 0.0, 0.5],
        [2.5, 0.0, 0.0],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            # Optimize the 3 spectral parameters
            def _param_obj(params):
                x = _generate_spectral_signal(params)
                return _objective(x)
            
            result = minimize(_param_obj, p0, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _generate_spectral_signal(result.x)
        except Exception:
            pass
    
    if _check_deadline():
        return best_x

    # Phase 2: Smooth Gradient Descent
    # Use the best spectral candidate as a starting point for gradient descent
    def _smooth_obj_and_grad(x):
        return _smooth_objective_and_grad(x, beta=50.0)
    
    try:
        result = minimize(_smooth_obj_and_grad, best_x, method='L-BFGS-B', 
                         jac=True, options={'maxiter': 200, 'gtol': 1e-4, 'ftol': 1e-7})
        if result.fun < best_val:
            # Verify with exact objective
            exact_val = _objective(result.x)
            if exact_val < best_val:
                best_val = exact_val
                best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x

    # Phase 3: Iterative Local Search (ILS) on Exact Objective
    # This is the structural change: using a robust local search on the non-smooth 
    # exact objective to escape local minima of the surrogate.
    rng = np.random.default_rng(42)
    step_size = 0.05
    current_x = best_x.copy()
    current_val = best_val
    stall_count = 0
    max_iterations = 50000
    
    for iteration in range(max_iterations):
        if _check_deadline():
            break
        
        # Random perturbation
        # We perturb a subset of indices or all indices with small noise
        # To be efficient, we can perturb all indices with Gaussian noise
        noise = rng.normal(0, step_size, size=N)
        x_trial = current_x + noise
        
        val_trial = _objective(x_trial)
        
        if val_trial < current_val:
            current_x = x_trial
            current_val = val_trial
            stall_count = 0
            # Slightly decrease step size on success to refine
            step_size = max(0.001, step_size * 0.99)
            
            # Update global best
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
        else:
            stall_count += 1
            # Increase step size periodically to escape local minima
            if stall_count % 500 == 0:
                step_size = min(0.5, step_size * 1.1)
            
            # Occasional large jump (restart) if stuck too long
            if stall_count > 5000:
                # Jump back to a slightly perturbed version of the best found so far
                # or a new random spectral signal
                if rng.random() < 0.1:
                    # Try a new spectral signal
                    new_params = [np.random.uniform(0.5, 3.0), np.random.uniform(-1, 1), np.random.uniform(0, 1)]
                    x_new = _generate_spectral_signal(new_params)
                    val_new = _objective(x_new)
                    if val_new < current_val:
                        current_x = x_new
                        current_val = val_new
                        step_size = 0.05
                        stall_count = 0
                    else:
                        # Keep current, but reset stall counter to avoid infinite loops in logic
                        stall_count = 0 

    return best_x
# EVOLVE-BLOCK-END
