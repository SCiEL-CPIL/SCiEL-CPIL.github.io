"""Rewritten program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_chebyt

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 140.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    """Compute the raw value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    """Compute value and gradient analytically for full-dim optimization."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    m = idx
    dh_conv = np.zeros(n)
    valid_j = np.arange(n)
    km = m - valid_j
    mask = (km >= 0) & (km < n)
    dh_conv[mask] = 2.0 * h[km[mask]]
    
    grad = 2.0 * n * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _soft_max(x, beta):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(x)
    shifted = (x - m) * beta
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted)) / len(x)) / beta

def _objective_soft_and_grad(coeffs, basis, beta):
    """
    Objective in coefficient space using soft-max for smoothness.
    """
    h = coeffs @ basis.T
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(coeffs)
    
    conv = _autoconv(h)
    
    m = _soft_max(conv, beta)
    
    shifted = (conv - m) * beta
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    dval_dm = 2.0 * N / (s * s)
    
    try:
        corr = np.correlate(weights, h, mode='full')
        start_idx = (len(weights) - 1)
        grad_h = 2.0 * dval_dm * corr[start_idx : start_idx + N]
    except Exception:
        return 1e10, np.zeros_like(coeffs)
    
    grad_coeffs = basis.T @ grad_h
    
    raw_sign = 1.0 if (2.0 * N * m / (s*s)) > 0 else -1.0
    grad_coeffs *= raw_sign
    
    val = abs(2.0 * N * m / (s * s))
    
    return val, grad_coeffs

def _make_initial_conditions():
    """Generate diverse initial conditions."""
    inits = []
    x = np.linspace(-1, 1, N)
    rng = np.random.default_rng(42)
    
    inits.append(np.ones(N))
    inits.append(np.linspace(1, -1, N))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x))
    inits.append(np.exp(-100 * x**2))
    inits.append(np.exp(-10 * x**2))
    inits.append(np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2))
    inits.append(1.0 + 0.1 * rng.standard_normal(N))
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -0.5))
    inits.append(1.0 - 2.0 * np.abs(x))
    inits.append(1.0 + 0.5 * np.sin(4 * np.pi * x))
    inits.append(np.exp(-30*(x-0.4)**2) + np.exp(-30*(x)**2) + np.exp(-30*(x+0.4)**2))
    inits.append(np.sign(np.sin(3*np.pi*x)))
    inits.append(2.0 * (x - np.floor(x + 0.5)))
    inits.append(1.0 + 0.3 * np.sin(8 * np.pi * x))
    inits.append(1.0 + 0.4 * np.sin(2*np.pi*x) + 0.2*np.sin(6*np.pi*x))
    inits.append(1.0 + 0.3 * rng.standard_normal(N))
    
    from scipy.ndimage import gaussian_filter1d
    noise = rng.standard_normal(N)
    inits.append(1.0 + 0.5 * gaussian_filter1d(noise, sigma=5))
    inits.append(np.linspace(-1, 2, N))
    inits.append(np.concatenate([np.linspace(-1, 1, N//2), np.linspace(1, -1, N - N//2)]))
    inits.append(1.0 - 0.5 * x**2)
    # Additional diverse inits
    inits.append(1.0 + 0.6 * np.cos(4 * np.pi * x) + 0.2 * np.cos(8 * np.pi * x))
    inits.append(np.exp(-20 * x**2) * (1.0 + 0.3 * np.cos(6 * np.pi * x)))
    inits.append(1.5 - 0.5 * np.abs(2 * x) ** 1.5)
    inits.append(0.5 + 0.8 * np.cos(2 * np.pi * x) * np.exp(-5 * x**2))
    
    return inits

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Chebyshev basis search with varying numbers of coefficients
    # and soft-max smoothness for better gradient information
    num_coeffs_list = [6, 8, 10, 12]
    basis_cache = {}
    betas = [30.0, 50.0, 80.0]
    
    for num_coeffs in num_coeffs_list:
        if not _check_time():
            break
        
        if num_coeffs not in basis_cache:
            x = np.linspace(-1, 1, N)
            basis = np.zeros((N, num_coeffs))
            for k in range(num_coeffs):
                basis[:, k] = eval_chebyt(k, x)
            basis_cache[num_coeffs] = basis
        basis = basis_cache[num_coeffs]
        
        for beta in betas:
            if not _check_time():
                break
            
            # Multiple random starts in coefficient space
            rng = np.random.default_rng(42 + num_coeffs * 100 + int(beta))
            for start_iter in range(3):
                if not _check_time():
                    break
                
                # Initialize coefficients
                c0 = 1.0 + 0.5 * rng.standard_normal()
                c_rest = 0.5 * rng.standard_normal(num_coeffs - 1)
                coeffs0 = np.concatenate([[c0], c_rest])
                
                try:
                    result = minimize(
                        _objective_soft_and_grad,
                        coeffs0,
                        args=(basis, beta),
                        method='L-BFGS-B',
                        jac=True,
                        options={
                            'maxiter': 100,
                            'ftol': 1e-12,
                            'gtol': 1e-10,
                            'maxfun': 300,
                        }
                    )
                    
                    h_try = result.x @ basis.T
                    val = _compute_value(h_try)
                    if val < best_val:
                        best_val = val
                        best_h = h_try.copy()
                except Exception:
                    continue
    
    # Phase 2: Full-dimension gradient optimization from diverse starts
    inits = _make_initial_conditions()
    
    for init_h in inits:
        if not _check_time():
            break
        
        h0 = init_h.astype(np.float64)
        
        try:
            result = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 150,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxfun': 500,
                    'maxls': 30,
                }
            )
            
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    # Phase 3: Polish the best result with tighter tolerances
    if best_val < 2.0 and _check_time():
        try:
            result = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 400,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxfun': 1000,
                    'maxls': 50,
                }
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: If time remains, try random perturbations around best
    if best_val < 2.0 and _check_time():
        rng = np.random.default_rng(123)
        for _ in range(8):
            if not _check_time():
                break
            perturbation = 0.03 * rng.standard_normal(N)
            h_try = best_h + perturbation
            try:
                result = minimize(
                    _objective_and_grad,
                    h_try,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 150,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxfun': 500,
                        'maxls': 35,
                    }
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
            except Exception:
                continue
    
    # Phase 5: Final high-precision polish
    if best_val < 2.0 and _check_time():
        try:
            result = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxfun': 1200,
                    'maxls': 60,
                }
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = _autoconv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
