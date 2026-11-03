import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac * 0.5 / (integral ** 2)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Fourier parameterization
    # h[k] = |sum_{m=0}^{deg} c_m exp(i * 2*pi * m * k / N)|^2
    # This ensures h[k] >= 0.
    # We optimize complex coefficients c_m.
    
    # Degree 6 seems to offer a good balance between expressiveness and search space.
    deg = 6
    n_coeffs = deg + 1
    n_params = 2 * n_coeffs
    
    # Precompute exponentials for speed
    k = np.arange(N)
    # Matrix of exponentials: M[i, m] = exp(i * 2*pi * m * i / N)
    # Shape: (N, deg+1)
    exponents = np.exp(1j * 2 * np.pi * np.outer(k, np.arange(n_coeffs)) / N)
    
    def params_to_h(params):
        c = params[0:n_coeffs] + 1j * params[n_coeffs:2*n_coeffs]
        # P = exponents @ c
        P = exponents.dot(c)
        h = np.abs(P) ** 2
        return h
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        h = params_to_h(params)
        return compute_score(h)
    
    # Bounds
    # Coefficients can be scaled arbitrarily, but the score is scale-invariant (max_ac scales with s^2, integral^2 scales with s^2).
    # So we can fix the scale of c to avoid scaling issues.
    # Let's constrain the magnitude of c to be within [0.1, 10] or similar.
    # Actually, differential_evolution works best with bounded variables.
    # Let's use bounds (-1, 1) for real and imag parts.
    bounds = [(-1.0, 1.0)] * n_params
    
    best_score = np.inf
    best_params = None
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            seed=42, 
            polish=True,
            tol=1e-5,
            maxfev=5000
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
    except Exception as e:
        pass
    
    # Polish with L-BFGS-B for higher precision
    if best_params is not None and time.time() < deadline - 10:
        try:
            # Use L-BFGS-B on the real parameters
            res = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        return [1.0] * N
        
    h = params_to_h(best_params)
    
    # Ensure non-negativity and return
    h = np.maximum(h, 0.0)
    
    return list(h)
