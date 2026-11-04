import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600

def _score(heights):
    """Compute the normalized maximum of the autoconvolution."""
    h = 1.0 / 1200  # interval width
    f = np.array(heights, dtype=float)
    # Autoconvolution via FFT for efficiency
    # f*f at discrete points: (f * f)[k] = sum_i f[i] f[k-i] * h
    # Using FFT
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * h
    # The autoconvolution on the original domain [-1/4, 1/4] corresponds to indices
    # Actually, the convolution of two functions on [-1/4,1/4] gives a function on [-1/2, 1/2]
    # We need max of the autoconvolution over its domain
    max_conv = np.max(conv)
    # Normalization: likely (∫f)^2 or ||f||_2^2
    integral = np.sum(f) * h
    norm = integral ** 2
    if norm < 1e-15:
        return 1e10
    return max_conv / norm

def _objective(params):
    """Objective for optimization: minimize the score."""
    heights = np.array(params)
    if np.any(heights < -1e-10):
        return 1e10
    heights = np.maximum(heights, 0.0)
    return _score(heights)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial guess: symmetric tent-like function
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/1200  # midpoints
    # Start with a symmetric bump
    init = np.exp(-4 * (x / 0.25) ** 2)
    init = init / np.max(init) if np.max(init) > 0 else init
    
    # Use differential evolution with bounds
    bounds = [(0.0, 1.0)] * N
    
    best_score = _score(init)
    best_heights = init.copy()
    
    # Try a few quick local improvements first
    # Symmetric constraint: enforce symmetry to reduce search space
    half = N // 2
    # Make symmetric
    sym_init = np.zeros(N)
    sym_init[:half] = init[:half]
    sym_init[half:] = init[:half][::-1]
    best_score = _score(sym_init)
    best_heights = sym_init.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return list(best_heights)
    
    # Run differential evolution on half the variables (symmetric)
    def _sym_objective(half_params):
        if time.time() > deadline - 5:
            return 1e10
        h = np.array(half_params)
        if np.any(h < -1e-10):
            return 1e10
        h = np.maximum(h, 0.0)
        full = np.zeros(N)
        full[:half] = h
        full[half:] = h[::-1]
        return _score(full)
    
    sym_bounds = [(0.0, 1.0)] * half
    sym_init_half = sym_init[:half].copy()
    
    try:
        result = differential_evolution(
            _sym_objective,
            bounds=sym_bounds,
            x0=sym_init_half,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True,
            init='sobol'
        )
        if result.fun < best_score:
            h = np.maximum(result.x, 0.0)
            full = np.zeros(N)
            full[:half] = h
            full[half:] = h[::-1]
            best_score = _score(full)
            best_heights = full
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a quick local optimization with Nelder-Mead on the full vector
        from scipy.optimize import minimize
        try:
            res = minimize(
                _objective,
                best_heights,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                h = np.maximum(res.x, 0.0)
                best_score = _score(h)
                best_heights = h
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
