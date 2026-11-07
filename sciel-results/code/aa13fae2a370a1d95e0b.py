"""Second autocorrelation inequality: maximize value = l2/(norm_1*norm_inf).

Strategy: Optimize in a low-dimensional DCT coefficient space.
The 50 heights are represented as the absolute values of the inverse DCT
of a 20-dimensional coefficient vector. This enforces non-negativity and
allows the optimizer to explore global shape features (symmetry, smoothness)
more efficiently than optimizing the 50 raw heights directly.
We use scipy.optimize.minimize with L-BFGS-B on the negative value.
Multiple diverse restarts are used to escape local minima.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import idct

N = 50
DCT_DIM = 20  # Number of DCT coefficients to optimize

def _eval(h):
    """Exact evaluator (mirrors the public contract)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] * y[:-1] + y[:-1] * y[1:] + y[1:] * y[1:])
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _dct_to_h(coeffs):
    """Convert DCT coefficients to nonnegative heights."""
    # idct expects the full length N, so pad with zeros
    full_coeffs = np.zeros(N)
    full_coeffs[:len(coeffs)] = coeffs
    h = idct(full_coeffs, type=2) * np.sqrt(2.0 / N)
    # Enforce non-negativity
    h = np.abs(h)
    return h

def _objective(coeffs):
    """Objective function to minimize: -value(h)."""
    h = _dct_to_h(coeffs)
    val = _eval(h)
    return -val

def construct():
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin under the 150s limit

    best_val = 0.0
    best_h = np.ones(N)  # Seed: 50 equal heights
    best_coeffs = None

    # Seed value
    best_val = _eval(best_h)

    # Define diverse initializations for DCT coefficients
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Uniform-like (all coefficients similar)
    inits.append(np.ones(DCT_DIM) * 0.5)
    
    # 2. Gaussian-like (low frequency emphasis)
    for _ in range(3):
        c = rng.normal(0, 1, DCT_DIM)
        c *= np.exp(-np.linspace(0, 2, DCT_DIM))  # Decay with frequency
        inits.append(c)
        
    # 3. Triangular-like (mid-frequency emphasis)
    for _ in range(2):
        c = rng.normal(0, 1, DCT_DIM)
        freqs = np.linspace(0, 1, DCT_DIM)
        c *= np.sin(np.pi * freqs)  # Emphasize mid frequencies
        inits.append(c)
        
    # 4. Random
    for _ in range(2):
        inits.append(rng.normal(0, 1, DCT_DIM))

    # Run optimization for each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                _objective, 
                init, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success:
                h = _dct_to_h(res.x)
                val = _eval(h)
                if val > best_val:
                    best_val = val
                    best_h = h
                    best_coeffs = res.x
        except Exception:
            continue

    # If we found a good solution, use it. Otherwise, fallback to seed.
    if best_val > 0:
        return list(best_h)
    else:
        return list(np.ones(N))
