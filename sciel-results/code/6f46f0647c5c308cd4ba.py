"""
Optimized construct for the second autocorrelation inequality.
Uses DCT-coefficient low-dim optimization + Nelder-Mead refinement.
"""
import time
import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize

N = 50

def _evaluate(h):
    """Compute the task value for a given height vector h (nonnegative)."""
    if len(h) != N:
        return 0.0
    if np.any(~np.isfinite(h)) or np.any(h < 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)  # 2*N - 1 = 99
    y = np.concatenate(([0.0], g, [0.0]))  # length m+2 = 101
    # spacing w
    xs = np.linspace(-0.5, 0.5, m + 2)
    w = xs[1] - xs[0]
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y has length m+2, indices 0..m+1
    # i ranges 0..m
    yi = y[:m+1]
    yip1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yip1 + yip1**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-30:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value


def _dct_to_heights(coeffs, n_coeffs):
    """
    Convert DCT coefficients to nonnegative heights.
    coeffs: array of length n_coeffs (low-frequency DCT coefficients)
    Returns: array of length N, nonnegative.
    """
    # Pad coefficients to full DCT length
    full_coeffs = np.zeros(N)
    full_coeffs[:n_coeffs] = coeffs
    
    # Inverse DCT (type II, normalized)
    h = idct(full_coeffs, type=2, norm='ortho')
    
    # Enforce nonnegativity
    h = np.abs(h)
    
    # Ensure finite
    h = np.where(np.isfinite(h), h, 0.0)
    
    return h


def _value_from_coeffs(coeffs, n_coeffs):
    """Objective: negative of value (for minimization)."""
    h = _dct_to_heights(coeffs, n_coeffs)
    v = _evaluate(h)
    return -v


def _nelder_mead_on_heights(h0, maxiter=500, timeout=10.0):
    """Run Nelder-Mead on the 50-dim height space with log-parametrization."""
    # Log-parametrize: h = exp(x) to keep nonnegativity
    h0 = np.maximum(h0, 1e-8)
    x0 = np.log(h0)
    
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > timeout:
            return 1e10
        h = np.exp(x)
        return -_evaluate(h)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
        best_x = result.x
    except Exception:
        best_x = x0
    
    best_h = np.exp(best_x)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)
    return best_h


def _candidate_supergaussian(kappa, width):
    """Super-Gaussian: h[i] = exp(-((x/width)^kappa))"""
    x = np.linspace(-0.25, 0.25, N)
    h = np.exp(-np.abs(x / width) ** kappa)
    return h


def _candidate_trapezoid(center, half_width, slope_width):
    """Trapezoid shape."""
    x = np.linspace(-0.25, 0.25, N)
    h = np.clip(1.0 - np.abs(x - center) / (half_width + slope_width), 0, 1)
    # Make it more trapezoidal
    h = np.clip(1.0 - np.abs(x - center) / half_width, 0, 1)
    return h


def _candidate_asymmetric(alpha, beta):
    """Asymmetric peak."""
    x = np.linspace(-0.25, 0.25, N)
    xc = 0.0
    h = np.exp(-alpha * np.maximum(x - xc, 0)**2 - beta * np.maximum(xc - x, 0)**2)
    return h


def _candidate_random_smooth(rng, n_points=5):
    """Random smooth curve via interpolation of few points."""
    pts = np.sort(rng.uniform(-0.25, 0.25, n_points))
    vals = rng.uniform(0.1, 1.0, n_points)
    x = np.linspace(-0.25, 0.25, N)
    h = np.interp(x, pts, vals)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse candidates and screen
    candidates = []
    
    # Uniform (seed)
    candidates.append(np.ones(N))
    
    # Super-Gaussian family
    for kappa in [2, 3, 4, 6, 8, 12, 16]:
        for width in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.24]:
            if time.time() > deadline - 30:
                break
            candidates.append(_candidate_supergaussian(kappa, width))
    
    # Trapezoids
    for hw in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            if time.time() > deadline - 30:
                break
            candidates.append(_candidate_trapezoid(center, hw, 0.02))
    
    # Asymmetric
    for alpha in [10, 30, 50, 80]:
        for beta in [10, 30, 50, 80]:
            if alpha != beta:
                if time.time() > deadline - 30:
                    break
                candidates.append(_candidate_asymmetric(alpha, beta))
    
    # Random smooth
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        candidates.append(_candidate_random_smooth(rng, n_points=rng.integers(3, 8)))
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 20:
            break
        h = np.clip(h, 0, None)
        h = np.where(np.isfinite(h), h, 0.0)
        v = _evaluate(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: DCT-coefficient optimization on top candidates
    if time.time() < deadline - 15:
        # Take top few candidates and optimize their DCT coefficients
        scored = []
        for h in candidates[:100]:
            h = np.clip(h, 0, None)
            h = np.where(np.isfinite(h), h, 0.0)
            v = _evaluate(h)
            scored.append((v, h))
        scored.sort(key=lambda x: -x[0])
        
        top_candidates = [h for v, h in scored[:8]]
        # Also add best_h
        if best_h not in top_candidates:
            top_candidates.append(best_h)
        
        for h0 in top_candidates:
            if time.time() > deadline - 10:
                break
            
            # DCT coefficient optimization (low-dim)
            for n_coeffs in [5, 8, 12]:
                if time.time() > deadline - 8:
                    break
                
                # Initialize from current best
                h_init = _dct_to_heights(np.zeros(n_coeffs), n_coeffs)
                # Set DC component to match mean
                coeffs0 = np.zeros(n_coeffs)
                # Use the DCT of h0 as initialization
                h0_clipped = np.clip(h0, 0, None)
                dct_full = dct(h0_clipped, type=2, norm='ortho')
                coeffs0 = dct_full[:n_coeffs].copy()
                
                try:
                    result = minimize(_value_from_coeffs, coeffs0, args=(n_coeffs,),
                                      method='Nelder-Mead',
                                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
                    best_coeffs = result.x
                    h_opt = _dct_to_heights(best_coeffs, n_coeffs)
                    v_opt = _evaluate(h_opt)
                    if v_opt > best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    # Phase 3: Nelder-Mead refinement on the best found
    if time.time() < deadline - 5:
        h_refined = _nelder_mead_on_heights(best_h, maxiter=400, timeout=10.0)
        v_refined = _evaluate(h_refined)
        if v_refined > best_val:
            best_val = v_refined
            best_h = h_refined
    
    # Final safety check
    best_h = np.clip(best_h, 0, None)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)
    
    return list(best_h)
