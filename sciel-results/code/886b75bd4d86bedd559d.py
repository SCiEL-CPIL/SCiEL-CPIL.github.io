"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    # x is the autoconvolution vector
    # We want to approximate max(x)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    # Subtract m for numerical stability
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective for autoconvolution peak.
    h is normalized internally so sum(h) = N.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = np.convolve(h_norm, h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _dct_to_h(coeffs):
    """Convert DCT-II coefficients to time-domain signal."""
    h = idct(coeffs, type=2, norm='ortho')
    return h

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses DCT spectral representation with soft-max surrogate optimization.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with all-ones (score 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Quick parametric families to get a good starting point
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: DCT-based optimization with soft-max surrogate
    # Start from best_h so far
    c_best = dct(best_h, type=2, norm='ortho')
    
    # We'll optimize in DCT space, but fix DC to maintain non-zero sum
    # Actually, let's optimize all coefficients but use a constraint-free approach
    # by normalizing internally
    
    # Two-stage soft-max: low beta first, then high beta
    for beta in [1.0, 5.0, 20.0]:
        if time.time() > deadline - 20:
            break
        
        # Objective in time domain using soft-max
        def obj_time(h):
            return _objective_soft(h, beta)
        
        # Try a few random restarts around best_h
        rng = np.random.RandomState(42 + int(beta))
        
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            # Start from best_h or perturbed version
            if trial == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.1 * rng.randn(N)
                h_init -= np.mean(h_init)  # Keep zero-mean AC
                h_init += 1.0  # Add DC back
                h_init = h_init * (N / np.sum(h_init))
            
            try:
                result = minimize(
                    obj_time,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_val:
                    h_opt = result.x
                    v_opt = _eval_value(h_opt)  # Use exact eval for comparison
                    if v_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 3: Coordinate descent polish in DCT space
    if time.time() < deadline - 10:
        c = dct(best_h, type=2, norm='ortho')
        K = 80  # number of DCT modes to polish
        
        # Coordinate descent with small steps
        step = 0.01
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for k in range(1, K):
                if time.time() > deadline - 5:
                    break
                old = c[k]
                for delta in [step, -step, 2*step, -2*step]:
                    c[k] = old + delta
                    h = _dct_to_h(c)
                    if not np.all(np.isfinite(h)):
                        c[k] = old
                        continue
                    s = np.sum(h)
                    if abs(s) < 1e-10:
                        c[k] = old
                        continue
                    v = _eval_value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        c[k] = old
                if not improved and iteration > 30:
                    step *= 0.5
                    if step < 0.001:
                        break
            if step < 0.001:
                break
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
