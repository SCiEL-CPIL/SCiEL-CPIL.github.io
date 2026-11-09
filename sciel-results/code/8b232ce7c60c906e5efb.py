import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time

N = 50

def _autoconv_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    ac = fftconvolve(h, h, mode='full')
    return np.max(ac)

def _objective(params):
    """Minimize ||h||_1 * ||h*h||_inf subject to ||h||_2 = 1 and h >= 0."""
    h = np.maximum(params, 0)
    norm2 = np.sqrt(np.sum(h**2))
    if norm2 < 1e-12:
        return 1e10
    h = h / norm2  # normalize to ||h||_2 = 1
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    return l1 * peak

def _make_initialization(method, seed=0):
    """Create diverse initializations."""
    rng = np.random.RandomState(seed)
    if method == 0:
        # Cosine window
        x = np.linspace(-1, 1, N)
        h = 0.5 * (1 + np.cos(np.pi * x))
    elif method == 1:
        # Uniform
        h = np.ones(N)
    elif method == 2:
        # Gaussian-like
        x = np.linspace(-1, 1, N)
        h = np.exp(-4 * x**2)
    elif method == 3:
        # Two peaks
        x = np.linspace(-1, 1, N)
        h = np.exp(-8 * (x - 0.5)**2) + np.exp(-8 * (x + 0.5)**2)
    elif method == 4:
        # Random nonnegative
        h = rng.rand(N)
    elif method == 5:
        # Triangular
        x = np.linspace(-1, 1, N)
        h = np.maximum(0, 1 - np.abs(x))
    elif method == 6:
        # Sparse-ish
        h = np.zeros(N)
        idx = rng.choice(N, size=15, replace=False)
        h[idx] = rng.rand(15)
    elif method == 7:
        # Bump in center
        x = np.linspace(-1, 1, N)
        h = np.exp(-10 * x**2)
    elif method == 8:
        # Sinc-like
        x = np.linspace(-1, 1, N)
        h = np.abs(np.sinc(x))
    else:
        # Broad flat top
        x = np.linspace(-1, 1, N)
        h = np.where(np.abs(x) < 0.7, 1.0, 0.3 * np.exp(-2 * (np.abs(x) - 0.7)**2))
    
    h = np.maximum(h, 0)
    norm = np.sqrt(np.sum(h**2))
    if norm > 1e-12:
        h = h / norm
    return h

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_h = None
    best_score = -1.0
    
    # Try multiple initializations with L-BFGS-B
    n_starts = 9
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        h0 = _make_initialization(i, seed=i * 42)
        
        try:
            result = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            h_opt = np.maximum(result.x, 0)
            norm2 = np.sqrt(np.sum(h_opt**2))
            if norm2 > 1e-12:
                h_opt = h_opt / norm2
            
            # Compute C2
            l1 = np.sum(h_opt)
            peak = _autoconv_peak(h_opt)
            if l1 > 1e-12 and peak > 1e-12:
                c2 = 1.0 / (l1 * peak)  # Since ||h||_2 = 1, ||h||_2^2 = 1
            else:
                c2 = 0.0
            
            if c2 > best_score:
                best_score = c2
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # If we didn't find anything better, fall back to cosine window
    if best_h is None or best_score <= 1.0:
        x = np.linspace(-1, 1, N)
        best_h = 0.5 * (1 + np.cos(np.pi * x))
        norm2 = np.sqrt(np.sum(best_h**2))
        best_h = best_h / norm2
    
    # Final polish: try a few more random perturbations around the best
    if time.time() < deadline - 10:
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbation = rng.randn(N) * 0.1
            h_try = best_h + perturbation
            h_try = np.maximum(h_try, 0)
            norm2 = np.sqrt(np.sum(h_try**2))
            if norm2 > 1e-12:
                h_try = h_try / norm2
            
            l1 = np.sum(h_try)
            peak = _autoconv_peak(h_try)
            if l1 > 1e-12 and peak > 1e-12:
                c2_try = 1.0 / (l1 * peak)
                if c2_try > best_score:
                    best_score = c2_try
                    best_h = h_try.copy()
    
    return list(best_h)
