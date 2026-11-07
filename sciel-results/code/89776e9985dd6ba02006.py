import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # safety margin under 150s kill

def _eval(h):
    h = np.asarray(h, dtype=float)
    if h.size != N:
        return -np.inf
    if not np.all(np.isfinite(h)):
        return -np.inf
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(np.abs(g)) / (m + 1)
    ninf = np.max(np.abs(g))
    if n1 * ninf < 1e-30:
        return -np.inf
    return l2 / (n1 * ninf)

def _neg_eval(h):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    return -_eval(h)

def _cd(h0, budget):
    """L-BFGS-B optimization (replacing coordinate descent for speed)."""
    try:
        bounds = [(0, 10)] * N
        res = minimize(_neg_eval, h0, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8,
                                "maxls": 50, "maxfun": 2000})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _nm(h0, budget):
    """Nelder-Mead optimization."""
    try:
        res = minimize(_neg_eval, h0, method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-10,
                                "adaptive": True})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _gaussian_shapes(t, sigmas):
    shapes = []
    for s in sigmas:
        sh = np.exp(-t**2 / (2 * s**2))
        shapes.append(sh)
    return shapes

def _cosine_shapes(t, widths):
    shapes = []
    for w in widths:
        sh = np.maximum(np.cos(np.pi * t / w), 0.0)
        shapes.append(sh)
    return shapes

def _tapered_shapes(t, alphas):
    shapes = []
    for a in alphas:
        sh = np.maximum(1.0 - a * np.abs(t) * 2.0, 0.0)
        shapes.append(sh)
        sh2 = np.maximum(1.0 - a * t**2, 0.0)
        shapes.append(sh2)
    return shapes

def _generate_guesses():
    """Generate diverse initial guesses for multi-start optimization."""
    t = np.linspace(-0.5, 0.5, N)
    guesses = []
    
    # Uniform
    guesses.append(np.ones(N))
    
    # Gaussian bumps with various widths and centers
    for sigma in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        for center in [-0.3, -0.15, 0.0, 0.15, 0.3]:
            sh = np.exp(-(t - center)**2 / (2 * sigma**2))
            sh = np.maximum(sh, 0)
            if np.max(sh) > 1e-6:
                guesses.append(sh / np.max(sh))
    
    # Cosine bumps
    for width in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]:
        for center in [-0.3, -0.15, 0.0, 0.15, 0.3]:
            sh = np.maximum(np.cos(np.pi * (t - center) / width), 0.0)
            if np.max(sh) > 1e-6:
                guesses.append(sh / np.max(sh))
    
    # Power-law tapers
    for p in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        sh = np.maximum(1.0 - (2.0 * np.abs(t))**p, 0.0)
        if np.max(sh) > 1e-6:
            guesses.append(sh / np.max(sh))
    
    # Exponential tapers
    for a in [1.0, 2.0, 3.0, 5.0, 8.0]:
        sh = np.exp(-a * np.abs(t))
        if np.max(sh) > 1e-6:
            guesses.append(sh / np.max(sh))
    
    # Sine-based shapes
    for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
        sh = np.maximum(np.sin(np.pi * t) ** a, 0.0)
        if np.max(sh) > 1e-6:
            guesses.append(sh / np.max(sh))
        sh2 = np.maximum(np.cos(np.pi * t * 0.5) ** a, 0.0)
        if np.max(sh2) > 1e-6:
            guesses.append(sh2 / np.max(sh2))
    
    # Smooth bumps (Cauchy-like)
    for c in [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
        sh = 1.0 / (1.0 + c * t**2)
        if np.max(sh) > 1e-6:
            guesses.append(sh / np.max(sh))
    
    # Two-bump shapes
    for sigma in [0.1, 0.15, 0.2, 0.3]:
        for gap in [0.1, 0.2, 0.3]:
            sh = np.exp(-(t - gap)**2 / (2 * sigma**2)) + np.exp(-(t + gap)**2 / (2 * sigma**2))
            sh = np.maximum(sh, 0)
            if np.max(sh) > 1e-6:
                guesses.append(sh / np.max(sh))
    
    # Random smooth shapes via Fourier series
    rng = np.random.RandomState(42)
    for _ in range(20):
        coeffs = rng.randn(10)
        sh = np.ones(N)
        for k in range(1, 10):
            sh += coeffs[k] * np.cos(2 * np.pi * k * t)
        sh = np.maximum(sh, 0)
        if np.max(sh) > 1e-6:
            guesses.append(sh / np.max(sh))
    
    return guesses

def construct():
    t = np.linspace(-0.5, 0.5, N)
    
    # Generate initial guesses
    guesses = _generate_guesses()
    
    # Evaluate all initial guesses
    best_h = np.ones(N)
    best_v = _eval(best_h)
    
    for guess in guesses:
        guess = np.asarray(guess, dtype=float)
        guess = np.maximum(guess, 0.0)
        v = _eval(guess)
        if v > best_v:
            best_v = v
            best_h = guess.copy()
    
    # Multi-start L-BFGS-B optimization
    top_guesses = [np.asarray(g, dtype=float) for g in guesses]
    top_guesses.append(best_h.copy())
    
    # Shuffle and take a subset for diversity
    rng = np.random.RandomState(123)
    if len(top_guesses) > 15:
        indices = rng.choice(len(top_guesses), size=15, replace=False)
        top_guesses = [top_guesses[i] for i in indices]
    
    # Allocate time for multiple optimization runs
    time_per_run = 8.0
    remaining = _DEADLINE - time.time() - 20.0
    max_runs = int(remaining // time_per_run)
    max_runs = min(max_runs, 12)
    
    for i in range(min(len(top_guesses), max_runs)):
        if time.time() > _DEADLINE - 20.0:
            break
        
        h0 = top_guesses[i].copy()
        h, v = _cd(h0, time.time() + time_per_run)
        if v > best_v:
            best_v = v
            best_h = h.copy()
    
    # Final Nelder-Mead polish
    if time.time() < _DEADLINE - 10.0:
        h, v = _nm(best_h, time.time() + 10.0)
        if v > best_v:
            best_v = v
            best_h = h.copy()
    
    # One more L-BFGS-B pass
    if time.time() < _DEADLINE - 5.0:
        h, v = _cd(best_h, time.time() + 5.0)
        if v > best_v:
            best_v = v
            best_h = h.copy()
    
    # Deadline guard: return best found
    final = np.maximum(best_h, 0.0)
    final = np.where(np.isfinite(final), final, 0.0)
    
    # Final sanity check
    if not np.all(np.isfinite(final)) or np.any(final < 0):
        final = np.ones(N)
    
    return final.tolist()
