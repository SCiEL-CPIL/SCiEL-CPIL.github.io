import numpy as np
import time
import scipy.optimize as opt

N = 50
_DEADLINE = time.time() + 138.0  # safety margin under 150s kill

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
    v = _eval(h)
    if not np.isfinite(v):
        return 1e10
    return -v

def _lbfgs(h0, maxiter=300):
    try:
        res = opt.minimize(_neg_eval, h0, method='L-BFGS-B',
                           bounds=[(0, None)] * N,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None:
            return res.x
    except Exception:
        pass
    return h0

def _cd(h0, maxiter=200):
    h = h0.copy()
    step = 0.1
    for _ in range(maxiter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(N):
            old = h[i]
            # Try a few step sizes
            for s in [step, step/2, step/4, -step, -step/2, -step/4]:
                candidate = h.copy()
                candidate[i] = max(0.0, old + s)
                v = _eval(candidate)
                if np.isfinite(v) and v > _eval(h) + 1e-12:
                    h = candidate
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    return h

def _nm(h0, maxiter=100):
    try:
        res = opt.minimize(_neg_eval, h0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.x is not None and np.all(res.x >= -1e-12):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return h0

def _make_initial_guesses(rng, n_guesses):
    guesses = []
    # Uniform
    guesses.append(rng.random(N))
    # All ones (seed)
    guesses.append(np.ones(N))
    # Gaussian-like centered
    x = np.linspace(-0.5, 0.5, N)
    for sigma in [0.1, 0.2, 0.3]:
        guesses.append(np.exp(-x**2 / (2 * sigma**2)))
    # Cosine peaks
    for freq in [1, 2, 3]:
        guesses.append(np.maximum(0, np.cos(2 * np.pi * freq * x)))
    # Two Gaussians
    for sigma in [0.1, 0.15]:
        g1 = np.exp(-(x + 0.2)**2 / (2 * sigma**2))
        g2 = np.exp(-(x - 0.2)**2 / (2 * sigma**2))
        guesses.append(g1 + g2)
    # Concentrated
    for center in [0, N//2, N-1]:
        h = np.zeros(N)
        h[center] = 1.0
        guesses.append(h)
    # Random sparse
    for _ in range(5):
        h = np.zeros(N)
        idx = rng.choice(N, size=5, replace=False)
        h[idx] = 1.0
        guesses.append(h)
    # Random smooth
    for _ in range(5):
        h = rng.random(N)
        # Smooth with a simple moving average
        kernel = np.ones(3) / 3
        h = np.convolve(h, kernel, mode='same')
        h = np.maximum(h, 0)
        guesses.append(h)
    
    # Ensure all are valid
    valid_guesses = []
    for g in guesses:
        g = np.asarray(g, dtype=float)
        if g.size == N and np.all(np.isfinite(g)) and np.all(g >= -1e-12):
            g = np.maximum(g, 0.0)
            if np.sum(g) > 1e-10:
                valid_guesses.append(g)
    
    # Add more random guesses if we have time
    while len(valid_guesses) < n_guesses and time.time() < _DEADLINE - 10:
        h = rng.random(N)
        if np.sum(h) > 1e-10:
            valid_guesses.append(h)
    
    return valid_guesses[:n_guesses]

def construct():
    start_time = time.time()
    best_val = -np.inf
    best_h = None
    
    # Evaluate seed
    seed_h = np.ones(N)
    seed_val = _eval(seed_h)
    best_val = seed_val
    best_h = seed_h.copy()
    
    rng = np.random.default_rng(42)
    guesses = _make_initial_guesses(rng, n_guesses=50)
    
    # Sort guesses by initial value (best first)
    def sort_key(g):
        v = _eval(g)
        return -v if np.isfinite(v) else 1e10
    
    guesses.sort(key=sort_key)
    
    # Process each guess
    for i, h0 in enumerate(guesses):
        if time.time() > _DEADLINE - 5:
            break
        
        # L-BFGS-B
        h1 = _lbfgs(h0, maxiter=300)
        v1 = _eval(h1)
        if np.isfinite(v1) and v1 > best_val:
            best_val = v1
            best_h = h1.copy()
        
        # Coordinate Descent polish
        h2 = _cd(h1, maxiter=100)
        v2 = _eval(h2)
        if np.isfinite(v2) and v2 > best_val:
            best_val = v2
            best_h = h2.copy()
        
        # Nelder-Mead polish
        h3 = _nm(h2, maxiter=50)
        v3 = _eval(h3)
        if np.isfinite(v3) and v3 > best_val:
            best_val = v3
            best_h = h3.copy()
        
        # Final L-BFGS-B polish with tighter tolerance
        h4 = _lbfgs(h3, maxiter=500)
        v4 = _eval(h4)
        if np.isfinite(v4) and v4 > best_val:
            best_val = v4
            best_h = h4.copy()
    
    # If we have time, do a few more random restarts
    while time.time() < _DEADLINE - 10:
        h0 = rng.random(N)
        if np.sum(h0) < 1e-10:
            continue
        h1 = _lbfgs(h0, maxiter=200)
        v1 = _eval(h1)
        if np.isfinite(v1) and v1 > best_val:
            best_val = v1
            best_h = h1.copy()
            # Polish
            h2 = _cd(h1, maxiter=50)
            v2 = _eval(h2)
            if np.isfinite(v2) and v2 > best_val:
                best_val = v2
                best_h = h2.copy()
    
    # Ensure best_h is valid
    if best_h is None or not np.all(np.isfinite(best_h)):
        best_h = seed_h
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
