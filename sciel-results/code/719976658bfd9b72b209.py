import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _eval(h):
    """Compute the objective value (higher is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    # Vectorized l2 computation
    yi = y[:m+1]
    yip = y[1:m+2]
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yip + yip**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _neg_eval(h):
    """Negative objective for minimization."""
    return -_eval(h)

def _lbfgs(h0, deadline):
    """Run L-BFGS-B optimization from initial point h0."""
    def _check_time():
        return time.time() > deadline - 3.0
    
    result = minimize(
        _neg_eval, h0,
        method='L-BFGS-B',
        bounds=[(0, None)] * N,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000,
                 'callback': _check_time}
    )
    h_opt = np.maximum(0, result.x)
    return h_opt, _eval(h_opt)

def _nelder_mead(h0, deadline):
    """Run Nelder-Mead optimization from initial point h0."""
    def _check_time():
        return time.time() > deadline - 3.0
    
    result = minimize(
        _neg_eval, h0,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True,
                 'callback': _check_time}
    )
    h_opt = np.maximum(0, result.x)
    return h_opt, _eval(h_opt)

def _cd(h0, deadline, max_rounds=50):
    """Coordinate descent with adaptive step sizes."""
    h = h0.copy()
    h = np.maximum(0, h)
    current_val = _eval(h)
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        step = 0.1 / (1 + 0.5 * rnd)  # shrinking step
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old = h[i]
            best_delta = 0.0
            best_val = current_val
            # Try several step sizes
            for delta in [step, -step, step*2, -step*2, step*0.5, -step*0.5,
                         step*0.25, -step*0.25]:
                new_val = old + delta
                if new_val < 0:
                    continue
                h[i] = new_val
                val = _eval(h)
                if val > best_val:
                    best_val = val
                    best_delta = delta
            if best_delta != 0:
                h[i] = old + best_delta
                current_val = best_val
                improved = True
            else:
                h[i] = old
        if not improved:
            break
    return h, current_val

def _make_initial_guesses():
    """Generate diverse initial guesses."""
    x = np.linspace(-1, 1, N)
    guesses = []
    
    # Flat
    guesses.append(np.ones(N))
    
    # Gaussians with various widths
    for sigma in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
        guesses.append(np.exp(-x**2 / (2 * sigma**2)))
    
    # Triangular
    guesses.append(np.maximum(0, 1 - np.abs(x)))
    
    # Parabolic
    guesses.append(np.maximum(0, 1 - x**2))
    
    # Bell curve
    guesses.append(1.0 / (1.0 + x**4))
    
    # Sigmoid-like
    guesses.append(1.0 / (1.0 + np.exp(-5*x)))
    guesses.append(1.0 / (1.0 + np.exp(5*x)))
    
    # Cosine bump
    guesses.append(np.maximum(0, np.cos(np.pi * x / 2)))
    
    # Two bumps
    guesses.append(np.exp(-((x-0.5)**2)/0.1) + np.exp(-((x+0.5)**2)/0.1))
    
    # Plateau
    guesses.append(np.where(np.abs(x) < 0.5, 1.0, 0.1))
    
    # Random with smoothing
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        h = rng.rand(N)
        # Smooth
        h = np.convolve(h, np.ones(5)/5, mode='same')
        h = np.maximum(0, h)
        guesses.append(h)
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate initial guesses
    guesses = _make_initial_guesses()
    
    # First pass: L-BFGS-B on all guesses, collect results
    lbfgs_results = []
    for i, h0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break
        h_opt, val = _lbfgs(h0, deadline)
        lbfgs_results.append((val, h_opt))
    
    # Sort by value, keep top ones
    lbfgs_results.sort(key=lambda x: -x[0])
    
    # Second pass: Nelder-Mead on top L-BFGS results and top initial guesses
    best_val = 0.0
    best_h = np.ones(N)
    
    # Check initial guesses
    for h0 in guesses:
        if time.time() > deadline - 30:
            break
        val = _eval(h0)
        if val > best_val:
            best_val = val
            best_h = h0.copy()
    
    # Nelder-Mead on top L-BFGS results
    top_n = min(5, len(lbfgs_results))
    for i in range(top_n):
        if time.time() > deadline - 15:
            break
        val_lbfgs, h_lbfgs = lbfgs_results[i]
        if val_lbfgs > best_val:
            best_val = val_lbfgs
            best_h = h_lbfgs.copy()
        
        h_nm, val_nm = _nelder_mead(h_lbfgs, deadline)
        if val_nm > best_val:
            best_val = val_nm
            best_h = h_nm.copy()
    
    # Coordinate descent on best result
    if time.time() < deadline - 5:
        h_cd, val_cd = _cd(best_h, deadline, max_rounds=30)
        if val_cd > best_val:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 3:
        h_nm2, val_nm2 = _nelder_mead(best_h, deadline)
        if val_nm2 > best_val:
            best_val = val_nm2
            best_h = h_nm2.copy()
    
    # Ensure nonnegativity and finite
    best_h = np.maximum(0, np.nan_to_num(best_h, nan=0.0, posinf=0.0, neginf=0.0))
    
    return [float(x) for x in best_h]
