"""Second autocorrelation inequality: maximize (L2^2)/(L1 * sup|autoconv|) over 50 nonneg heights."""
import numpy as np
import time
import random
from scipy.optimize import minimize, Bounds, LinearConstraint

N = 50

# Deadline guard: kill if wall-clock exceeds this
DEADLINE = 115.0  # seconds, safety margin before 120s

def _autoconv(x):
    """Compute autoconvolution of x with itself (circular is not what we want; use linear)."""
    # Linear convolution: (x * x)[k] = sum_i x[i]*x[k-i]
    # Use FFT for speed
    n = len(x)
    size = 2 * n
    fx = np.fft.rfft(x, size)
    auto = np.fft.irfft(fx * fx, size)[:n]
    return auto

def _score(x):
    """Compute the ratio: (sum x^2) / (sum x * max|autoconv|)."""
    x = np.asarray(x, dtype=float)
    l2_sq = np.sum(x ** 2)
    l1 = np.sum(x)
    if l1 < 1e-12:
        return 0.0
    ac = _autoconv(x)
    sup_ac = np.max(np.abs(ac))
    if sup_ac < 1e-12:
        return l2_sq / l1  # degenerate
    return l2_sq / (l1 * sup_ac)

def _score_neg(x):
    """Negative score for minimization."""
    return -_score(x)

def _build_bounds():
    lb = np.zeros(N)
    ub = np.ones(N) * 10.0  # generous upper bound
    return Bounds(lb, ub)

def _initial_guesses(n_starts=20):
    """Generate diverse initial guesses."""
    guesses = []
    # Uniform
    guesses.append(np.ones(N))
    # Triangular
    for center in [0.25, 0.5, 0.75]:
        t = np.linspace(0, 1, N)
        guesses.append(np.maximum(0, 1 - 4 * np.abs(t - center)))
    # Gaussian-like
    for center in [0.3, 0.5, 0.7]:
        t = np.linspace(0, 1, N)
        guesses.append(np.exp(-50 * (t - center) ** 2))
    # Ramp
    guesses.append(np.linspace(0, 1, N))
    guesses.append(np.linspace(1, 0, N))
    # Sparse: few non-zero entries
    for _ in range(5):
        x = np.zeros(N)
        idx = random.sample(range(N), k=random.randint(3, 10))
        x[idx] = 1.0
        guesses.append(x)
    # Random
    for _ in range(10):
        guesses.append(np.random.rand(N))
    return guesses

def _local_optimize(x0, bounds, deadline):
    """Run SLSQP or similar local optimization from x0."""
    if time.time() > deadline:
        return x0, _score(x0)
    
    best_x = np.array(x0, dtype=float)
    best_s = _score(best_x)
    
    # Try SLSQP
    try:
        res = minimize(_score_neg, best_x, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success or res.fun < _score_neg(best_x):
            cand = np.clip(res.x, 0, None)
            s = _score(cand)
            if s > best_s:
                best_x = cand
                best_s = s
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x, best_s
    
    # Polish with L-BFGS-B
    try:
        res2 = minimize(_score_neg, best_x, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14})
        cand = np.clip(res2.x, 0, None)
        s = _score(cand)
        if s > best_s:
            best_x = cand
            best_s = s
    except Exception:
        pass
    
    return best_x, best_s

def _refine_with_coordinate_descent(x, max_iters=500, deadline=None):
    """Coordinate descent refinement."""
    x = np.array(x, dtype=float)
    bounds = _build_bounds()
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_val = x[i]
            # Optimize over x[i] in [0, ub[i]]
            def obj(v):
                x[i] = v
                return -_score(x)
            from scipy.optimize import minimize_scalar
            try:
                res = minimize_scalar(obj, bounds=(0.0, bounds[1][i]), method='bounded',
                                      options={'xatol': 1e-14})
                if res.success:
                    new_val = res.x
                    if abs(new_val - old_val) > 1e-14:
                        x[i] = new_val
                        improved = True
                else:
                    x[i] = old_val
            except Exception:
                x[i] = old_val
        if not improved:
            break
    return x

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    bounds = _build_bounds()
    best_x = np.ones(N)
    best_s = _score(best_x)
    
    # Multi-start local optimization
    guesses = _initial_guesses(n_starts=20)
    random.seed(42)
    np.random.seed(42)
    
    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 5:
            break
        x_opt, s = _local_optimize(x0, bounds, deadline - 3)
        if s > best_s:
            best_s = s
            best_x = x_opt.copy()
    
    # Additional random restarts if time permits
    while time.time() < deadline - 10:
        x0 = np.random.rand(N)
        # Scale to have similar L1
        x0 = x0 / np.sum(x0) * N
        x_opt, s = _local_optimize(x0, bounds, deadline - 3)
        if s > best_s:
            best_s = s
            best_x = x_opt.copy()
    
    # Final coordinate descent polish
    if time.time() < deadline - 2:
        best_x = _refine_with_coordinate_descent(best_x, max_iters=200, deadline=deadline - 1)
        final_s = _score(best_x)
        if final_s > best_s:
            best_s = final_s
    
    # Ensure non-negative
    best_x = np.clip(best_x, 0, None)
    
    return list(best_x)
