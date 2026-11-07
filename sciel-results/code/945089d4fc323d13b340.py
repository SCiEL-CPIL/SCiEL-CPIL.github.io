"""Second autocorrelation inequality - optimized."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3)*(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_params(params):
    """Numerically stable: h = exp(params) normalized to sum=1"""
    shifted = params - np.max(params)
    e = np.exp(shifted)
    s = e.sum()
    if s < 1e-300:
        return np.ones(N) / N
    return e / s

def _neg_value(params):
    h = _h_from_params(params)
    return -_compute_value(h)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    rng = np.random.default_rng(42)

    def _try_optimize(h0):
        """Run two-stage optimization: Nelder-Mead then L-BFGS-B polish."""
        nonlocal best_h, best_val
        h0 = np.asarray(h0, dtype=float)
        h0 = np.abs(h0)
        s = h0.sum()
        if s < 1e-300:
            h0 = np.ones(N) / N
        else:
            h0 = h0 / s
        params0 = np.log(h0 + 1e-15)

        # Stage 1: Nelder-Mead for robust global-ish search
        try:
            res1 = minimize(_neg_value, params0, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7,
                                    'adaptive': True})
            cur_params = res1.x
            cur_val = -res1.fun
        except Exception:
            return

        # Stage 2: L-BFGS-B polish with tighter tolerance
        if time.time() < deadline - 15:
            try:
                res2 = minimize(_neg_value, cur_params, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxfun': 5000})
                if -res2.fun > cur_val:
                    cur_params = res2.x
                    cur_val = -res2.fun
            except Exception:
                pass

        if cur_val > best_val:
            best_val = cur_val
            best_h = _h_from_params(cur_params)

    # Diverse initial shapes
    x = np.linspace(-1, 1, N)
    candidates = []
    candidates.append(np.ones(N))                              # uniform
    candidates.append(np.exp(-2 * x**2))                       # Gaussian
    candidates.append(np.exp(-5 * x**2))                       # narrow Gaussian
    candidates.append(np.maximum(0, 1 - 2*np.abs(x)))          # triangular
    candidates.append(np.maximum(0, np.cos(np.pi * x)))        # cosine bump
    candidates.append(np.maximum(0, 1 - 4*x**2))               # parabolic
    candidates.append(np.exp(-x**2) + 0.3 * np.exp(-8*(x-0.5)**2))  # asymmetric
    candidates.append(np.exp(-x**2) + 0.3 * np.exp(-8*(x+0.5)**2))  # asymmetric other
    candidates.append(np.ones(N) * (1 + 0.3 * np.sin(2*np.pi*x)))   # ripple
    candidates.append(np.maximum(0, 1 - 3*np.abs(x)))          # wide triangle
    # Bimodal
    candidates.append(np.exp(-8*(x-0.4)**2) + np.exp(-8*(x+0.4)**2))
    # Plateau
    candidates.append(np.where(np.abs(x) < 0.3, 1.0, 0.1))
    # Random starts
    for _ in range(8):
        candidates.append(np.abs(rng.standard_normal(N)) + 0.05)

    for h0 in candidates:
        if time.time() > deadline - 20:
            break
        _try_optimize(h0)

    # Additional random restarts if time permits
    extra = 0
    while time.time() < deadline - 10 and extra < 15:
        h0 = np.abs(rng.standard_normal(N)) + 0.01
        _try_optimize(h0)
        extra += 1

    # Final high-precision polish on the best solution
    if time.time() < deadline - 5:
        best_params = np.log(best_h + 1e-15)
        try:
            res_final = minimize(_neg_value, best_params, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                         'maxfun': 10000})
            if -res_final.fun > best_val:
                best_val = -res_final.fun
                best_h = _h_from_params(res_final.x)
        except Exception:
            pass

    best_h = best_h / best_h.sum()
    return list(best_h)
# EVOLVE-BLOCK-END
