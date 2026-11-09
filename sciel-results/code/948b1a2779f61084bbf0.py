"""Refined Cohn-Goncalves Laguerre construction: optimize 12 double-root locations."""
import time
import numpy as np
from scipy.special import roots_laguerre, eval_laguerre
from scipy.optimize import minimize, differential_evolution, minimize_scalar

M = 12          # number of double roots
N = 1           # dimension
DEADLINE = 560.0  # safety margin before 630s kill

# ---- Objective: C4 upper bound for a set of 12 positive double roots ----
# For the Fourier sign uncertainty, the relevant quantity is the integral
#   C4 = int_0^inf  |P(t)|^2  t^{2N} e^{-t} dt
# where P(t) = prod_{j=1}^{M} (t - r_j)^2  is the Laguerre combination.
# We minimize this (lower is better).

def _C4_objective(roots):
    """Compute the C4 upper bound for given sorted positive roots."""
    # P(t) = prod (t - r_j)^2
    # Use log-space to avoid overflow for large t
    # C4 = int_0^inf |P(t)|^2 t^{2N} e^{-t} dt
    # Use Gauss-Laguerre quadrature with weight e^{-t}:
    #   int_0^inf f(t) e^{-t} dt ~ sum_i w_i f(x_i)
    # Here f(t) = |P(t)|^2 t^{2N}
    # Use high-order quadrature for accuracy
    n_quad = 200
    x, w = np.polynomial.laguerre.laggauss(n_quad)
    # P(x_i) = prod (x_i - r_j)^2
    # log|P| = 2 * sum_j log|x_i - r_j|
    diff = x[:, None] - roots[None, :]  # shape (n_quad, M)
    log_abs_P2 = 2.0 * np.sum(np.log(np.abs(diff)), axis=1)
    log_f = log_abs_P2 + 2 * N * np.log(x)  # log(|P|^2 * t^{2N})
    # C4 = sum w_i * exp(log_f_i)
    # Handle overflow: if log_f is too large, C4 is huge
    max_log = np.max(log_f)
    if max_log > 700:
        return np.exp(max_log) * 1e6  # effectively infinite penalty
    f_vals = np.exp(log_f - max_log) * np.exp(max_log)
    C4 = np.sum(w * f_vals)
    return C4


def _objective_vec(roots):
    """Vectorized objective for optimizers; adds small penalty for non-positive or unsorted."""
    if np.any(roots <= 0):
        return 1e10
    roots_sorted = np.sort(roots)
    return _C4_objective(roots_sorted)


def construct():
    t_start = time.time()

    def _elapsed():
        return time.time() - t_start

    # ---- Initial guess: zeros of Laguerre polynomial L_M ----
    # The zeros of L_M are positive and well-distributed.
    # Scale them to a reasonable range.
    x_lag, _ = roots_laguerre(M)
    # x_lag are in (0, ~M+something). Try a few scales.
    best_roots = None
    best_val = np.inf

    # Try several scaling factors for the Laguerre zeros
    scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]
    for s in scales:
        if _elapsed() > DEADLINE - 30:
            break
        roots = s * x_lag
        val = _C4_objective(roots)
        if val < best_val:
            best_val = val
            best_roots = roots.copy()

    # Also try the seed arithmetic progression as a baseline
    seed_roots = np.array([4.0 * (i + 1) for i in range(M)])
    seed_val = _C4_objective(seed_roots)
    if seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots.copy()

    # ---- Refine with local optimization (Nelder-Mead) ----
    if _elapsed() < DEADLINE - 60:
        res = minimize(
            _objective_vec,
            best_roots,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_roots = np.sort(res.x)

    # ---- Try BFGS from multiple starts ----
    if _elapsed() < DEADLINE - 120:
        starts = [best_roots.copy()]
        # Add a few perturbed starts
        for s in [0.5, 2.0, 5.0]:
            starts.append(s * x_lag)
        for start in starts:
            if _elapsed() > DEADLINE - 60:
                break
            try:
                res = minimize(
                    _objective_vec,
                    start,
                    method='BFGS',
                    options={'maxiter': 2000, 'gtol': 1e-10},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_roots = np.sort(res.x)
            except Exception:
                pass

    # ---- Differential evolution (global search) if time permits ----
    if _elapsed() < DEADLINE - 180:
        # Bounds: each root in [0.01, 200]
        bounds = [(0.01, 200.0)] * M
        try:
            res = differential_evolution(
                _objective_vec,
                bounds,
                seed=42,
                maxiter=300,
                tol=1e-10,
                popsize=15,
                polish=True,
                init='sobol',
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.sort(res.x)
        except Exception:
            pass

    # ---- Final high-precision polish ----
    if _elapsed() < DEADLINE - 30:
        try:
            res = minimize(
                _objective_vec,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.sort(res.x)
        except Exception:
            pass

    # Ensure positive and sorted
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)

    return [float(r) for r in best_roots]
