import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    def c4_bound_from_log_roots(log_r):
        """Compute a proxy for the C4 upper bound from log-spaced roots."""
        if np.any(log_r <= 0) or np.any(np.diff(log_r) <= 1e-12):
            return 1e10
        r = np.exp(log_r)
        # The C4 bound involves the last sign change location and root distribution
        # Use a combination of spread and local density penalties
        spread = log_r[-1] - log_r[0]
        # Penalize clustering: sum of inverse local spacings
        diffs = np.diff(log_r)
        # Soft minimum to avoid division by zero
        min_diff = np.minimum.reduce(diffs)
        clustering_penalty = np.sum(1.0 / (diffs + 1e-10))
        # Encourage uniformity in log-space
        variance_penalty = np.var(log_r)
        return spread + 0.5 * clustering_penalty + 0.1 * variance_penalty

    # Multi-start differential evolution in log-space with 12 DOF
    best_result = None
    best_score = 1e18

    # Try multiple random seeds for diversity
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() - start_time > deadline - 60:
            break

        def objective_12d(log_r):
            elapsed = time.time() - start_time
            if elapsed > deadline:
                return 1e10
            if np.any(log_r <= 0) or np.any(np.diff(log_r) <= 1e-12):
                return 1e10
            return c4_bound_from_log_roots(log_r)

        # Bounds for log-space: roots between 1e-3 and 1e5
        log_bounds = [(-7, 12)] * 12

        try:
            de_res = differential_evolution(
                objective_12d, log_bounds, seed=seed,
                maxiter=200, tol=1e-12, popsize=15,
                polish=False, init='latinhypercube'
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            continue

    if best_result is None:
        # Fallback: geometric sequence
        log_r = np.linspace(0, 5, 12)
        return list(np.exp(log_r))

    # Polish with L-BFGS-B in log-space
    if time.time() - start_time < deadline - 20:
        try:
            polish_res = minimize(
                c4_bound_from_log_roots, best_result,
                method='L-BFGS-B',
                bounds=[(-7, 12)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_res.fun < best_score:
                best_result = polish_res.x
                best_score = polish_res.fun
        except Exception:
            pass

    # Final check: ensure strictly increasing positive roots
    log_r = best_result
    if np.any(log_r <= 0) or np.any(np.diff(log_r) <= 1e-12):
        # Fallback
        log_r = np.linspace(0, 5, 12)

    r = np.exp(log_r)
    return list(r)
