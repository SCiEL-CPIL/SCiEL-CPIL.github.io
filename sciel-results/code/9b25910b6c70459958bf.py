# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] efficiently using sorted pair sums."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support
        
        # Precompute all pairwise sums and their weights
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = s[i_idx] + s[j_idx]
        pair_weights = w[i_idx] * w[j_idx]
        
        # Sort pair sums for efficient threshold queries
        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        cum_weights = np.cumsum(sorted_weights)
        
        P = 0.0
        for l in range(n):
            target = 2.0 * s[l]
            thresholds = target - s
            idxs = np.searchsorted(sorted_sums, thresholds, side='left')
            inner_sums = np.zeros(n)
            mask = idxs > 0
            inner_sums[mask] = cum_weights[idxs[mask] - 1]
            P += w[l] * np.sum(w * inner_sums)
            
        return P

    def objective(params, support_fixed=None):
        """Objective: minimize -P. If support_fixed is None, optimize both support and weights."""
        if support_fixed is not None:
            # Only optimize weights
            w = np.maximum(params, 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                return 1.0
            w = w / w_sum
            return -compute_P(support_fixed, w)
        else:
            # Optimize both support and weights
            n = len(params) // 2
            s_raw = params[:n]
            w_raw = params[n:]
            # Ensure support is non-negative
            s = np.maximum(s_raw, 0.0)
            # Ensure weights are non-negative
            w = np.maximum(w_raw, 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                return 1.0
            w = w / w_sum
            return -compute_P(s, w)

    best_score = 0.0
    best_support = None
    best_weights = None

    def try_configuration(support, weights, label=""):
        nonlocal best_score, best_support, best_weights
        score = compute_P(support, weights)
        if score > best_score:
            best_score = score
            best_support = support.copy()
            best_weights = weights.copy()
        return score

    # Strategy 1: Optimize weights only for various fixed supports
    for n_try in [5, 8, 10, 12, 15, 20, 25]:
        if time.time() > deadline - 30.0:
            break
        
        # Uniform grid
        support = np.arange(n_try, dtype=np.float64)
        w0 = np.ones(n_try) / n_try
        
        # Try multiple initial weight distributions
        for w_init in [
            w0,
            np.array([1.0, 0.5, 0.25, 0.125] + [0.05] * (n_try - 4)) if n_try >= 4 else w0,
            np.array([0.1] * (n_try // 2) + [0.3] * (n_try - n_try // 2)),
            np.linspace(0.1, 0.5, n_try),
            np.linspace(0.5, 0.1, n_try),
        ]:
            w_init = np.maximum(w_init, 0.0)
            w_sum = np.sum(w_init)
            if w_sum < 1e-15:
                continue
            w_init = w_init / w_sum
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0.0, 1.0)] * n_try
            
            try:
                res = minimize(lambda w: objective(w, support), w_init, method='SLSQP', 
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 200, 'ftol': 1e-10})
                if res.success:
                    w_opt = np.maximum(res.x, 0.0)
                    s_opt = np.sum(w_opt)
                    if s_opt > 1e-9:
                        w_opt = w_opt / s_opt
                        score = try_configuration(support, w_opt, f"n={n_try},weights")
            except Exception:
                pass

    # Strategy 2: Optimize both support and weights for small n
    for n_try in [4, 5, 6, 8]:
        if time.time() > deadline - 30.0:
            break
        
        # Multiple initial configurations
        initial_configs = [
            # Uniform support, uniform weights
            (np.arange(n_try, dtype=np.float64), np.ones(n_try) / n_try),
            # Support with a large gap
            (np.array([0.0, 1.0, 2.0, 5.0] + [10.0] * (n_try - 4)), np.ones(n_try) / n_try),
            # Geometric-like support
            (2.0 ** np.arange(n_try, dtype=np.float64), np.ones(n_try) / n_try),
            # Support concentrated near 0
            (np.array([0.0, 0.5, 1.0, 1.5] + [2.0] * (n_try - 4)), np.ones(n_try) / n_try),
        ]
        
        for s_init, w_init in initial_configs:
            if len(s_init) != n_try or len(w_init) != n_try:
                continue
            s_init = np.maximum(s_init, 0.0)
            w_init = np.maximum(w_init, 0.0)
            w_sum = np.sum(w_init)
            if w_sum < 1e-15:
                continue
            w_init = w_init / w_sum
            
            params = np.concatenate([s_init, w_init])
            bounds = [(0.0, 100.0)] * n_try + [(0.0, 1.0)] * n_try
            
            try:
                res = minimize(lambda p: objective(p), params, method='SLSQP',
                              bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-10})
                if res.success:
                    s_opt = np.maximum(res.x[:n_try], 0.0)
                    w_opt = np.maximum(res.x[n_try:], 0.0)
                    s_opt_sum = np.sum(w_opt)
                    if s_opt_sum > 1e-9:
                        w_opt = w_opt / s_opt_sum
                        score = try_configuration(s_opt, w_opt, f"n={n_try},both")
            except Exception:
                pass

    # Strategy 3: Hill climbing with random perturbations
    if time.time() < deadline - 20.0:
        if best_support is None:
            n_hc = 8
            best_support = np.arange(n_hc, dtype=np.float64)
            best_weights = np.ones(n_hc) / n_hc
            best_score = compute_P(best_support, best_weights)
        
        n_hc = len(best_support)
        rng = np.random.default_rng(42)
        
        for iteration in range(500):
            if time.time() > deadline - 10.0:
                break
            
            # Random perturbation
            perturbation_scale = 0.1 * (1.0 - iteration / 500.0)
            
            s_pert = best_support + rng.normal(0, perturbation_scale, n_hc)
            s_pert = np.maximum(s_pert, 0.0)
            
            w_pert = best_weights + rng.normal(0, perturbation_scale * 0.5, n_hc)
            w_pert = np.maximum(w_pert, 0.0)
            w_sum = np.sum(w_pert)
            if w_sum < 1e-15:
                continue
            w_pert = w_pert / w_sum
            
            score = compute_P(s_pert, w_pert)
            if score > best_score:
                best_score = score
                best_support = s_pert.copy()
                best_weights = w_pert.copy()

    # Final fallback
    if best_score == 0.0 or best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n

    # Ensure validity
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    elif np.sum(best_weights) <= 1e-9:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights


# EVOLVE-BLOCK-END
