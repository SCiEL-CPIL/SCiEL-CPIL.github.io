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
    deadline = start_time + 590.0  # Safety margin before 600s kill

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] efficiently using sorted pair sums."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        
        # Precompute all pairwise sums and weights
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = support[i_idx] + support[j_idx]
        pair_weights = w[i_idx] * w[j_idx]
        
        # Sort and compute cumulative sums
        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        cum_weights = np.cumsum(sorted_weights)
        
        # For each l, compute contribution
        P = 0.0
        for l in range(n):
            target = 2.0 * support[l]
            thresholds = target - support
            idxs = np.searchsorted(sorted_sums, thresholds, side='left')
            inner_sums = np.zeros(n)
            mask = idxs > 0
            inner_sums[mask] = cum_weights[idxs[mask] - 1]
            P += w[l] * np.sum(w * inner_sums)
            
        return P

    def optimize_weights_for_support(support, max_iter=200, ftol=1e-10):
        """Optimize weights for a fixed support using SLSQP."""
        n = len(support)
        w0 = np.ones(n) / n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        
        def neg_P(w):
            w_clipped = np.maximum(w, 0.0)
            s = np.sum(w_clipped)
            if s < 1e-15:
                return 1.0
            w_norm = w_clipped / s
            return -compute_P(support, w_norm)
        
        try:
            res = minimize(neg_P, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': ftol})
            if res.success:
                w_opt = np.maximum(res.x, 0.0)
                s = np.sum(w_opt)
                if s > 1e-9:
                    w_opt = w_opt / s
                    return w_opt, compute_P(support, w_opt)
        except Exception:
            pass
        return None, 0.0

    def simulated_annealing_weights(support, w_init, n_iter=5000, T_start=0.01, T_end=0.001):
        """Simulated annealing to optimize weights for fixed support."""
        n = len(support)
        w = w_init.copy()
        best_w = w.copy()
        best_score = compute_P(support, w)
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / n_iter)
        
        for it in range(n_iter):
            T *= cooling
            if time.time() > deadline - 5.0:
                break
            
            # Random perturbation: move mass from one index to another
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = np.random.uniform(-0.1, 0.1)
            w_new = w.copy()
            w_new[i] -= delta
            w_new[j] += delta
            w_new = np.maximum(w_new, 0.0)
            s = np.sum(w_new)
            if s < 1e-15:
                continue
            w_new = w_new / s
            
            score_new = compute_P(support, w_new)
            delta_score = score_new - best_score
            
            if delta_score > 0 or np.random.random() < np.exp(delta_score / T):
                w = w_new
                if score_new > best_score:
                    best_score = score_new
                    best_w = w_new.copy()
        
        return best_w, best_score

    best_score = 0.0
    best_support = None
    best_weights = None

    # Try various support configurations
    support_configs = [
        np.arange(5, dtype=np.float64),
        np.arange(8, dtype=np.float64),
        np.arange(10, dtype=np.float64),
        np.arange(12, dtype=np.float64),
        np.arange(15, dtype=np.float64),
        np.arange(20, dtype=np.float64),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 25], dtype=np.float64),
        np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10], dtype=np.float64),
        np.array([0, 1, 2, 4, 8, 16, 32], dtype=np.float64),
    ]

    for support in support_configs:
        if time.time() > deadline - 30.0:
            break
        
        # First optimize with SLSQP
        w_opt, score = optimize_weights_for_support(support, max_iter=150)
        if w_opt is not None and score > best_score:
            best_score = score
            best_support = support.copy()
            best_weights = w_opt.copy()
        
        # Then try simulated annealing starting from the optimized weights
        if w_opt is not None:
            w_sa, score_sa = simulated_annealing_weights(support, w_opt, n_iter=3000)
            if score_sa > best_score:
                best_score = score_sa
                best_support = support.copy()
                best_weights = w_sa.copy()
    
    # If no better found, fall back to uniform n=32
    if best_score == 0.0 or best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
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
