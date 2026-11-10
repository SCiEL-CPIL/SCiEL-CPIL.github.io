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
        """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support
        
        # Precompute all pairwise sums and their weights
        # pair_sums[i,j] = s[i] + s[j]
        # pair_weights[i,j] = w[i] * w[j]
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
        # Sum over l (X4) and k (X1)
        # Inner sum is over i,j (X2,X3) such that s[i]+s[j] < 2*s[l] - s[k]
        for l in range(n):
            target_base = 2.0 * s[l]
            for k in range(n):
                threshold = target_base - s[k]
                # Find number of pairs (i,j) with sum < threshold
                # searchsorted returns index of first element >= threshold
                # We want sum of weights where sum < threshold, so index - 1
                idx = np.searchsorted(sorted_sums, threshold, side='left')
                if idx > 0:
                    inner_sum = cum_weights[idx - 1]
                else:
                    inner_sum = 0.0
                P += w[l] * w[k] * inner_sum
            
        return P

    def optimize_weights(support):
        """Optimize weights for a fixed support."""
        n = len(support)
        # Initial guess: uniform
        w0 = np.ones(n) / n
        
        def objective(w):
            # w are raw weights, normalize inside compute_P
            # But SLSQP works better with constraints
            # Let's use bounds and no explicit sum constraint, normalize in compute_P
            # Actually, compute_P normalizes. So we can just maximize P(w)
            # But P(w) is homogeneous of degree 0 in w? 
            # P = sum w_k w_l ... / (sum w)^2 ? No.
            # The formula in compute_P uses w = weights / sum(weights).
            # So P is scale invariant.
            # We can fix sum(w)=1 to avoid scaling issues in optimizer.
            return -compute_P(support, w)
        
        # Use SLSQP with sum constraint
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-9, 1.0)] * n  # Avoid zero weights to keep it simple, or (0,1)
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 200, 'ftol': 1e-10, 'iprint': 0})
            if res.success:
                w_opt = np.maximum(res.x, 0.0)
                s_opt = np.sum(w_opt)
                if s_opt > 1e-9:
                    w_opt = w_opt / s_opt
                    return w_opt
        except Exception:
            pass
        return w0

    def optimize_joint(support_init, weights_init):
        """Optimize both support and weights for small n."""
        n = len(support_init)
        params_init = np.concatenate([support_init, weights_init])
        
        def objective(params):
            s_raw = params[:n]
            w_raw = params[n:]
            s = np.maximum(s_raw, 0.0)
            # Sort support to ensure consistent ordering? 
            # The problem is symmetric in indices, but support points are distinct.
            # If we allow support to cross, the mapping of weights to support changes.
            # To handle this, we should sort support and weights together?
            # No, weights are tied to specific support points.
            # If s[0] > s[1], it's just a different configuration.
            # However, compute_P assumes s is the support array.
            # It doesn't require sorting.
            # But for stability, let's sort s and permute w accordingly?
            # Actually, the optimizer might swap them.
            # Let's just clamp to 0.
            w = np.maximum(w_raw, 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 1.0
            w = w / w_sum
            return -compute_P(s, w)
        
        bounds = [(0.0, 100.0)] * n + [(0.0, 1.0)] * n
        
        try:
            res = minimize(objective, params_init, method='SLSQP', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10, 'iprint': 0})
            if res.success:
                s_opt = np.maximum(res.x[:n], 0.0)
                w_opt = np.maximum(res.x[n:], 0.0)
                w_sum = np.sum(w_opt)
                if w_sum > 1e-9:
                    w_opt = w_opt / w_sum
                    return s_opt, w_opt
        except Exception:
            pass
        return support_init, weights_init

    best_score = 0.0
    best_support = None
    best_weights = None

    def try_configuration(support, weights):
        nonlocal best_score, best_support, best_weights
        if np.any(support < -1e-9) or np.any(weights < -1e-9):
            return
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return
        weights = weights / w_sum
        score = compute_P(support, weights)
        if score > best_score:
            best_score = score
            best_support = support.copy()
            best_weights = weights.copy()

    # Phase 1: Weight optimization for various fixed supports
    # Try different n and support structures
    rng = np.random.default_rng(42)
    
    support_templates = []
    
    # 1. Uniform grids
    for n in range(3, 16):
        support_templates.append((np.arange(n, dtype=np.float64), n))
        
    # 2. Geometric grids
    for n in range(3, 12):
        support_templates.append((2.0 ** np.arange(n, dtype=np.float64), n))
        
    # 3. Random supports
    for n in range(3, 16):
        for _ in range(3):
            s = np.sort(rng.uniform(0, 50, n))
            support_templates.append((s, n))
            
    # 4. Clustered: many near 0, one large
    for n in range(4, 12):
        s = np.zeros(n)
        s[-1] = 10.0
        s[:-1] = rng.uniform(0, 1, n-1)
        s = np.sort(s)
        support_templates.append((s, n))

    for s_init, n in support_templates:
        if time.time() > deadline - 20.0:
            break
        # Optimize weights
        w_opt = optimize_weights(s_init)
        try_configuration(s_init, w_opt)

    # Phase 2: Joint optimization for small n
    if time.time() < deadline - 30.0:
        for n in [3, 4, 5, 6]:
            # Try a few initial configs
            for _ in range(5):
                s_init = np.sort(rng.uniform(0, 20, n))
                w_init = np.ones(n) / n
                s_opt, w_opt = optimize_joint(s_init, w_init)
                try_configuration(s_opt, w_opt)
                
                # Also try with 0 included
                s_init[0] = 0.0
                s_init = np.sort(s_init)
                s_opt, w_opt = optimize_joint(s_init, w_init)
                try_configuration(s_opt, w_opt)

    # Phase 3: Hill Climbing on the best result
    if best_support is not None and time.time() < deadline - 10.0:
        n_hc = len(best_support)
        s_curr = best_support.copy()
        w_curr = best_weights.copy()
        score_curr = best_score
        
        for i in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb support
            s_pert = s_curr + rng.normal(0, 0.1, n_hc)
            s_pert = np.maximum(s_pert, 0.0)
            # Sort support? If we sort, we must reorder weights.
            # Let's not sort to keep weight-support association stable during local search,
            # unless we also optimize weights.
            # Actually, compute_P doesn't care about order.
            # But if s_pert[0] > s_pert[1], it's just a permutation.
            # Let's optimize weights for this new support.
            w_pert = optimize_weights(s_pert)
            score_pert = compute_P(s_pert, w_pert)
            
            if score_pert > score_curr:
                s_curr = s_pert
                w_curr = w_pert
                score_curr = score_pert
                try_configuration(s_curr, w_curr)
            else:
                # Try smaller step?
                pass

    # Fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights
