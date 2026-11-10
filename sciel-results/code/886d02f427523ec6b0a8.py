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

    def optimize_weights_ccd(support, weights, max_iters=20, tol=1e-8):
        """
        Optimize weights using Cyclic Coordinate Descent.
        For a fixed support, we maximize P(w) subject to w_i >= 0, sum(w)=1.
        The function P(w) is differentiable w.r.t each w_i.
        We use the fact that if we fix all w_j for j != i, the term involving w_i
        in the expansion of P is linear in w_i (since P is homogeneous of degree 0,
        but effectively we can view it as maximizing a function where the gradient
        w.r.t w_i can be computed).
        
        Actually, let's derive the gradient or use a simpler approach:
        P = sum_{l,k} w_l w_k C_{lk}
        where C_{lk} = sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j.
        
        This is a quadratic form in w? No, C_{lk} depends on w.
        P = sum_{l,k,i,j: s_i+s_j < 2s_l-s_k} w_l w_k w_i w_j.
        This is a degree-4 polynomial in w.
        
        Gradient w.r.t w_m:
        dP/dw_m = 4 * sum_{l,k,i,j: condition} (indicator terms) ...
        
        Specifically, the term w_l w_k w_i w_j contributes to dP/dw_m if m is l, k, i, or j.
        So dP/dw_m = 4 * (sum_{l=k=m} ... + sum_{l=i=m} ... + ...)
        
        Let's precompute the "influence" of each point.
        Actually, a simpler CCD step:
        For each i, fix all other weights. The part of P that depends on w_i is:
        P = w_i * A_i + B_i
        where A_i is the derivative contribution and B_i is the rest.
        Since P is a 4th degree polynomial, this is not linear.
        However, for small n, we can just do a 1D optimization for each w_i?
        Or, more simply, use the gradient ascent step.
        
        Let's compute the gradient numerically or analytically.
        Given the complexity, let's stick to the robust SLSQP for weights but use CCD
        as a fast local refiner if we can derive the gradient.
        
        Alternative: The inspiration used CCD. Let's implement a simple gradient step for weights.
        Gradient of P w.r.t w_m:
        P = sum_{l,k,i,j: s_i+s_j < 2s_l-s_k} w_l w_k w_i w_j
        dP/dw_m = sum_{l,k,i,j: cond} (delta_{ml} + delta_{mk} + delta_{mi} + delta_{mj}) w_l w_k w_i w_j
        
        We can compute this by iterating over all quadruples? No, too slow.
        
        Let's use the structure:
        Let F(l, k) = sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j.
        P = sum_{l,k} w_l w_k F(l, k).
        
        dP/dw_m = sum_{l,k} (delta_{ml} + delta_{mk}) w_l w_k F(l, k) + sum_{l,k} w_l w_k dF(l,k)/dw_m.
        
        dF(l,k)/dw_m = sum_{i,j: cond} (delta_{mi} + delta_{mj}) w_i w_j.
        
        So dP/dw_m = 2 * sum_k w_m w_k F(m, k) + 2 * sum_l w_l w_m F(l, m)  [from first part, delta_ml and delta_mk]
                     + sum_{l,k} w_l w_k [ sum_{j: s_m+s_j < 2s_l-s_k} w_j + sum_{i: s_i+s_m < 2s_l-s_k} w_i ]
        
        This is computable in O(n^3) or O(n^4) if not careful.
        Given n is small (< 20), O(n^4) might be acceptable for a few iterations.
        But let's just use SLSQP for weights as it was working, and use CCD for support?
        Or just keep SLSQP for both and improve the search.
        
        The current program's weakness might be getting stuck in local minima.
        Let's use the vectorized compute_P from the inspiration (which is faster)
        and keep the multi-phase search strategy of the current program.
        """
        return weights  # Placeholder, we will use SLSQP as in current program but with vectorized compute_P

    # We will use the vectorized compute_P from the inspiration.
    # The current program's compute_P was O(n^2 log n) per call due to sorting inside?
    # No, the current program sorted once. The inspiration sorts once too.
    # The difference is the inner loop.
    # Current:
    # for l in range(n):
    #   for k in range(n):
    #     searchsorted...
    #     P += ...
    # Inspiration:
    # for l in range(n):
    #   thresholds = 2*s[l] - s
    #   idxs = searchsorted(sorted_sums, thresholds, 'left')
    #   inner_sums = ...
    #   P += w[l] * sum(w * inner_sums)
    # The inspiration is faster because it vectorizes the inner sum over k for a fixed l.
    # This reduces Python loop overhead.
    
    def optimize_weights(support):
        """Optimize weights for a fixed support."""
        n = len(support)
        w0 = np.ones(n) / n
        
        def objective(w):
            return -compute_P(support, w)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-12, 'iprint': 0})
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
            w = np.maximum(w_raw, 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 1.0
            w = w / w_sum
            return -compute_P(s, w)
        
        bounds = [(0.0, 50.0)] * n + [(0.0, 1.0)] * n
        
        try:
            res = minimize(objective, params_init, method='SLSQP', bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-12, 'iprint': 0})
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

    rng = np.random.default_rng(42)
    
    # Phase 1: Weight optimization for various fixed supports
    # Use a wider range of n and support structures
    support_templates = []
    
    # 1. Uniform grids
    for n in range(3, 21):
        support_templates.append((np.arange(n, dtype=np.float64), n))
        
    # 2. Geometric grids
    for n in range(3, 15):
        support_templates.append((2.0 ** np.arange(n, dtype=np.float64), n))
        
    # 3. Random supports
    for n in range(3, 21):
        for _ in range(5):
            s = np.sort(rng.uniform(0, 50, n))
            support_templates.append((s, n))
            
    # 4. Clustered: many near 0, one large
    for n in range(4, 15):
        s = np.zeros(n)
        s[-1] = 10.0
        s[:-1] = rng.uniform(0, 1, n-1)
        s = np.sort(s)
        support_templates.append((s, n))
        
    # 5. Two-point mixture structures (approximated by discrete points)
    # e.g., mass at 0 and 1
    for n in range(2, 10):
        s = np.zeros(n)
        s[-1] = 1.0
        support_templates.append((s, n))
        
    for s_init, n in support_templates:
        if time.time() > deadline - 30.0:
            break
        # Optimize weights
        w_opt = optimize_weights(s_init)
        try_configuration(s_init, w_opt)

    # Phase 2: Joint optimization for small n
    if time.time() < deadline - 60.0:
        for n in [3, 4, 5, 6, 7, 8]:
            # Try a few initial configs
            for _ in range(10):
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
        
        for i in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb support and weights
            scale = 0.1 * (1.0 - i/500.0)
            s_pert = s_curr + rng.normal(0, scale, n_hc)
            s_pert = np.maximum(s_pert, 0.0)
            w_pert = w_curr + rng.normal(0, scale*0.5, n_hc)
            w_pert = np.maximum(w_pert, 0.0)
            w_sum = np.sum(w_pert)
            if w_sum < 1e-9:
                continue
            w_pert = w_pert / w_sum
            
            # Optionally re-optimize weights for the new support
            if i % 5 == 0:
                w_pert = optimize_weights(s_pert)
            
            score_pert = compute_P(s_pert, w_pert)
            
            if score_pert > score_curr:
                s_curr = s_pert
                w_curr = w_pert
                score_curr = score_pert
                try_configuration(s_curr, w_curr)

    # Fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights
