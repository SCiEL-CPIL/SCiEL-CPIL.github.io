import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0

    def calc_prob(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        p_prefix = np.cumsum(p_sorted)
        
        # Thresholds for X3, X4
        # Condition: X1 + X2 < 2*X4 - X3
        # Let s_k be value of X3, s_l be value of X4.
        # Threshold is 2*s_l - s_k.
        # We sum p_k * p_l * P(Z < 2*s_l - s_k)
        
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # Searchsorted on sorted s_sums
        # Number of elements < threshold
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        if np.any(mask):
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def compute_score(params):
        """
        params: [p_0, ..., p_{n-1}] (weights, sum to 1)
        support is fixed.
        """
        if len(params) < 2:
            return 0.0
        p = np.array(params, dtype=np.float64)
        # Normalize to handle slight numerical drift if passed directly
        s = p.sum()
        if s <= 1e-9:
            return 0.0
        p = p / s
        return calc_prob(support_global, p)

    # Global support for the grid approach
    # We will use a grid of support points.
    # Let's try a grid from 0 to some max_val.
    # Based on the structure, points near 0 and 1 are important.
    # Let's use a non-uniform grid or a uniform grid on [0, 1] or [0, 2].
    
    # Strategy: 
    # 1. Define a support grid.
    # 2. Use projected gradient ascent on the simplex to maximize P.
    #    Since P is not convex in p, gradient ascent might get stuck, but it's a different family than DE on small n.
    #    Actually, P is a 4th order polynomial in p. 
    #    Let's analyze the gradient.
    #    P = sum_{k,l} p_k p_l sum_{i,j: s_i+s_j < 2s_l-s_k} p_i p_j
    #    This is sum_{k,l,i,j} M_{k,l,i,j} p_k p_l p_i p_j where M is 1 if condition holds.
    #    Gradient wrt p_m:
    #    dP/dp_m = 4 * sum_{k,l,i,j: m in {k,l,i,j}} ... 
    #    More simply, let F(p) = P.
    #    F(p) = sum_{a,b,c,d} C_{a,b,c,d} p_a p_b p_c p_d
    #    where C_{a,b,c,d} = 1 if s_a + s_b < 2 s_d - s_c (mapping k=a, l=d, i=b, j=c? No, indices are symmetric in pairs).
    #    Let's stick to the formula:
    #    P = sum_{k,l} p_k p_l Q_{k,l}
    #    where Q_{k,l} = sum_{i,j: s_i+s_j < 2s_l-s_k} p_i p_j.
    #    Q_{k,l} is a quadratic form in p.
    #    So P is a quartic form.
    
    #    Gradient:
    #    dP/dp_m = sum_{k,l} (delta_{m,k} p_l + p_k delta_{m,l}) Q_{k,l} + sum_{k,l} p_k p_l dQ_{k,l}/dp_m
    #    dQ_{k,l}/dp_m = sum_{i,j: cond} (delta_{m,i} p_j + p_i delta_{m,j})
    #    = 2 * sum_{j: s_m+s_j < 2s_l-s_k} p_j + 2 * sum_{i: s_i+s_m < 2s_l-s_k} p_i
    #    Since condition is symmetric in i,j, the two sums are identical.
    #    dQ_{k,l}/dp_m = 4 * sum_{j: s_m+s_j < 2s_l-s_k} p_j  (Wait, check symmetry)
    #    Condition: s_i + s_j < 2s_l - s_k.
    #    Term 1: i=m. sum_j p_j where s_m + s_j < ...
    #    Term 2: j=m. sum_i p_i where s_i + s_m < ...
    #    These are the same sum. So dQ_{k,l}/dp_m = 2 * (sum_{j: s_m+s_j < 2s_l-s_k} p_j).
    #    
    #    So dP/dp_m = sum_{k,l} (delta_{m,k} p_l + p_k delta_{m,l}) Q_{k,l} + sum_{k,l} p_k p_l * 2 * R_{m,k,l}
    #    where R_{m,k,l} = sum_{j: s_m+s_j < 2s_l-s_k} p_j.
    #    
    #    This is O(n^4) to compute naively. With n=100, n^4 = 10^8, which is too slow for many iterations.
    #    We need a faster gradient or a different method.
    
    #    Alternative: Use the fast O(n^2 log n) evaluation but approximate gradient with finite differences?
    #    Finite differences for n=100 requires 100 evaluations per step.
    #    Each evaluation is O(n^2 log n) ~ 10^4 * 10 = 10^5 ops.
    #    100 * 10^5 = 10^7 ops per step.
    #    1000 steps = 10^10 ops. Too slow.
    
    #    Let's try smaller n for gradient ascent, e.g., n=20.
    #    n=20, n^2=400. Eval is fast.
    #    Finite diff: 20 evals per step.
    #    1000 steps * 20 * (20^2 log 20) ~ 1000 * 20 * 400 * 5 = 4e7. Feasible.
    
    #    Or, we can derive the gradient more efficiently.
    #    Let's try n=30 with finite differences.
    
    #    Let's also try a grid search / random restarts with local refinement.
    
    #    Plan:
    #    1. Try small n (2, 3, 4) with DE (as before) to get a good baseline.
    #    2. Try n=10, 20 with Projected Gradient Ascent using finite differences.
    #    3. Return the best.

    best_score = -1.0
    best_support = None
    best_weights = None

    # 1. Small n DE (fast)
    def run_de(n):
        if time.time() > deadline:
            return None, None, -1.0
        
        # Support points for n=2,3,4 are fixed or partially fixed in previous code.
        # For general n, we can optimize support and weights together?
        # DE on support+weights is high dimensional.
        # Let's stick to optimizing weights on a fixed support for n>4.
        # For n<=4, optimize both.
        
        if n == 2:
            support = np.array([0.0, 1.0])
            def obj(p):
                return -calc_prob(support, np.array([p[0], 1-p[0]]))
            import scipy.optimize as optimize
            res = optimize.differential_evolution(obj, bounds=[(1e-6, 1-1e-6)], seed=42, maxiter=500)
            if res.success:
                w = np.array([res.x[0], 1-res.x[0]])
                s = calc_prob(support, w)
                return support, w, s
            return None, None, -1.0
        
        elif n == 3:
            def obj(p):
                p0, pa, a = p
                if p0 < 1e-9 or pa < 1e-9 or p0+pa > 1-1e-9 or a < 1e-9 or a > 1-1e-9:
                    return 0.0
                support = np.array([0.0, a, 1.0])
                w = np.array([p0, pa, 1-p0-pa])
                return -calc_prob(support, w)
            import scipy.optimize as optimize
            res = optimize.differential_evolution(obj, bounds=[(1e-6, 0.9), (1e-6, 0.9), (1e-6, 0.9)], seed=42, maxiter=500)
            if res.success:
                p0, pa, a = res.x
                if p0 > 1e-9 and pa > 1e-9 and p0+pa < 1-1e-9 and a > 1e-9 and a < 1-1e-9:
                    support = np.array([0.0, a, 1.0])
                    w = np.array([p0, pa, 1-p0-pa])
                    s = calc_prob(support, w)
                    return support, w, s
            return None, None, -1.0

        elif n == 4:
            def obj(p):
                p0, p1, p2, a1, a2 = p
                if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or 
                    p0+p1+p2 > 1-1e-9 or 
                    a1 < 1e-9 or a2 < 1e-9 or a1 > 1-1e-9 or a2 > 1-1e-9):
                    return 0.0
                support = np.array([0.0, a1, a2, 1.0])
                w = np.array([p0, p1, p2, 1-p0-p1-p2])
                return -calc_prob(support, w)
            import scipy.optimize as optimize
            res = optimize.differential_evolution(obj, bounds=[(1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.9), (0.1, 0.9)], seed=42, maxiter=500)
            if res.success:
                p0, p1, p2, a1, a2 = res.x
                if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and 
                    p0+p1+p2 < 1-1e-9 and 
                    a1 > 1e-9 and a2 > 1e-9 and a1 < 1-1e-9 and a2 < 1-1e-9):
                    support = np.array([0.0, a1, a2, 1.0])
                    w = np.array([p0, p1, p2, 1-p0-p1-p2])
                    s = calc_prob(support, w)
                    return support, w, s
            return None, None, -1.0
        return None, None, -1.0

    for n in [2, 3, 4]:
        if time.time() > deadline:
            break
        s_sup, s_w, s_score = run_de(n)
        if s_score > best_score:
            best_score = s_score
            best_support = s_sup.copy()
            best_weights = s_w.copy()

    # 2. Projected Gradient Ascent on fixed support grid
    # Try different grid sizes and support ranges.
    
    def run_pga(n, support_range, seed=42):
        if time.time() > deadline:
            return None, None, -1.0
        
        support = np.linspace(support_range[0], support_range[1], n)
        
        # Initialize weights
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(n))
        
        # Finite difference gradient
        eps = 1e-7
        max_iter = 200
        lr = 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradient
            grad = np.zeros(n)
            for i in range(n):
                p_plus = p.copy()
                p_plus[i] += eps
                # Renormalize? No, finite diff on unnormalized then normalize?
                # Better to work on simplex directly.
                # Move p_i by eps, decrease others proportionally?
                # Or just use central difference on the normalized function?
                # Let's use the definition: dP/dp_i on the simplex.
                # Approximate by moving p_i to p_i + eps and scaling others to sum to 1.
                p_plus = p.copy()
                p_plus[i] += eps
                s_plus = p_plus.sum()
                p_plus = p_plus / s_plus
                
                p_minus = p.copy()
                p_minus[i] -= eps
                if p_minus[i] < 0:
                    p_minus[i] = 0
                s_minus = p_minus.sum()
                if s_minus > 0:
                    p_minus = p_minus / s_minus
                else:
                    p_minus = p.copy()
                    p_minus[i] = 0
                    s_minus = p_minus.sum()
                    p_minus = p_minus / s_minus
                
                f_plus = calc_prob(support, p_plus)
                f_minus = calc_prob(support, p_minus)
                grad[i] = (f_plus - f_minus) / (2*eps)
            
            # Projected gradient step
            p_new = p + lr * grad
            # Project to simplex
            # Simple projection: clip and normalize
            p_new = np.maximum(p_new, 0)
            s_new = p_new.sum()
            if s_new < 1e-9:
                p_new = np.ones(n) / n
            else:
                p_new = p_new / s_new
            
            f_new = calc_prob(support, p_new)
            if f_new > best_score:
                # Update global best
                pass # We return at the end of the function for this specific run
                
            p = p_new
            
            # Check convergence
            if np.max(np.abs(grad)) < 1e-6:
                break
        
        s_final = calc_prob(support, p)
        return support, p, s_final

    # Try PGA with n=10, 20, 50
    grid_configs = [
        (10, (0.0, 1.0)),
        (20, (0.0, 1.0)),
        (10, (0.0, 2.0)),
        (20, (0.0, 2.0)),
        (50, (0.0, 1.0)),
    ]
    
    for n, rng_range in grid_configs:
        if time.time() > deadline - 10:
            break
        s_sup, s_w, s_score = run_pga(n, rng_range)
        if s_score > best_score:
            best_score = s_score
            best_support = s_sup.copy()
            best_weights = s_w.copy()
            
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Ensure non-negative and normalized
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    sum_w = np.sum(best_weights)
    if sum_w > 1e-9:
        best_weights = best_weights / sum_w
    else:
        best_weights = np.ones(len(best_support)) / len(best_support)

    return best_support, best_weights
