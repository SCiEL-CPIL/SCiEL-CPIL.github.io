import numpy as np
import scipy.optimize as optimize
import time
from itertools import product

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

    def build_precomputed(support):
        """Precompute all support-dependent quantities for fast weight evaluation."""
        n = len(support)
        s_sums = support[:, None] + support[None, :]
        s_sums_flat = s_sums.flatten()
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        
        # Map from flat index (i*n+j) to sorted position
        # For each pair (i,j), we need its position in the sorted array
        # s_sorted[pos] = s_sums[i,j]
        # We need: for threshold t, sum of p_i*p_j over pairs with s_i+s_j < t
        # This is: sum over sorted positions where s_sorted < t of (p_i * p_j)
        
        # Create a mapping: for each sorted position, which (i,j) pair is it?
        # s_sums_flat is flattened as row-major: index = i*n + j
        # sort_idx gives us the original flat indices in sorted order
        pair_i = sort_idx // n
        pair_j = sort_idx % n
        
        s_k_grid = support[:, None]  # (n, 1)
        s_l_grid = support[None, :]  # (1, n)
        thresholds = 2.0 * s_l_grid - s_k_grid  # (n, n)
        
        # For each (k,l) pair, find how many sorted sums are < threshold
        idxs = np.searchsorted(s_sorted, thresholds, side='left')  # (n, n)
        
        return {
            's_sorted': s_sorted,
            'pair_i': pair_i,
            'pair_j': pair_j,
            'thresholds': thresholds,
            'idxs': idxs,
            'n': n
        }
    
    def calc_prob_fast(support, weights, precomp=None):
        """Compute probability using precomputed structure if available."""
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        if precomp is None:
            precomp = build_precomputed(support)
        
        s_sorted = precomp['s_sorted']
        pair_i = precomp['pair_i']
        pair_j = precomp['pair_j']
        idxs = precomp['idxs']
        
        # For each sorted position, the weight is p[pair_i[pos]] * p[pair_j[pos]]
        p_prods_sorted = p[pair_i] * p[pair_j]
        p_prefix = np.cumsum(p_prods_sorted)
        
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        if np.any(mask):
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def calc_prob(support, weights):
        """Compute probability without precomputation (for new supports)."""
        precomp = build_precomputed(support)
        return calc_prob_fast(support, weights, precomp)

    def optimize_weights_fast(support, p, max_iters=200):
        """Optimize weights given fixed support using coordinate ascent with precomputation."""
        n = len(support)
        precomp = build_precomputed(support)
        best_score = calc_prob_fast(support, p, precomp)
        step = 0.05
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                if p[i] < 1e-10:
                    continue
                # Try increasing and decreasing weight i
                for direction in [1, -1]:
                    new_p = p.copy()
                    new_p[i] += direction * step
                    if new_p[i] < 1e-10:
                        continue
                    new_p = np.maximum(new_p, 0)
                    new_p_sum = new_p.sum()
                    if new_p_sum <= 1e-10:
                        continue
                    new_p = new_p / new_p_sum
                    s = calc_prob_fast(support, new_p, precomp)
                    if s > best_score + 1e-15:
                        best_score = s
                        p = new_p.copy()
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-10:
                    break
        return p, best_score

    def simulated_annealing(support, weights, precomp, max_iters=5000, T_init=0.01, T_min=1e-8):
        """Simulated annealing on support points and weights."""
        n = len(support)
        best_score = calc_prob_fast(support, weights, precomp)
        best_sup = support.copy()
        best_w = weights.copy()
        
        current_score = best_score
        current_sup = support.copy()
        current_w = weights.copy()
        
        T = T_init
        cooling_rate = (T_min / T_init) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            T *= cooling_rate
            
            # Randomly perturb a support point or weight
            if np.random.random() < 0.4:
                # Perturb a support point
                i = np.random.randint(1, n - 1)  # Don't perturb 0 or 1
                delta = np.random.normal(0, 0.02)
                new_sup = current_sup.copy()
                new_sup[i] += delta
                # Ensure ordering and bounds
                if i > 0 and new_sup[i] <= new_sup[i-1] + 1e-10:
                    continue
                if i < n-1 and new_sup[i] >= new_sup[i+1] - 1e-10:
                    continue
                if new_sup[i] < 1e-8 or new_sup[i] > 1 - 1e-8:
                    continue
                
                new_precomp = build_precomputed(new_sup)
                new_score = calc_prob_fast(new_sup, current_w, new_precomp)
                
                if new_score > current_score:
                    current_sup = new_sup
                    current_w = current_w  # weights unchanged
                    current_score = new_score
                    precomp = new_precomp
                    if new_score > best_score:
                        best_score = new_score
                        best_sup = new_sup.copy()
                        best_w = current_w.copy()
                else:
                    delta_E = current_score - new_score
                    if delta_E < 0:
                        accept = True
                    else:
                        accept = np.random.random() < np.exp(-delta_E / T)
                    if accept:
                        current_sup = new_sup
                        current_score = new_score
                        precomp = new_precomp
            else:
                # Perturb weights
                i = np.random.randint(n)
                delta = np.random.normal(0, 0.02)
                new_w = current_w.copy()
                new_w[i] += delta
                new_w = np.maximum(new_w, 0)
                w_sum = new_w.sum()
                if w_sum <= 1e-10:
                    continue
                new_w = new_w / w_sum
                
                new_score = calc_prob_fast(current_sup, new_w, precomp)
                
                if new_score > current_score:
                    current_w = new_w
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_sup = current_sup.copy()
                        best_w = new_w.copy()
                else:
                    delta_E = current_score - new_score
                    if delta_E < 0:
                        accept = True
                    else:
                        accept = np.random.random() < np.exp(-delta_E / T)
                    if accept:
                        current_w = new_w
                        current_score = new_score
        
        return best_sup, best_w, best_score

    best_score = -1.0
    best_support = None
    best_weights = None

    # Start with n=3 as a good baseline
    np.random.seed(42)
    
    # n=2: simple
    if time.time() < deadline:
        # P[X1+X2 < 2*X4] for support [0,1] with prob p at 0
        # X4=0 (prob p): threshold = 0, P(X1+X2<0) = 0
        # X4=1 (prob 1-p): threshold = 2, P(X1+X2<2) = 1 (since max sum is 2, strict < 2 means sum < 2)
        # Actually X1+X2 can be 0,1,2. P(X1+X2 < 2) = P(X1+X2 <= 1) = 1 - P(X1=1,X2=1) = 1 - (1-p)^2
        # P = (1-p) * (1 - (1-p)^2) = (1-p) * (2p - p^2) = 2p(1-p) - p^2(1-p)
        # Let's just optimize numerically
        best_p2 = 0.5
        best_s2 = 0.0
        for p in np.linspace(0.01, 0.99, 100):
            s = calc_prob(np.array([0.0, 1.0]), np.array([p, 1-p]))
            if s > best_s2:
                best_s2 = s
                best_p2 = p
        if best_s2 > best_score:
            best_score = best_s2
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([best_p2, 1-best_p2])

    # n=3 with differential evolution
    if time.time() < deadline - 30:
        bounds3 = [(0.01, 0.9), (0.01, 0.9), (0.01, 0.9)]
        def obj3(params):
            p0, pa, a = params
            if p0 < 1e-9 or pa < 1e-9 or p0 + pa > 1 - 1e-9 or a < 1e-9 or a > 1 - 1e-9:
                return 0.0
            return -calc_prob(np.array([0.0, a, 1.0]), np.array([p0, pa, 1.0 - p0 - pa]))
        res3 = optimize.differential_evolution(obj3, bounds=bounds3, seed=42, maxiter=300, tol=1e-6, polish=False)
        score3 = -res3.fun
        p0, pa, a = res3.x
        if p0 > 1e-9 and pa > 1e-9 and p0+pa < 1-1e-9 and a > 1e-9 and a < 1-1e-9:
            support3 = np.array([0.0, a, 1.0])
            weights3 = np.array([p0, pa, 1.0 - p0 - pa])
            s = calc_prob(support3, weights3)
            # Optimize weights
            w_opt, s_opt = optimize_weights_fast(support3, weights3, max_iters=100)
            if s_opt > best_score:
                best_score = s_opt
                best_support = support3.copy()
                best_weights = w_opt.copy()

    # n=4 with differential evolution
    if time.time() < deadline - 60:
        bounds4 = [(0.01, 0.4), (0.01, 0.4), (0.01, 0.4), (0.05, 0.5), (0.5, 0.95)]
        def obj4(params):
            p0, p1, p2, a1, a2 = params
            if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or 
                p0 + p1 + p2 > 1 - 1e-9 or 
                a1 < 1e-9 or a2 < 1e-9 or a1 > 1 - 1e-9 or a2 > 1 - 1e-9):
                return 0.0
            if a1 >= a2:
                return 0.0
            return -calc_prob(np.array([0.0, a1, a2, 1.0]), np.array([p0, p1, p2, 1.0 - p0 - p1 - p2]))
        res4 = optimize.differential_evolution(obj4, bounds=bounds4, seed=42, maxiter=300, tol=1e-6, polish=False)
        score4 = -res4.fun
        p0, p1, p2, a1, a2 = res4.x
        if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and 
            p0+p1+p2 < 1-1e-9 and 
            a1 > 1e-9 and a2 > 1e-9 and a1 < 1-1e-9 and a2 < 1-1e-9 and a1 < a2):
            support4 = np.array([0.0, a1, a2, 1.0])
            weights4 = np.array([p0, p1, p2, 1.0 - p0 - p1 - p2])
            s = calc_prob(support4, weights4)
            w_opt, s_opt = optimize_weights_fast(support4, weights4, max_iters=100)
            if s_opt > best_score:
                best_score = s_opt
                best_support = support4.copy()
                best_weights = w_opt.copy()

    # n=5 with differential evolution
    if time.time() < deadline - 90:
        bounds5 = [(0.01, 0.3), (0.01, 0.3), (0.01, 0.3), (0.01, 0.3), 
                   (0.02, 0.3), (0.2, 0.55), (0.5, 0.9)]
        def obj5(params):
            p0, p1, p2, p3, a1, a2, a3 = params
            if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or p3 < 1e-9 or 
                p0+p1+p2+p3 > 1-1e-9 or 
                a1 < 1e-9 or a2 < 1e-9 or a3 < 1e-9 or 
                a1 > 1-1e-9 or a2 > 1-1e-9 or a3 > 1-1e-9 or a1 >= a2 or a2 >= a3):
                return 0.0
            return -calc_prob(np.array([0.0, a1, a2, a3, 1.0]), np.array([p0, p1, p2, p3, 1.0 - p0 - p1 - p2 - p3]))
        res5 = optimize.differential_evolution(obj5, bounds=bounds5, seed=42, maxiter=200, tol=1e-6, polish=False)
        score5 = -res5.fun
        p0, p1, p2, p3, a1, a2, a3 = res5.x
        if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and p3 > 1e-9 and 
            p0+p1+p2+p3 < 1-1e-9 and 
            a1 > 1e-9 and a2 > 1e-9 and a3 > 1e-9 and 
            a1 < 1-1e-9 and a2 < 1-1e-9 and a3 < 1-1e-9 and a1 < a2 < a3):
            support5 = np.array([0.0, a1, a2, a3, 1.0])
            weights5 = np.array([p0, p1, p2, p3, 1.0 - p0 - p1 - p2 - p3])
            s = calc_prob(support5, weights5)
            w_opt, s_opt = optimize_weights_fast(support5, weights5, max_iters=80)
            if s_opt > best_score:
                best_score = s_opt
                best_support = support5.copy()
                best_weights = w_opt.copy()

    # Now use simulated annealing to refine the best solution found
    if best_support is not None and time.time() < deadline - 20:
        precomp = build_precomputed(best_support)
        sa_iters = min(3000, int((deadline - time.time() - 15) * 20))
        if sa_iters > 100:
            sa_sup, sa_w, sa_score = simulated_annealing(best_support, best_weights, precomp, max_iters=sa_iters, T_init=0.005, T_min=1e-9)
            if sa_score > best_score:
                best_score = sa_score
                best_support = sa_sup
                best_weights = sa_w

    # Final weight optimization
    if best_support is not None and time.time() < deadline - 5:
        w_final, s_final = optimize_weights_fast(best_support, best_weights, max_iters=100)
        if s_final > best_score:
            best_score = s_final
            best_weights = w_final

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    sum_w = np.sum(best_weights)
    if sum_w > 1e-9:
        best_weights = best_weights / sum_w
    else:
        best_weights = np.ones(len(best_support)) / len(best_support)

    return best_support, best_weights
