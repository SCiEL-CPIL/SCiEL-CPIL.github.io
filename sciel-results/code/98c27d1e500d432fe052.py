# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64.
    
    P = sum_{k,l} p_k * p_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j)
    """
    n = len(support)
    if n == 0:
        return 0.0
    s = support.astype(np.float64)
    w = weights.astype(np.float64)
    total_w = w.sum()
    if total_w <= 1e-9:
        return 0.0
    p = w / total_w
    
    # Pairwise sums and probabilities
    ss = s[:, None] + s[None, :]  # (n, n)
    pp = p[:, None] * p[None, :]  # (n, n)
    
    # Sort pair sums for efficient threshold queries
    ss_flat = ss.ravel()
    pp_flat = pp.ravel()
    order = np.argsort(ss_flat, kind='mergesort')
    ss_sorted = ss_flat[order]
    pp_sorted = pp_flat[order]
    pp_cumsum = np.cumsum(pp_sorted)
    
    # Vectorize over (k, l) pairs
    # Threshold matrix: T[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n, n), T[k,l]
    
    # For each (k,l), find idx = searchsorted(ss_sorted, threshold, 'left')
    # Then pair_prob = pp_cumsum[idx-1] if idx > 0 else 0
    # Use np.searchsorted for all thresholds at once
    idxs = np.searchsorted(ss_sorted, thresholds.ravel(), side='left')
    idxs = idxs.reshape(n, n)
    
    # Get pair probabilities
    # For idx=0, pair_prob=0; for idx>0, pair_prob=pp_cumsum[idx-1]
    safe_idxs = np.maximum(idxs - 1, 0)
    pair_probs = pp_cumsum[safe_idxs]
    pair_probs[idxs == 0] = 0.0
    
    # P = sum_{k,l} p[k] * p[l] * pair_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * pair_probs)
    return float(P)


def _optimize_weights_for_support(support, maxiter=500):
    """Optimize weights for a fixed support using Nelder-Mead."""
    n = len(support)
    s = support.astype(np.float64)
    
    def neg_P(w_raw):
        w = np.maximum(w_raw, 0.0)
        total = w.sum()
        if total <= 1e-9:
            return 0.0
        w_norm = w / total
        return -_compute_P(s, w_norm)
    
    # Multiple random restarts
    rng = np.random.default_rng(12345)
    best_val = 0.0
    best_w = None
    
    for restart in range(10):
        if restart == 0:
            w0 = np.ones(n) / n
        else:
            w0 = rng.dirichlet(np.ones(n))
        
        res = minimize(
            neg_P, w0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        w_opt = np.maximum(res.x, 0.0)
        if w_opt.sum() > 1e-9:
            w_opt = w_opt / w_opt.sum()
            val = _compute_P(s, w_opt)
            if val > best_val:
                best_val = val
                best_w = w_opt.copy()
    
    return best_w, best_val


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Maximize P[X1 + X2 + X3 < 2*X4].
    
    Strategy: 
    1. Try structured 2-point and 3-point supports with optimized weights
    2. Try small fixed supports (n=4,5,6) with optimized weights
    3. Use Nelder-Mead to jointly optimize support and weights for small n
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s margin
    
    best_p = 0.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts):
        nonlocal best_p, best_support, best_weights
        if len(sup) != len(wts) or len(sup) == 0:
            return
        if np.any(np.isnan(sup)) or np.any(np.isinf(sup)):
            return
        if np.any(np.isnan(wts)) or np.any(np.isinf(wts)):
            return
        if np.any(sup < -1e-9):
            return
        if np.any(wts < -1e-9):
            return
        if wts.sum() <= 1e-9:
            return
        p_val = _compute_P(sup, wts)
        if p_val > best_p:
            best_p = p_val
            best_support = sup.copy()
            best_weights = wts.copy()
    
    # === Phase 1: 2-point distributions ===
    # Support: [0, a], weights: [1-p, p]
    # Analytical: P = P[X1+X2+X3 < 2*X4]
    # X can be 0 (prob 1-p) or a (prob p)
    # X4 = 0: need X1+X2+X3 < 0, impossible since all >= 0. Contribution: 0
    # X4 = a: need X1+X2+X3 < 2a
    #   X1+X2+X3 can be: 0, a, 2a, 3a
    #   0 < 2a: yes (prob (1-p)^3)
    #   a < 2a: yes (prob 3*p*(1-p)^2)
    #   2a < 2a: no (strict inequality)
    #   3a < 2a: no
    # So P = p * [(1-p)^3 + 3*p*(1-p)^2] = p*(1-p)^2*[(1-p) + 3p] = p*(1-p)^2*(1+2p)
    
    def two_point_P(p, a):
        # a doesn't matter for 2-point [0,a] as long as a > 0
        return p * (1-p)**2 * (1 + 2*p)
    
    # Optimize over p in (0,1)
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda p: -two_point_P(p, 1.0), bounds=(0.01, 0.99), method='bounded')
    p_opt = res.x
    s_two = np.array([0.0, 1.0], dtype=np.float64)
    w_two = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    update_best(s_two, w_two)
    
    # === Phase 2: 3-point distributions ===
    # Support: [0, a, b] with a < b
    # Weights: [p0, p1, p2]
    # Fix b=1 by scale invariance, optimize over a, p0, p1, p2
    
    def three_point_obj(params):
        a, p0, p1 = params
        a = max(0.01, min(0.99, a))
        p0 = max(0.0, p0)
        p1 = max(0.0, p1)
        p2 = max(0.0, 1.0 - p0 - p1)
        s = np.array([0.0, a, 1.0], dtype=np.float64)
        w = np.array([p0, p1, p2], dtype=np.float64)
        if w.sum() <= 1e-9:
            return 0.0
        return -_compute_P(s, w)
    
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        x0 = np.array([
            rng.uniform(0.2, 0.8),
            rng.uniform(0.1, 0.5),
            rng.uniform(0.1, 0.5)
        ])
        res = minimize(three_point_obj, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
        a, p0, p1 = res.x
        a = max(0.01, min(0.99, a))
        p0 = max(0.0, p0)
        p1 = max(0.0, p1)
        p2 = max(0.0, 1.0 - p0 - p1)
        s_three = np.array([0.0, a, 1.0], dtype=np.float64)
        w_three = np.array([p0, p1, p2], dtype=np.float64)
        if w_three.sum() > 1e-9:
            w_three = w_three / w_three.sum()
            update_best(s_three, w_three)
    
    # === Phase 3: 4-point distributions with optimized weights ===
    # Try several fixed support structures
    support_structures_4 = [
        np.array([0.0, 0.3, 0.6, 1.0]),
        np.array([0.0, 0.25, 0.5, 1.0]),
        np.array([0.0, 0.4, 0.7, 1.0]),
        np.array([0.0, 0.33, 0.66, 1.0]),
        np.array([0.0, 0.2, 0.5, 1.0]),
        np.array([0.0, 0.5, 0.75, 1.0]),
    ]
    
    for s_struct in support_structures_4:
        if time.time() > deadline - 60:
            break
        w_opt, val = _optimize_weights_for_support(s_struct, maxiter=200)
        if w_opt is not None:
            update_best(s_struct, w_opt)
    
    # === Phase 4: Joint optimization for n=3,4,5 ===
    def joint_obj(params, n):
        s = params[:n]
        w = params[n:]
        s = np.maximum(s, 0.0)
        w = np.maximum(w, 0.0)
        # Normalize s to [0,1] for stability
        if s.max() > 0:
            s = s / s.max()
        total_w = w.sum()
        if total_w <= 1e-9:
            return 0.0
        w_norm = w / total_w
        return -_compute_P(s, w_norm)
    
    for n in [3, 4, 5]:
        if time.time() > deadline - 30:
            break
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            s_init = np.sort(rng.uniform(0.1, 1.0, n))
            s_init[0] = 0.0
            w_init = rng.dirichlet(np.ones(n))
            x0 = np.concatenate([s_init, w_init])
            
            res = minimize(
                lambda x: joint_obj(x, n), x0, method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            s_opt = np.maximum(res.x[:n], 0.0)
            w_opt = np.maximum(res.x[n:], 0.0)
            if s_opt.max() > 0:
                s_opt = s_opt / s_opt.max()
            if w_opt.sum() > 1e-9:
                w_opt = w_opt / w_opt.sum()
                update_best(s_opt, w_opt)
    
    # === Phase 5: Try 5-point and 6-point with structured supports ===
    for n in [5, 6]:
        if time.time() > deadline - 20:
            break
        s_struct = np.linspace(0.0, 1.0, n)
        w_opt, val = _optimize_weights_for_support(s_struct, maxiter=300)
        if w_opt is not None:
            update_best(s_struct, w_opt)
        
        # Geometric spacing
        if n > 2:
            s_geo = np.geomspace(0.01, 1.0, n)
            s_geo[0] = 0.0
            w_opt, val = _optimize_weights_for_support(s_geo, maxiter=300)
            if w_opt is not None:
                update_best(s_geo, w_opt)
    
    # Final cleanup
    if best_support is not None and best_weights is not None:
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        tw = best_weights.sum()
        if tw > 1e-9:
            best_weights = best_weights / tw
        return best_support.astype(np.float64), best_weights.astype(np.float64)
    
    # Fallback
    return np.array([0.0, 1.0], dtype=np.float64), np.array([0.5, 0.5], dtype=np.float64)


# EVOLVE-BLOCK-END
