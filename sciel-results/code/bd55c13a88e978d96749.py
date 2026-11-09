import numpy as np
import time
from scipy.optimize import linprog


def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape: expect (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
    
    n = P.shape[0]
    C = w.shape[0] if w.ndim == 2 else 5
    cap = 25.0
    
    # Time budget
    t0 = time.time()
    deadline = t0 + 310.0
    
    eps = 1e-9
    
    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return np.clip(h, 1e-9, 1e6)
    
    # Precompute weight statistics
    wmax = w.max(axis=0)
    wsum = w.sum(axis=0)
    wmean = w.mean(axis=0)
    
    # LP relaxation to get dual prices (shadow prices)
    # Maximize P^T x subject to w x <= cap, 0 <= x <= 1
    # linprog minimizes, so we minimize -P^T x
    dual_prices = np.ones(C)  # fallback
    try:
        c_obj = -P
        A_ub = w  # shape (C, n)
        b_ub = np.full(C, cap)
        bounds = [(0, 1)] * n
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            # Dual variables (shadow prices) for the inequality constraints
            # In HiGHS, dual variables are in res.ineqlin.marginals (for older) 
            # or res.ineqlin.marginals
            # Actually for linprog with method='highs', duals are in res.ineqlin.marginals
            # Let's check: the dual of min c'x s.t. Ax <= b is max b'y s.t. A'y >= c, y >= 0
            # The marginals give the dual prices
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                if hasattr(res.ineqlin, 'marginals'):
                    dual_prices = np.abs(res.ineqlin.marginals)
                    # Ensure non-negative
                    dual_prices = np.maximum(dual_prices, 0)
    except Exception:
        pass
    
    # Reduced cost heuristic: P[i] - sum_c dual[c] * w[c,i]
    # Items with high reduced cost are "efficient" relative to the LP optimum
    reduced_costs = P - w.T @ dual_prices
    
    # Base candidates
    candidates = [
        sanitize((P / (wmax + eps)) ** 2),
        sanitize(P / (wmax + eps)),
        sanitize((P / (wsum + eps)) ** 2),
        sanitize(P / (wsum + eps)),
        sanitize(P / (wmean + eps)),
        sanitize((P / (wmax + eps)) * (1.0 / (1.0 + wsum / C))),
        sanitize(P ** 2 / (wmax + eps)),
        sanitize(P / (wmax + eps) * (1.0 - wsum / (C * 2.0 + eps))),
        # LP-based candidates
        sanitize(np.maximum(reduced_costs, eps)),
        sanitize(reduced_costs ** 2),
        sanitize(P / (wmax + eps) * np.maximum(reduced_costs, eps)),
        sanitize((P / (wsum + eps)) * np.maximum(reduced_costs, eps)),
        # Blend LP with ratio
        sanitize(0.5 * (P / (wmax + eps)) + 0.5 * np.maximum(reduced_costs, eps)),
        sanitize(0.3 * (P / (wmax + eps)) + 0.7 * np.maximum(reduced_costs, eps)),
        sanitize(0.7 * (P / (wmax + eps)) + 0.3 * np.maximum(reduced_costs, eps)),
    ]
    
    def fast_colony_sim(h, seed, n_iter=30, n_ants=20):
        """Vectorized colony simulation for a given heuristic."""
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n)
        best_total = 0.0
        
        for it in range(n_iter):
            ant_totals = np.zeros(n_ants)
            
            for a in range(n_ants):
                remaining = np.full(C, cap)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                while True:
                    # Find items that fit
                    fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                    if not fits_mask.any():
                        break
                    
                    fit_indices = np.where(fits_mask)[0]
                    probs = pheromone[fit_indices] * h[fit_indices]
                    probs = probs / probs.sum()
                    
                    idx = rng.choice(fit_indices, p=probs)
                    used[idx] = True
                    remaining -= w[:, idx]
                    total += P[idx]
                
                ant_totals[a] = total
            
            iter_best = ant_totals.max()
            best_total = max(best_total, iter_best)
            pheromone = pheromone * 0.9 + iter_best / 10.0
        
        return best_total
    
    # Evaluate candidates
    best_h = candidates[0]
    best_score = -1.0
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
        score = fast_colony_sim(h, 42)
        if score > best_score:
            best_score = score
            best_h = h
    
    # Try to refine: mix top candidates or adjust exponents
    if time.time() < deadline - 60.0:
        # Try power variations of the best base
        base_ratio = P / (wmax + eps)
        for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize(base_ratio ** alpha)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try mixing with sum-based
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize((P / (wmax + eps)) ** alpha * (1.0 / (1.0 + wsum / C)))
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try LP-based refinements
        if time.time() < deadline - 60.0:
            rc = np.maximum(reduced_costs, eps)
            for alpha in [0.5, 1.0, 1.5, 2.0]:
                if time.time() > deadline - 30.0:
                    break
                h = sanitize(rc ** alpha)
                score = fast_colony_sim(h, 42)
                if score > best_score:
                    best_score = score
                    best_h = h
            
            # Blend LP reduced cost with ratio-based
            for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
                if time.time() > deadline - 30.0:
                    break
                h = sanitize(beta * (P / (wmax + eps)) + (1 - beta) * rc)
                score = fast_colony_sim(h, 42)
                if score > best_score:
                    best_score = score
                    best_h = h
    
    return best_h
