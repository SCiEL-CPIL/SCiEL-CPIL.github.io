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
    wprod = np.prod(w, axis=0)
    
    # LP relaxation to get dual prices (shadow prices)
    dual_prices = np.ones(C)  # fallback
    try:
        c_obj = -P
        A_ub = w
        b_ub = np.full(C, cap)
        bounds = [(0, 1)] * n
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                if hasattr(res.ineqlin, 'marginals'):
                    dual_prices = np.abs(res.ineqlin.marginals)
                    dual_prices = np.maximum(dual_prices, 0)
    except Exception:
        pass
    
    # Reduced cost heuristic
    reduced_costs = P - w.T @ dual_prices
    
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
    
    # Generate diverse candidates
    candidates = []
    
    # Ratio-based candidates with various exponents
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        candidates.append(sanitize((P / (wmax + eps)) ** alpha))
    
    # Sum-based
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(sanitize((P / (wsum + eps)) ** alpha))
    
    # Product-based
    for alpha in [0.5, 1.0, 1.5]:
        candidates.append(sanitize((P / (wprod + eps)) ** alpha))
    
    # LP reduced cost based
    rc = np.maximum(reduced_costs, eps)
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(sanitize(rc ** alpha))
    
    # Blends of ratio and LP
    base_ratio = P / (wmax + eps)
    for beta in [0.2, 0.4, 0.6, 0.8]:
        candidates.append(sanitize(beta * base_ratio + (1 - beta) * rc))
    
    # Weighted by constraint tightness
    tightness = 1.0 / (1.0 + wsum / C)
    candidates.append(sanitize(base_ratio * tightness))
    candidates.append(sanitize(rc * tightness))
    
    # Prize squared over weight
    candidates.append(sanitize(P ** 2 / (wmax + eps)))
    candidates.append(sanitize(P ** 2 / (wsum + eps)))
    
    # Normalize candidates for fair comparison
    for i in range(len(candidates)):
        c = candidates[i]
        if c.max() > 0:
            candidates[i] = c / c.max()
    
    # Evaluate candidates
    best_h = candidates[0]
    best_score = -1.0
    scored_candidates = []
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 60.0:
            break
        score = fast_colony_sim(h, 42)
        scored_candidates.append((score, h))
        if score > best_score:
            best_score = score
            best_h = h
    
    # Sort by score and try to refine top candidates
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Try mixing top candidates
    if len(scored_candidates) >= 2 and time.time() < deadline - 45.0:
        top1_score, top1_h = scored_candidates[0]
        top2_score, top2_h = scored_candidates[1]
        
        for mix_w in [0.3, 0.5, 0.7]:
            if time.time() > deadline - 30.0:
                break
            mixed = sanitize(mix_w * top1_h + (1 - mix_w) * top2_h)
            score = fast_colony_sim(mixed, 42)
            if score > best_score:
                best_score = score
                best_h = mixed
    
    # Local polish: try small perturbations of the best heuristic
    if time.time() < deadline - 30.0:
        base = best_h.copy()
        base_mean = base.mean()
        
        # Try scaling
        for scale in [0.8, 0.9, 1.1, 1.2]:
            if time.time() > deadline - 15.0:
                break
            h = sanitize(base * scale)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try adding small random perturbations
        for trial in range(5):
            if time.time() > deadline - 15.0:
                break
            rng_p = np.random.RandomState(100 + trial)
            perturbation = rng_p.randn(n) * 0.1 * base_mean
            h = sanitize(base + perturbation)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
    
    # Final: ensure output is valid
    best_h = sanitize(best_h)
    
    return best_h
