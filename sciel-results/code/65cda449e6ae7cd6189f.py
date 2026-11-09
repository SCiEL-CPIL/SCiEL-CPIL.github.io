import numpy as np
import time


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
    eps = 1e-9
    
    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return np.clip(h, 1e-9, 1e6)
    
    # Precompute weight statistics
    wmax = w.max(axis=0)
    wsum = w.sum(axis=0)
    wmean = w.mean(axis=0)
    
    # Time budget
    t0 = time.time()
    deadline = t0 + 310.0
    
    # Pareto dominance filter: identify items not strictly dominated
    # An item i is dominated by j if P[j] >= P[i] and w[:,j] <= w[:,i] (all constraints)
    # and at least one strict inequality
    def pareto_scores(P, w, cap):
        """Assign high scores to Pareto-optimal items, lower to others."""
        n = P.shape[0]
        C = w.shape[0]
        scores = np.ones(n)
        
        # For each item, check if it's dominated by any other
        # Vectorized: for item i, is there j with P[j] >= P[i] and all w[c,j] <= w[c,i]?
        # P[j] >= P[i] for all j: P >= P[i]
        # w[:,j] <= w[:,i] for all c: w <= w[:,i] (broadcast)
        
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            # Find items that could dominate i
            better_or_equal_prize = P >= P[i]
            better_or_equal_weight = np.all(w <= w[:, i], axis=0)
            # At least one strict
            strict_prize = P > P[i]
            strict_weight = np.any(w < w[:, i], axis=0)
            any_strict = strict_prize | strict_weight
            
            dominators = better_or_equal_prize & better_or_equal_weight & any_strict
            if dominators.any():
                dominated[i] = True
        
        # Pareto-optimal items get high base score
        pareto_mask = ~dominated
        base_score = np.where(pareto_mask, 10.0, 1.0)
        
        # Within Pareto-optimal, rank by prize/weight efficiency
        ratio = P / (wmax + eps)
        # Normalize ratio to [0, 1]
        r_min = ratio.min()
        r_max = ratio.max()
        if r_max > r_min:
            ratio_norm = (ratio - r_min) / (r_max - r_min)
        else:
            ratio_norm = np.ones(n)
        
        # Final score: Pareto items get 10 * (1 + ratio_norm), others get 1 * ratio_norm
        final = np.where(pareto_mask, 10.0 * (1.0 + ratio_norm), 1.0 + 0.5 * ratio_norm)
        return sanitize(final)
    
    # Build candidate heuristics
    candidates = []
    
    # 1. Pareto-based
    candidates.append(pareto_scores(P, w, cap))
    
    # 2. LP-based (simplified: use prize/weight ratio as proxy for dual prices)
    # Since we can't easily get LP duals without scipy, use a heuristic approximation
    # The "most constraining" resource: find which constraint is tightest on average
    # For each constraint c, compute how much of cap is used per unit prize
    constraint_tightness = np.zeros(C)
    for c in range(C):
        # Items that use constraint c heavily relative to their prize
        if P.max() > 0:
            constraint_tightness[c] = np.sum(w[c] * P) / (cap * P.sum() + eps)
    
    # Weighted prize/weight using constraint tightness
    # Items that use tight constraints less get higher scores
    weighted_weight = w.T @ constraint_tightness  # shape (n,)
    lp_proxy = P / (weighted_weight + eps)
    candidates.append(sanitize(lp_proxy))
    
    # 3. Simple prize/weight ratio variants
    candidates.append(sanitize(P / (wmax + eps)))
    candidates.append(sanitize(P / (wsum + eps)))
    candidates.append(sanitize(P ** 2 / (wsum + eps)))
    candidates.append(sanitize((P / (wmax + eps)) ** 2))
    
    # 4. Combined: Pareto * ratio
    pareto_h = candidates[0]
    ratio_h = candidates[2]
    candidates.append(sanitize(pareto_h * ratio_h))
    
    # 5. Prize-weighted Pareto: higher prize gets more weight in Pareto score
    prize_weighted_pareto = pareto_scores(P ** 2, w, cap)
    candidates.append(sanitize(prize_weighted_pareto))
    
    # 6. Multi-objective: consider each constraint separately
    for c in range(C):
        candidates.append(sanitize(P / (w[c] + eps)))
    
    # 7. Geometric mean of ratios across constraints
    geo_mean_ratio = np.ones(n)
    for c in range(C):
        geo_mean_ratio *= (P / (w[c] + eps))
    geo_mean_ratio = geo_mean_ratio ** (1.0 / C)
    candidates.append(sanitize(geo_mean_ratio))
    
    # 8. Harmonic mean of ratios
    harm_mean_ratio = np.zeros(n)
    for c in range(C):
        harm_mean_ratio += (w[c] + eps) / P
    harm_mean_ratio = C / (harm_mean_ratio + eps)
    candidates.append(sanitize(harm_mean_ratio))
    
    # Fast colony simulation
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
    
    # Try to refine: mix top candidates
    if time.time() < deadline - 60.0:
        # Try blending Pareto with ratio
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize(alpha * candidates[0] + (1 - alpha) * candidates[2])
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try power variations of Pareto
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize(candidates[0] ** alpha)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
    
    return best_h
