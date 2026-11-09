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
    
    # Soft-max of weights (smooth surrogate for hard max)
    def softmax_weights(temperature=0.1):
        max_w = np.max(w, axis=0, keepdims=True)
        scaled = (w - max_w) / temperature
        scaled = np.clip(scaled, -50, 50)
        exp_w = np.exp(scaled)
        log_sum_exp = np.log(np.sum(exp_w, axis=0))
        return temperature * log_sum_exp + np.squeeze(max_w, axis=0)
    
    # Soft-min of prize/weight ratios (smooth surrogate for min ratio)
    def softmin_ratios(temperature=0.1):
        ratios = prizes / (weights + eps)  # shape (C, n)
        min_r = np.min(ratios, axis=0, keepdims=True)
        scaled = (ratios - min_r) / temperature
        scaled = np.clip(scaled, -50, 50)
        exp_r = np.exp(-scaled)  # negative because we want min
        log_sum_exp = np.log(np.sum(exp_r, axis=0))
        return temperature * log_sum_exp + np.squeeze(min_r, axis=0)
    
    # Pareto dominance count
    dominance_count = np.zeros(n)
    for i in range(n):
        if time.time() > deadline - 60.0:
            break
        for j in range(n):
            if i == j:
                continue
            if P[j] >= P[i] and np.all(w[:, j] <= w[:, i]):
                if P[j] > P[i] or np.any(w[:, j] < w[:, i]):
                    dominance_count[i] += 1
    
    # LP relaxation to get dual prices (shadow prices)
    # Solve: max P^T x s.t. W x <= cap, 0 <= x <= 1
    # Use scipy if available, otherwise fallback to greedy approximation
    def compute_lp_duals():
        """Compute dual prices from LP relaxation using scipy.optimize.linprog."""
        try:
            from scipy.optimize import linprog
        except ImportError:
            return None
        
        # Objective: maximize P^T x => minimize -P^T x
        c = -P
        
        # Constraints: W x <= cap
        A_ub = w  # shape (C, n)
        b_ub = np.full(C, cap)
        
        # Bounds: 0 <= x <= 1
        bounds = [(0, 1)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                # Dual prices are in res.ineqlin
                duals = np.abs(res.ineqlin)
                return duals
        except Exception:
            pass
        
        return None
    
    dual_prices = compute_lp_duals()
    
    # Base candidates with various strategies
    softmax_w = softmax_weights(0.1)
    softmin_r = softmin_ratios(0.1)
    dominance_factor = 1.0 / (1.0 + 2.0 * dominance_count)
    
    candidates = [
        # Original candidates
        sanitize((P / (wmax + eps)) ** 2),
        sanitize(P / (wmax + eps)),
        sanitize((P / (wsum + eps)) ** 2),
        sanitize(P / (wsum + eps)),
        sanitize(P / (wmean + eps)),
        sanitize((P / (wmax + eps)) * (1.0 / (1.0 + wsum / C))),
        sanitize(P ** 2 / (wmax + eps)),
        sanitize(P / (wmax + eps) * (1.0 - wsum / (C * 2.0 + eps))),
        
        # Soft-max based candidates
        sanitize(P / (softmax_w + eps)),
        sanitize((P / (softmax_w + eps)) ** 2),
        sanitize(P / (softmax_w + eps) * dominance_factor),
        sanitize((P / (softmax_w + eps)) ** 2 * dominance_factor),
        
        # Soft-min ratio based candidates
        sanitize(softmin_r),
        sanitize(softmin_r ** 2),
        sanitize(softmin_r * dominance_factor),
        sanitize(softmin_r ** 2 * dominance_factor),
        
        # Combined: soft-max weight with soft-min ratio
        sanitize((P / (softmax_w + eps)) * (softmin_r / (softmin_r + eps))),
        sanitize((P / (softmax_w + eps)) ** 2 * (softmin_r / (softmin_r + eps))),
    ]
    
    # LP relaxation based candidates
    if dual_prices is not None:
        # Reduced cost: P_i - sum_c dual_c * w_{c,i}
        reduced_cost = P - w.T @ dual_prices
        # Heuristic: items with high positive reduced cost are most attractive
        rc_heuristic = np.maximum(reduced_cost, 0) + eps
        candidates.append(sanitize(rc_heuristic))
        candidates.append(sanitize(rc_heuristic ** 2))
        candidates.append(sanitize(rc_heuristic * dominance_factor))
        candidates.append(sanitize(rc_heuristic ** 2 * dominance_factor))
        
        # Also try: prize / (1 + weighted weight by duals)
        weighted_weight = w.T @ dual_prices
        dual_ratio = P / (1.0 + weighted_weight + eps)
        candidates.append(sanitize(dual_ratio))
        candidates.append(sanitize(dual_ratio ** 2))
        candidates.append(sanitize(dual_ratio * dominance_factor))
        candidates.append(sanitize(dual_ratio ** 2 * dominance_factor))
    
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
        
        # Try soft-max based power variations
        soft_ratio = P / (softmax_w + eps)
        for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize(soft_ratio ** alpha)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try soft-min ratio power variations
        for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize(softmin_r ** alpha)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try different soft-max temperatures
        for temp in [0.05, 0.1, 0.2, 0.5]:
            if time.time() > deadline - 30.0:
                break
            sw = softmax_weights(temp)
            h = sanitize(P / (sw + eps))
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try different soft-min temperatures
        for temp in [0.05, 0.1, 0.2, 0.5]:
            if time.time() > deadline - 30.0:
                break
            sr = softmin_ratios(temp)
            h = sanitize(sr)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # LP-based refinement
        if dual_prices is not None and time.time() < deadline - 30.0:
            reduced_cost = P - w.T @ dual_prices
            rc_pos = np.maximum(reduced_cost, 0) + eps
            
            # Try power variations of reduced cost
            for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
                if time.time() > deadline - 30.0:
                    break
                h = sanitize(rc_pos ** alpha)
                score = fast_colony_sim(h, 42)
                if score > best_score:
                    best_score = score
                    best_h = h
            
            # Try mixing reduced cost with dominance
            for alpha in [0.5, 1.0, 1.5, 2.0]:
                if time.time() > deadline - 30.0:
                    break
                h = sanitize(rc_pos ** alpha * dominance_factor)
                score = fast_colony_sim(h, 42)
                if score > best_score:
                    best_score = score
                    best_h = h
            
            # Try dual ratio variations
            weighted_weight = w.T @ dual_prices
            dual_ratio = P / (1.0 + weighted_weight + eps)
            for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                if time.time() > deadline - 30.0:
                    break
                h = sanitize(dual_ratio ** alpha)
                score = fast_colony_sim(h, 42)
                if score > best_score:
                    best_score = score
                    best_h = h
    
    return best_h
