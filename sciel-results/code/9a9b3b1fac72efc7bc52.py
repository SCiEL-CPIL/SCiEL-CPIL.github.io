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
    C = 5
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
    
    # Pareto dominance filter: identify items not strictly dominated
    dominance_count = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if P[j] >= P[i] and np.all(w[:, j] <= w[:, i]):
                if P[j] > P[i] or np.any(w[:, j] < w[:, i]):
                    dominance_count[i] += 1
    
    tau = 2.0
    pareto_weight = np.exp(-dominance_count / tau)
    
    # Greedy feasible solution: build a high-quality feasible set
    # This biases the heuristic toward known-good items
    def greedy_feasible_set(P, w, cap, C, n, iterations=50):
        """Build multiple greedy feasible solutions and return item frequency."""
        item_freq = np.zeros(n)
        rng = np.random.RandomState(42)
        
        for _ in range(iterations):
            remaining = np.full(C, cap)
            used = np.zeros(n, dtype=bool)
            total = 0.0
            
            while True:
                fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                if not fits_mask.any():
                    break
                
                # Score: prize / max weight (value density)
                fit_indices = np.where(fits_mask)[0]
                scores = P[fit_indices] / (wmax[fit_indices] + eps)
                # Add small random perturbation for diversity
                scores += 0.01 * rng.rand(len(fit_indices))
                
                best_local = np.argmax(scores)
                idx = fit_indices[best_local]
                used[idx] = True
                item_freq[idx] += 1
                remaining -= w[:, idx]
                total += P[idx]
        
        return item_freq
    
    greedy_freq = greedy_feasible_set(P, w, cap, C, n)
    
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
        sanitize((P / (wmax + eps)) ** 2 * pareto_weight),
        sanitize((P / (wmax + eps)) * pareto_weight),
        sanitize((P / (wsum + eps)) ** 2 * pareto_weight),
        sanitize((P / (wsum + eps)) * pareto_weight),
        sanitize(P / (wmean + eps) * pareto_weight),
        sanitize((P / (wmax + eps)) ** 3 * pareto_weight),
        sanitize((P / (wmax + eps)) ** 1.5 * pareto_weight),
        # Greedy-seeded candidates: blend value density with greedy frequency
        sanitize((P / (wmax + eps)) * (1.0 + greedy_freq / max(1, greedy_freq.max()))),
        sanitize((P / (wmax + eps)) ** 1.5 * (1.0 + greedy_freq / max(1, greedy_freq.max()))),
        sanitize((P / (wsum + eps)) ** 2 * (1.0 + greedy_freq / max(1, greedy_freq.max()))),
        sanitize((P / (wmax + eps)) ** 2 * (1.0 + greedy_freq / max(1, greedy_freq.max())) * pareto_weight),
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
    
    # Refinement phase: local optimization around best candidate
    if time.time() < deadline - 60.0:
        # Try power variations of the best base with Pareto weighting
        base_ratio = P / (wmax + eps)
        for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize((base_ratio ** alpha) * pareto_weight)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try different tau values for Pareto weighting
        for tau_val in [0.5, 1.0, 1.5, 3.0, 5.0]:
            if time.time() > deadline - 30.0:
                break
            pw = np.exp(-dominance_count / tau_val)
            h = sanitize((P / (wmax + eps)) ** 2 * pw)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try mixing with sum-based and Pareto
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize((P / (wmax + eps)) ** alpha * (1.0 / (1.0 + wsum / C)) * pareto_weight)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try blending with greedy frequency at different strengths
        greedy_norm = greedy_freq / max(1, greedy_freq.max())
        for gamma in [0.5, 1.0, 2.0, 3.0, 5.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize(best_h * (1.0 + gamma * greedy_norm))
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
    
    # Polish phase: fine-tune the best heuristic via multiplicative perturbation
    if time.time() < deadline - 30.0:
        # Use a smoothed surrogate: log-sum-exp based soft-max over item attractiveness
        # This encourages exploration of near-optimal items rather than just the top few
        log_h = np.log(best_h + eps)
        # Compute a soft ranking: items with higher log_h get boosted, but with smoothing
        # that accounts for constraint tightness (items that fit more easily are slightly favored)
        
        # Estimate "fit ease" for each item: how many constraints it barely violates
        # If we scale weights by 2x, items that still fit are "easy"
        w_scaled = w * 2.0
        fit_easy = np.all(w_scaled <= cap, axis=0).astype(float)
        
        # Combine: boost items that are both high-value and easy to fit
        # Use a smooth blend to avoid discontinuities
        blend_alpha = 0.3
        h_polished = best_h * (1.0 + blend_alpha * fit_easy)
        h_polished = sanitize(h_polished)
        
        score = fast_colony_sim(h_polished, 42)
        if score > best_score:
            best_score = score
            best_h = h_polished
        
        # Try inverse weighting: penalize items that use up capacity quickly
        # This is a change of variables: instead of rewarding high value/weight,
        # we reward items that leave more room for others
        remaining_after = cap - w  # capacity left after taking item
        fit_room = np.prod(remaining_after, axis=0)  # product of remaining capacities
        h_room = sanitize(best_h * (fit_room / (cap ** C) + eps))
        
        if time.time() < deadline - 30.0:
            score = fast_colony_sim(h_room, 42)
            if score > best_score:
                best_score = score
                best_h = h_room
        
        # Try a geometric mean of value density and fit room
        h_geo = sanitize(np.sqrt(best_h * (fit_room / (cap ** C) + eps)))
        if time.time() < deadline - 30.0:
            score = fast_colony_sim(h_geo, 42)
            if score > best_score:
                best_score = score
                best_h = h_geo
        
        # Final: try small multiplicative perturbations to escape local optima
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > deadline - 30.0:
                break
            noise = 1.0 + 0.1 * rng.randn(n)
            h_perturbed = sanitize(best_h * noise)
            score = fast_colony_sim(h_perturbed, 42)
            if score > best_score:
                best_score = score
                best_h = h_perturbed
    
    return best_h
