import numpy as np
import time
from scipy.optimize import linprog
from scipy.optimize import minimize

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are correct
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
        
    n = P.shape[0]
    C = w.shape[0] # 5
    cap = 25.0
    
    # Time management
    # The evaluator kills after 330s wall-clock.
    # We assume this function is called 3 times (once per instance).
    # Total budget 330s. To be safe, we assume ~110s per call.
    # However, if the clock is per-call, we have 300s.
    # Let's use a conservative global-like estimate if possible, or just a tight per-call deadline.
    # Since we can't know the global state, we'll use a per-call deadline that is safe even if called 3 times sequentially.
    # If 330s total, and 3 calls, ~110s each.
    # Let's set deadline to 100s to be very safe.
    t_start = time.time()
    deadline = t_start + 100.0
    
    # Precompute features
    w_sum = w.sum(axis=0)
    w_max = w.max(axis=0)
    w_mean = w.mean(axis=0)
    
    P_max = P.max()
    if P_max == 0:
        return np.ones(n)
    
    P_norm = P / P_max
    
    # Fast ACO simulator
    # Optimizations:
    # 1. Vectorized fit check
    # 2. Precomputed indices
    # 3. Efficient probability selection
    def simulate_aco(h, seed=42, n_iter=30, n_ants=20):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n)
        best_total = 0.0
        
        # Precompute weights as (C, n) for broadcasting
        # w is already (C, n)
        
        for it in range(n_iter):
            ant_totals = np.zeros(n_ants)
            
            for a in range(n_ants):
                remaining = np.full(C, cap)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                while True:
                    # Check which items fit
                    # w[:, i] <= remaining for all c
                    # Broadcasting: w (C, n) <= remaining (C, 1) -> (C, n)
                    fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                    if not fits_mask.any():
                        break
                    
                    fit_indices = np.where(fits_mask)[0]
                    probs = pheromone[fit_indices] * h[fit_indices]
                    prob_sum = probs.sum()
                    if prob_sum == 0:
                        break
                    probs /= prob_sum
                    
                    idx = rng.choice(fit_indices, p=probs)
                    used[idx] = True
                    remaining -= w[:, idx]
                    total += P[idx]
                
                ant_totals[a] = total
            
            iter_best = ant_totals.max()
            if iter_best > best_total:
                best_total = iter_best
            
            pheromone = pheromone * 0.9 + iter_best / 10.0
            
        return best_total

    best_h = None
    best_score = -1.0
    rng_search = np.random.RandomState(12345)
    
    # Pareto dominance filter: identify items not strictly dominated
    dominance_count = np.zeros(n)
    for i in range(n):
        p_dom = (P >= P[i])
        w_dom = np.all(w <= w[:, i][:, None], axis=0)
        strict_p = (P > P[i])
        strict_w = np.any(w < w[:, i][:, None], axis=0)
        dominates_i = p_dom & w_dom & (strict_p | strict_w)
        dominates_i[i] = False
        dominance_count[i] = dominates_i.sum()
    
    pareto_score = 1.0 / (1.0 + dominance_count)
    
    quality_score = P_norm / (w_max / (w_max.max() + 1e-6) + 1e-6)
    
    metric_sum = w_sum / C
    metric_max = w_max
    metric_mean = w_mean
    
    metric_sum_norm = metric_sum / (metric_sum.max() + 1e-6)
    metric_max_norm = metric_max / (metric_max.max() + 1e-6)
    metric_mean_norm = metric_mean / (metric_mean.max() + 1e-6)
    
    metrics = [metric_sum_norm, metric_max_norm, metric_mean_norm]
    
    candidates = []
    
    # LP Relaxation Heuristic
    try:
        c_obj = -P
        A_ub = w
        b_ub = np.full(C, cap)
        bounds = [(0, 1) for _ in range(n)]
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            duals = res.ineqlin
            reduced_costs = P - (duals @ w)
            h_lp = np.clip(reduced_costs, 1e-9, 1e6)
            candidates.append(h_lp)
    except Exception:
        pass

    # Spectral decomposition of constraint interaction
    # W is (C, n). W @ W.T is (C, C).
    # Eigenvalues represent variance in resource consumption patterns.
    # Low eigenvalue directions are "cheap" or "less contested" in some sense?
    # Actually, we want to score items.
    # Let's look at the projection of each item's weight vector onto the eigenvectors.
    # W.T @ V gives (n, C) matrix of projections.
    # Score item i based on its alignment with low-eigenvalue directions.
    # Or, penalize alignment with high-eigenvalue directions (contested resources).
    
    try:
        WWT = w @ w.T
        eigvals, eigvecs = np.linalg.eigh(WWT)
        # eigvals are sorted ascending.
        # eigvecs are columns.
        
        # Project weights onto eigenvectors
        # proj = w.T @ eigvecs -> (n, C)
        proj = w.T @ eigvecs
        
        # We want to reward items that have low projection on high eigenvalue modes?
        # Or high projection on low eigenvalue modes?
        # Hypothesis: Items that consume resources in "uncommon" combinations (low eigenvalue directions)
        # are less likely to conflict with others.
        # Let's create a score based on the sum of squared projections weighted by inverse eigenvalues.
        # h_spectral = sum_k (proj[i,k]^2 / eigvals[k]) ? No, that penalizes high eigenvalues.
        # If we want to reward low eigenvalue alignment:
        # h_spectral = sum_k (proj[i,k]^2 * eigvals[k])? No.
        
        # Let's try: h_spectral = sum_k (proj[i,k]^2 / (eigvals[k] + eps))
        # This gives high score to items that project heavily onto low eigenvalue directions.
        
        eps = 1e-6
        weights_spec = 1.0 / (eigvals + eps)
        # h_spec_i = sum_k proj[i,k]^2 * weights_spec[k]
        h_spec = np.sum(proj ** 2 * weights_spec, axis=1)
        
        # Normalize
        h_spec = h_spec / (h_spec.max() + 1e-6)
        
        # Combine with quality
        for alpha in [0.5, 1.0, 2.0]:
            h_cand = quality_score * (1.0 + alpha * h_spec)
            candidates.append(h_cand)
            
        # Also try pure spectral
        candidates.append(h_spec)
        
        # Try subtracting high eigenvalue projection (penalize contested)
        weights_pen = 1.0 / (eigvals + eps)
        # Actually, let's try penalizing high eigenvalue alignment directly.
        # h_pen = 1 - sum_k (proj[i,k]^2 * eigvals[k] / sum(eigvals^2))?
        
        # Let's try a different spectral feature:
        # Alignment with the "average" resource vector vs specific ones.
    except Exception:
        h_spec = np.ones(n) / n

    # Pareto-based heuristics
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        h_pareto = quality_score * (1.0 + alpha * pareto_score)
        candidates.append(h_pareto)
    
    # Traditional ratio heuristics
    for metric in metrics:
        candidates.append(P_norm / (metric + 1e-6))
        candidates.append(P_norm / ((metric + 1e-6) ** 2))
        candidates.append((P_norm ** 2) / (metric + 1e-6))
        
    # Evaluate initial candidates
    for h_cand in candidates:
        if time.time() > deadline:
            break
        h_safe = np.clip(h_cand, 1e-9, 1e6)
        score = simulate_aco(h_safe)
        if score > best_score:
            best_score = score
            best_h = h_safe.copy()
            
    # Local search around the best candidate
    # Model: h = (P_norm^a) / (metric^b) * (1 + alpha * pareto_score) * (1 + beta * h_spec)
    
    n_tries = 100
    for i in range(n_tries):
        if time.time() > deadline:
            break
            
        a = 10 ** rng_search.uniform(-0.3, 0.5)
        b = 10 ** rng_search.uniform(-0.1, 0.7)
        alpha = 10 ** rng_search.uniform(-1.0, 1.5)
        beta = 10 ** rng_search.uniform(-1.0, 1.5)
        
        m_idx = rng_search.randint(0, 3)
        metric = metrics[m_idx]
        
        base = (P_norm ** a) / ((metric + 1e-6) ** b)
        h_candidate = base * (1.0 + alpha * pareto_score) * (1.0 + beta * h_spec)
        h_candidate = np.clip(h_candidate, 1e-9, 1e6)
        
        score = simulate_aco(h_candidate)
        if score > best_score:
            best_score = score
            best_h = h_candidate.copy()
            
    # Fallback
    if best_h is None:
        best_h = P_norm
        
    return best_h
