import numpy as np
import time
from scipy.optimize import linprog

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
    C = w.shape[0]  # 5
    cap = 25.0
    
    # Time management: 330s total wall-clock, 3 calls expected
    t_start = time.time()
    deadline = t_start + 95.0  # Conservative per-call deadline
    
    # Precompute features
    w_sum = w.sum(axis=0)
    w_max = w.max(axis=0)
    w_mean = w.mean(axis=0)
    w_min = w.min(axis=0)
    
    P_max = P.max()
    if P_max == 0:
        return np.ones(n)
    
    P_norm = P / P_max
    
    # Compute LP dual prices (shadow prices) for the continuous relaxation
    # This gives us the marginal value of each constraint
    # Maximize sum(P_i * x_i) subject to sum(w_ki * x_i) <= cap for k=0..4, 0<=x_i<=1
    # Dual: Minimize sum(cap * y_k) subject to sum(y_k * w_ki) >= P_i for i=0..n-1, y_k >= 0
    
    c_dual = np.full(C, cap)
    A_ub_dual = -w.T  # shape (n, C)
    b_ub_dual = -P     # shape (n,)
    bounds_dual = [(0, None)] * C
    
    rc_feature = np.ones(n) * 0.5
    reduced_costs = None
    
    try:
        if time.time() < deadline - 80.0:
            lp_result = linprog(c_dual, A_ub=A_ub_dual, b_ub=b_ub_dual, 
                               bounds=bounds_dual, method='highs')
            if lp_result.success:
                dual_prices = lp_result.x  # shape (C,)
                reduced_costs = P - w.T @ dual_prices
                rc_max = np.max(np.abs(reduced_costs))
                if rc_max > 1e-9:
                    # Use normalized reduced costs directly (clip to positive)
                    rc_norm = reduced_costs / rc_max
                    rc_feature = np.clip(rc_norm, -1, 1)
                else:
                    rc_feature = np.ones(n) * 0.5
    except Exception:
        pass
    
    # Fast low-fidelity ACO simulator for candidate evaluation
    def simulate_aco_fast(h, seed=42, n_iter=12, n_ants=8):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
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
    
    # Medium-fidelity ACO simulator for semi-final candidates
    def simulate_aco_med(h, seed=42, n_iter=20, n_ants=12):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
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
    
    # Full-fidelity ACO simulator for final verification
    def simulate_aco_full(h, seed=42, n_iter=30, n_ants=20):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
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
    
    # Candidate 1: Simple ratio (baseline)
    h_base = P_norm / (w_sum / C + 1e-6)
    score = simulate_aco_fast(h_base)
    if score > best_score:
        best_score = score
        best_h = h_base.copy()
    
    # Candidate 2: LP reduced cost based (direct, not sigmoid)
    if reduced_costs is not None:
        rc_pos = np.clip(reduced_costs / (np.max(np.abs(reduced_costs)) + 1e-9), 0, 1)
        h_lp = rc_pos * P_norm + 1e-6
        score = simulate_aco_fast(h_lp)
        if score > best_score:
            best_score = score
            best_h = h_lp.copy()
    
    # Candidate 3: Blend of ratio and LP
    if reduced_costs is not None:
        h_blend = 0.5 * h_base + 0.5 * h_lp
        h_blend = np.clip(h_blend, 1e-9, 1e6)
        score = simulate_aco_fast(h_blend)
        if score > best_score:
            best_score = score
            best_h = h_blend.copy()
    
    # Expanded search: diverse metrics
    metrics = [
        w_sum / C,
        w_max,
        w_mean,
        w_min,
        (w_sum + w_max) / (2 * C),
        w_sum / (w_max + 1e-6),
    ]
    
    if reduced_costs is not None:
        rc_abs = np.abs(reduced_costs)
        metrics.append(rc_abs / (rc_abs.max() + 1e-9))
        metrics.append(1.0 / (rc_abs + 0.1))
    
    for m_idx, metric in enumerate(metrics):
        if time.time() > deadline - 20.0:
            break
            
        metric_safe = metric + 1e-6
        metric_norm = metric_safe / (metric_safe.max() + 1e-6)
        
        # Grid search with more diverse alpha/beta combinations
        for alpha in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            if time.time() > deadline - 20.0:
                break
                
            for beta in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
                if time.time() > deadline - 20.0:
                    break
                    
                h_candidate = (P_norm / metric_norm ** beta) ** alpha
                h_candidate = np.clip(h_candidate, 1e-9, 1e6)
                
                score = simulate_aco_fast(h_candidate)
                if score > best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        
        # Random search for this metric
        for _ in range(15):
            if time.time() > deadline - 20.0:
                break
                
            beta = 10 ** rng_search.uniform(-0.5, 1.8)
            alpha = 10 ** rng_search.uniform(-0.5, 1.5)
            
            h_candidate = (P_norm / metric_norm ** beta) ** alpha
            h_candidate = np.clip(h_candidate, 1e-9, 1e6)
            
            score = simulate_aco_fast(h_candidate)
            if score > best_score:
                best_score = score
                best_h = h_candidate.copy()
    
    # Local refinement around best candidate found
    if best_h is not None and time.time() < deadline - 30.0:
        # Try perturbations in log-space
        best_h_log = np.log(best_h + 1e-9)
        for _ in range(20):
            if time.time() > deadline - 30.0:
                break
            
            perturbation = rng_search.normal(0, 0.1, size=n)
            h_candidate = np.exp(best_h_log + perturbation)
            h_candidate = np.clip(h_candidate, 1e-9, 1e6)
            
            score = simulate_aco_fast(h_candidate)
            if score > best_score:
                best_score = score
                best_h = h_candidate.copy()
        
        # Try blending with LP feature
        if reduced_costs is not None:
            for blend in [0.2, 0.4, 0.6, 0.8]:
                if time.time() > deadline - 30.0:
                    break
                rc_pos = np.clip(reduced_costs / (np.max(np.abs(reduced_costs)) + 1e-9), 0, 1)
                h_candidate = blend * best_h + (1 - blend) * (rc_pos * P_norm + 1e-6)
                h_candidate = np.clip(h_candidate, 1e-9, 1e6)
                
                score = simulate_aco_fast(h_candidate)
                if score > best_score:
                    best_score = score
                    best_h = h_candidate.copy()
    
    # Medium-fidelity verification of top candidates
    if best_h is not None and time.time() < deadline - 15.0:
        med_score = simulate_aco_med(best_h)
        # Also verify the base heuristic at medium fidelity
        base_med_score = simulate_aco_med(h_base)
        
        if med_score < base_med_score:
            best_h = h_base.copy()
    
    # Final high-fidelity verification
    if best_h is not None and time.time() < deadline - 5.0:
        full_score = simulate_aco_full(best_h)
        base_full_score = simulate_aco_full(h_base)
        
        if full_score < base_full_score:
            best_h = h_base.copy()
    
    # Fallback if time ran out or no improvement
    if best_h is None:
        best_h = P_norm
    
    # Ensure valid output
    best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)
    
    return best_h
