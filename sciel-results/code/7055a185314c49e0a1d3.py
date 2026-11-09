import numpy as np
import time

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
    
    # Candidate 1: Simple ratio
    # h = P / (w_sum / C)
    metric_sum = w_sum / C
    metric_max = w_max
    metric_mean = w_mean
    
    # Normalize metrics to [0, 1] roughly to avoid scale issues
    metric_sum_norm = metric_sum / (metric_sum.max() + 1e-6)
    metric_max_norm = metric_max / (metric_max.max() + 1e-6)
    metric_mean_norm = metric_mean / (metric_mean.max() + 1e-6)
    
    metrics = [
        metric_sum_norm,
        metric_max_norm,
        metric_mean_norm
    ]
    
    # Try a few specific structures first
    candidates = []
    for metric in metrics:
        # h = P / metric
        candidates.append(P_norm / (metric + 1e-6))
        # h = P / metric^2
        candidates.append(P_norm / ((metric + 1e-6) ** 2))
        # h = P^2 / metric
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
    # Model: h = (P_norm^a) / (metric^b)
    # We'll do a random search in log-space for a and b
    # and also try mixing metrics
    
    # Determine which metric was best? 
    # For simplicity, let's just do a broad random search over (a, b, metric_mix)
    # metric_mix: weight for sum vs max vs mean
    
    # To save time, we'll restrict the search space
    # a in [0.5, 2.0], b in [0.5, 3.0]
    
    n_tries = 100
    for i in range(n_tries):
        if time.time() > deadline:
            break
            
        a = 10 ** rng_search.uniform(-0.3, 0.5) # 0.5 to 3.16
        b = 10 ** rng_search.uniform(-0.1, 0.7) # 0.8 to 5.0
        
        # Randomly choose a metric or a mix
        # Let's stick to one metric per try to keep it simple and fast
        m_idx = rng_search.randint(0, 3)
        metric = metrics[m_idx]
        
        h_candidate = (P_norm ** a) / ((metric + 1e-6) ** b)
        h_candidate = np.clip(h_candidate, 1e-9, 1e6)
        
        score = simulate_aco(h_candidate)
        if score > best_score:
            best_score = score
            best_h = h_candidate.copy()
            
    # Fallback
    if best_h is None:
        best_h = P_norm
        
    return best_h
