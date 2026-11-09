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
    t_start = time.time()
    # Total budget 330s. 3 calls. ~110s per call.
    # Use 100s per call to be safe.
    deadline = t_start + 100.0
    
    # Precompute features
    w_sum = w.sum(axis=0)
    w_max = w.max(axis=0)
    w_mean = w.mean(axis=0)
    
    P_max = P.max()
    if P_max == 0:
        return np.ones(n)
    
    P_norm = P / P_max
    
    # Fast ACO simulator (Exact)
    def simulate_aco(h, seed=42, n_iter=30, n_ants=20):
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

    # Fast Greedy Surrogate Simulator
    # Approximates the behavior of ACO by greedily picking items with highest heuristic
    # that still fit. This captures the "attractiveness" ordering which drives ACO.
    def fast_greedy_score(h):
        h = np.asarray(h, dtype=float)
        # Sort indices by heuristic descending
        order = np.argsort(-h)
        
        remaining = np.full(C, cap)
        total = 0.0
        
        for idx in order:
            # Check if item fits
            if np.all(w[:, idx] <= remaining):
                remaining -= w[:, idx]
                total += P[idx]
            # Optimization: if remaining is very small, break?
            # Simple check: if no item can fit, break. 
            # But checking "no item can fit" is expensive. 
            # Given N=100, the loop is short.
        
        return total

    best_h = None
    best_exact_score = -1.0
    
    # Phase 1: Broad search with fast surrogate
    # Family: h = P^alpha / W^beta
    # W in {w_sum, w_max, w_mean}
    # alpha in [0.5, 3.0], beta in [0.5, 3.0]
    
    metrics = [w_sum, w_max, w_mean]
    metric_names = ['sum', 'max', 'mean']
    
    candidates = []
    
    # Grid search
    alphas = np.linspace(0.5, 3.0, 15)
    betas = np.linspace(0.5, 3.0, 15)
    
    for W in metrics:
        W_safe = W + 1e-6
        for a in alphas:
            for b in betas:
                h = (P_norm ** a) / (W_safe ** b)
                candidates.append(h)
                
    # Add some random candidates to explore further
    rng_search = np.random.RandomState(42)
    for _ in range(100):
        a = rng_search.uniform(0.1, 5.0)
        b = rng_search.uniform(0.1, 5.0)
        m_idx = rng_search.randint(0, 3)
        W = metrics[m_idx]
        W_safe = W + 1e-6
        h = (P_norm ** a) / (W_safe ** b)
        candidates.append(h)
        
    # Evaluate candidates with fast surrogate
    scores = []
    for h in candidates:
        if time.time() > deadline - 10: # Leave 10s for exact sim
            break
        s = fast_greedy_score(h)
        scores.append((s, h))
        
    # Sort by surrogate score
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Refine top candidates with exact ACO
    top_k = 20
    for s, h in scores[:top_k]:
        if time.time() > deadline - 5: # Leave 5s safety
            break
        
        h_safe = np.clip(h, 1e-9, 1e6)
        exact_score = simulate_aco(h_safe)
        
        if exact_score > best_exact_score:
            best_exact_score = exact_score
            best_h = h_safe.copy()
            
    # Fallback if no exact score was computed or all failed
    if best_h is None:
        # Use the best surrogate result
        if len(scores) > 0:
            best_h = scores[0][1]
        else:
            best_h = P_norm / (w_sum + 1e-6)
            
    return best_h
