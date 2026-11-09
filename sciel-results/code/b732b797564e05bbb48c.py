import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation. Since pheromone updates are uniform, tau stays constant,
    so selection is purely heuristic-driven. We exploit this for speed.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    eta = np.asarray(heuristic, dtype=np.float64)
    
    # Precompute: for each item i, which items j can follow i (d[i]+d[j] <= capacity)
    # And the eta values for valid transitions
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    
    # For each row i, store valid column indices and their eta values
    valid_cols = [np.where(valid[i])[0] for i in range(n)]
    eta_rows = [eta[i][valid[i]] for i in range(n)]
    
    rng = np.random.default_rng(seed)
    best_overall = np.inf
    
    for it in range(n_iters):
        ant_bins_list = []
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_fill = 0.0
            current_item = -1
            
            # Place first item randomly
            j = int(rng.integers(0, n))
            used[j] = True
            current_fill = d[j]
            current_item = j
            placed = 1
            
            while placed < n:
                remaining = capacity - current_fill
                
                # Find candidates: unused, fits in remaining space, and valid transition from current_item
                vc = valid_cols[current_item]
                if len(vc) > 0:
                    mask = (~used[vc]) & (d[vc] <= remaining)
                    valid_j = vc[mask]
                else:
                    valid_j = np.array([], dtype=int)
                
                if len(valid_j) == 0:
                    # No item fits; open new bin
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = int(unused_indices[int(rng.integers(0, len(unused_indices)))])
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    placed += 1
                    continue
                
                # Choose next item probabilistically based on eta
                eta_vals = eta[current_item, valid_j]
                total = float(np.sum(eta_vals))
                if total <= 1e-15:
                    j = int(valid_j[int(rng.integers(0, len(valid_j)))])
                else:
                    cum = np.cumsum(eta_vals)
                    cum = cum / cum[-1]
                    r = float(rng.random())
                    idx = int(np.searchsorted(cum, r))
                    if idx >= len(valid_j):
                        idx = len(valid_j) - 1
                    j = int(valid_j[idx])
                
                used[j] = True
                current_fill += d[j]
                current_item = j
                placed += 1
            
            ant_bins_list.append(bins)
        
        min_bins_this_iter = min(ant_bins_list) if ant_bins_list else np.inf
        if min_bins_this_iter < best_overall:
            best_overall = min_bins_this_iter
    
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Precompute pair sums and validity
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    start_time = time.time()
    time_limit = 280.0  # Leave margin for 3 instances + overhead
    
    # The core heuristic family: complementarity * size-weighting
    # eta_ij = (1/(1+slack_ij))^alpha * d_j^gamma
    # where alpha controls slack preference (higher = prefer tighter fits)
    # and gamma controls size preference (higher = prefer larger items)
    
    # First, evaluate a baseline
    base_slack = np.where(valid, 1.0 / (1.0 + slack), 1e-9)
    base_slack = np.where(np.isfinite(base_slack) & (base_slack > 0), base_slack, 1e-9)
    
    # Grid search over alpha (slack exponent) and gamma (size exponent)
    # alpha controls how strongly we prefer tight fits
    # gamma controls how strongly we prefer larger items
    
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    
    # Precompute d^gamma for each gamma
    d_powers = {}
    for gamma in gammas:
        if gamma == 0.0:
            d_powers[gamma] = np.ones(n)
        else:
            d_powers[gamma] = np.power(d, gamma)
    
    # Precompute slack^(-alpha) for each alpha
    slack_powers = {}
    for alpha in alphas:
        if alpha == 0.0:
            slack_powers[alpha] = np.where(valid, 1.0, 1e-9)
        else:
            slack_powers[alpha] = np.where(valid, np.power(1.0 / (1.0 + slack), alpha), 1e-9)
            slack_powers[alpha] = np.where(np.isfinite(slack_powers[alpha]) & (slack_powers[alpha] > 0), slack_powers[alpha], 1e-9)
    
    # Evaluate all combinations
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        for gamma in gammas:
            if time.time() - start_time > time_limit:
                break
            
            candidate = slack_powers[alpha] * d_powers[gamma][None, :]
            candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
            
            score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
            if score < best_score:
                best_score = score
                best_heuristic = candidate.copy()
    
    # If time allows, refine around the best combination
    if time.time() - start_time < time_limit and best_heuristic is not None:
        # Find which alpha/gamma was best
        # Re-evaluate to find the exact best
        best_alpha = alphas[0]
        best_gamma = gammas[0]
        best_score_local = np.inf
        
        for alpha in alphas:
            if time.time() - start_time > time_limit:
                break
            for gamma in gammas:
                if time.time() - start_time > time_limit:
                    break
                candidate = slack_powers[alpha] * d_powers[gamma][None, :]
                candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
                score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
                if score < best_score_local:
                    best_score_local = score
                    best_alpha = alpha
                    best_gamma = gamma
        
        # Local search around best_alpha and best_gamma
        alpha_deltas = [0.25, -0.25, 0.5, -0.5, 0.75, -0.75]
        gamma_deltas = [0.25, -0.25, 0.5, -0.5, 0.75, -0.75]
        
        for da in alpha_deltas:
            if time.time() - start_time > time_limit:
                break
            a = best_alpha + da
            if a <= 0:
                continue
            slack_powers[a] = np.where(valid, np.power(1.0 / (1.0 + slack), a), 1e-9)
            slack_powers[a] = np.where(np.isfinite(slack_powers[a]) & (slack_powers[a] > 0), slack_powers[a], 1e-9)
            
            for dg in gamma_deltas:
                if time.time() - start_time > time_limit:
                    break
                g = best_gamma + dg
                if g < 0:
                    continue
                d_powers[g] = np.power(d, g)
                
                candidate = slack_powers[a] * d_powers[g][None, :]
                candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
                
                score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
                if score < best_score:
                    best_score = score
                    best_heuristic = candidate.copy()
    
    # Also try blending with a pure slack-based heuristic (alpha=1, gamma=0)
    if time.time() - start_time < time_limit and best_heuristic is not None:
        slack_heur = np.where(valid, 1.0 / (1.0 + slack), 1e-9)
        slack_heur = np.where(np.isfinite(slack_heur) & (slack_heur > 0), slack_heur, 1e-9)
        
        for blend in [0.25, 0.5, 0.75]:
            if time.time() - start_time > time_limit:
                break
            candidate = (1.0 - blend) * best_heuristic + blend * slack_heur
            candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
            score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
            if score < best_score:
                best_score = score
                best_heuristic = candidate.copy()
    
    if best_heuristic is None:
        best_heuristic = np.where(valid, 1.0, 1e-9)
    
    # Final safety: ensure all values are positive and finite
    best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
    
    return best_heuristic
