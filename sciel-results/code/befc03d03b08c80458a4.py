import numpy as np
import time

def _run_aco_vectorized(prizes, dist, max_len, heuristic, seed):
    """
    Run ACO with vectorized feasibility and probability checks.
    Returns the best total prize found.
    """
    n = len(prizes)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix
    pheromone = np.ones((n, n), dtype=float)
    # Heuristic matrix, ensuring positive values
    heur = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Pre-compute distances to origin for quick feasibility checks
    dist_to_origin = dist[:, 0]
    
    best_overall = 0.0
    
    for it in range(30):
        best_iter = 0.0
        
        for ant in range(20):
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                feasible_mask = (~visited_mask) & (length + dist[current, :] + dist_to_origin <= max_len)
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    break
                
                probs = pheromone[current, feasible_indices] * heur[current, feasible_indices]
                s = probs.sum()
                if s <= 0:
                    break
                probs = probs / s
                
                j = int(rng.choice(feasible_indices, p=probs))
                
                length += dist[current, j]
                total_prize += prizes[j]
                visited_mask[j] = True
                current = j
            
            if total_prize > best_iter:
                best_iter = total_prize
            if total_prize > best_overall:
                best_overall = total_prize
        
        pheromone *= 0.9
        pheromone += best_iter / 10.0
    
    return best_overall

def _compute_centrality(prizes, dist, sigma=0.15):
    """
    Compute Gaussian-weighted centrality: for each node i, sum over j of
    prize[j] * exp(-dist[i,j]^2 / (2*sigma^2)). This captures local cluster
    density of prizes around each node.
    """
    n = len(prizes)
    # dist is n x n, prizes is n
    # centrality[i] = sum_j prizes[j] * exp(-dist[i,j]^2 / (2*sigma^2))
    sq_dists = dist ** 2
    weights = np.exp(-sq_dists / (2.0 * sigma ** 2))
    centrality = weights @ prizes
    return centrality

def _gaussian_smooth_heuristic(H, dist, sigma=0.2):
    """
    Apply Gaussian smoothing to the heuristic matrix to reduce sharp gradients.
    H_smooth[i,j] = sum_k H[i,k] * exp(-dist[j,k]^2 / (2*sigma^2)) / sum_k exp(...)
    This blends each row's heuristic values based on spatial proximity of target nodes.
    """
    n = H.shape[0]
    sq_dists = dist ** 2
    weights = np.exp(-sq_dists / (2.0 * sigma ** 2))
    # For each row i, smooth across the destination dimension j
    # H_smooth[i,j] = sum_k H[i,k] * W[j,k] / sum_k W[j,k]
    # W[j,k] = exp(-dist[j,k]^2 / (2*sigma^2))
    # weights is n x n where weights[j,k] = exp(-dist[j,k]^2/(2*sigma^2))
    row_sums = weights.sum(axis=1, keepdims=True)  # n x 1
    weights_norm = weights / row_sums  # n x n, each row sums to 1
    # H_smooth[i,j] = sum_k H[i,k] * weights_norm[j,k]
    # = H @ weights_norm.T
    H_smooth = H @ weights_norm.T
    return H_smooth

def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix with optional centrality boost and Gaussian smoothing.
    """
    eps = 1e-9
    p = prizes[None, :].astype(float)
    d = dist.astype(float)
    ret = dist[:, 0][None, :].astype(float)
    
    a = params.get('d_exp', 1.0)
    b = params.get('ret_weight', 0.0)
    c = params.get('p_exp', 1.0)
    k = params.get('p_add', 0.0)
    cent_exp = params.get('cent_exp', 0.0)
    smooth_sigma = params.get('smooth_sigma', 0.0)
    
    if params.get('exp_decay', None) is not None:
        beta = params['exp_decay']
        H = (p + k) ** c * np.exp(-beta * (d + b * ret))
    else:
        denom = (d + b * ret + eps) ** a
        numer = (p + k) ** c
        H = numer / denom
    
    # Apply centrality boost: multiply by centrality[j]^cent_exp
    if cent_exp > 0:
        centrality = _compute_centrality(prizes, dist, sigma=params.get('cent_sigma', 0.15))
        # centrality is n-vector, broadcast to n x n along rows
        cent_col = centrality[None, :] ** cent_exp
        H = H * cent_col
    
    # Apply Gaussian smoothing
    if smooth_sigma > 0:
        H = _gaussian_smooth_heuristic(H, dist, sigma=smooth_sigma)
    
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _seed_heuristic(prizes, dist):
    """
    Fallback heuristic: Prize / Dist
    """
    H = (prizes[None, :] + 1e-9) / (dist + 1e-9)
    return np.where(np.isfinite(H) & (H > 0), H, 1e-9)

def heuristics(prizes, distance_matrix, max_len):
    try:
        start_time = time.time()
        deadline = start_time + 300.0
        
        prizes = np.asarray(prizes, dtype=float)
        dist = np.asarray(distance_matrix, dtype=float)
        n = len(prizes)
        
        # Structurally diverse candidates: base forms + centrality + smoothing
        candidates = [
            # Base power-law
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.5, 'ret_weight': 0.5, 'p_exp': 1.5, 'p_add': 0.0},
            {'d_exp': 2.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            # With centrality
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0, 'cent_exp': 1.0, 'cent_sigma': 0.15},
            {'d_exp': 1.5, 'ret_weight': 0.5, 'p_exp': 1.5, 'p_add': 0.0, 'cent_exp': 1.5, 'cent_sigma': 0.2},
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 2.0, 'p_add': 0.0, 'cent_exp': 1.0, 'cent_sigma': 0.1},
            # With Gaussian smoothing
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0, 'smooth_sigma': 0.2},
            {'d_exp': 1.5, 'ret_weight': 0.5, 'p_exp': 1.5, 'p_add': 0.0, 'smooth_sigma': 0.3},
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0, 'smooth_sigma': 0.15},
            # Combined: centrality + smoothing
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0, 'cent_exp': 1.0, 'cent_sigma': 0.15, 'smooth_sigma': 0.2},
            {'d_exp': 1.5, 'ret_weight': 0.5, 'p_exp': 1.5, 'p_add': 0.0, 'cent_exp': 1.0, 'cent_sigma': 0.2, 'smooth_sigma': 0.25},
            # Exponential decay forms
            {'exp_decay': 2.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'exp_decay': 2.5, 'ret_weight': 0.5, 'p_exp': 1.0, 'p_add': 0.0, 'cent_exp': 1.0, 'cent_sigma': 0.15},
        ]
        
        num_seeds = 3
        results = []
        
        for params in candidates:
            if time.time() > deadline - 60.0:
                break
                
            H = _make_heuristic(prizes, dist, max_len, params)
            scores = []
            
            for s in range(num_seeds):
                if time.time() > deadline - 60.0:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                results.append((mean_score, params, H, scores))
        
        if not results:
            return _seed_heuristic(prizes, dist)
        
        results.sort(key=lambda x: -x[0])
        best_mean, best_params, best_H, best_scores = results[0]
        
        # Refine with more seeds if time permits
        if time.time() < deadline - 30.0:
            current_seed_count = len(best_scores)
            target_seed_count = current_seed_count + 5
            
            for s in range(current_seed_count, target_seed_count):
                if time.time() > deadline - 30.0:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, best_H, s)
                best_scores.append(score)
        
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
        return best_H
        
    except Exception:
        try:
            prizes = np.asarray(prizes, dtype=float)
            dist = np.asarray(distance_matrix, dtype=float)
            return _seed_heuristic(prizes, dist)
        except Exception:
            try:
                n = len(prizes)
            except Exception:
                n = 50
            return np.ones((n, n))
