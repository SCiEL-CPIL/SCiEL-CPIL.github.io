import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic_matrix, seed, n_iterations=30, n_ants=20):
    """
    Simulates the ACO process to evaluate a heuristic matrix.
    Returns the highest total prize collected by any ant across all iterations.
    """
    n = len(prizes)
    np.random.seed(seed)
    pheromone = np.ones((n, n))
    best_global = 0.0
    
    for _ in range(n_iterations):
        best_iter = 0.0
        
        for _ant in range(n_ants):
            current = 0
            visited = [False] * n
            visited[0] = True
            length = 0.0
            prize_sum = prizes[0]
            
            while True:
                candidates = []
                probs = []
                for j in range(1, n):
                    if visited[j]:
                        continue
                    if length + dist[current, j] + dist[j, 0] <= max_len:
                        candidates.append(j)
                        h = heuristic_matrix[current, j]
                        if h <= 0 or not np.isfinite(h):
                            h = 1e-9
                        p = pheromone[current, j] * h
                        probs.append(p)
                
                if not candidates:
                    break
                
                total_p = sum(probs)
                if total_p == 0:
                    break
                
                probs = np.array(probs)
                probs = probs / probs.sum()
                
                r = np.random.rand()
                cumsum = np.cumsum(probs)
                chosen_idx = np.searchsorted(cumsum, r)
                if chosen_idx >= len(candidates):
                    chosen_idx = len(candidates) - 1
                
                next_node = candidates[chosen_idx]
                
                length += dist[current, next_node]
                visited[next_node] = True
                prize_sum += prizes[next_node]
                current = next_node
        
        best_iter = max(best_iter, prize_sum)
        
        pheromone = pheromone * 0.9
        pheromone += best_iter / 10.0
        
        if best_iter > best_global:
            best_global = best_iter
            
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    def gen_prize_over_dist(p, d, ml):
        """H[i,j] = prize[j] / dist[i,j]"""
        return (p[None, :] + 1e-9) / (d + 1e-9)
    
    def gen_prize_over_total_dist(p, d, ml):
        """H[i,j] = prize[j] / (dist[i,j] + dist[j,0])"""
        d_0j = d[0, :]
        return (p[None, :] + 1e-9) / (d + d_0j[None, :] + 1e-9)
    
    def gen_prize_exp_decay(p, d, ml):
        """H[i,j] = prize[j] * exp(-dist[i,j])"""
        return (p[None, :] + 1e-9) * np.exp(-(d + 1e-9))
    
    def gen_prize_over_dist_sq(p, d, ml):
        """H[i,j] = prize[j] / dist[i,j]^2"""
        return (p[None, :] + 1e-9) / (d + 1e-9)**2
    
    def gen_prize_over_combined(p, d, ml):
        """H[i,j] = prize[j] / (dist[i,j] + dist[j,0] + dist[i,0])"""
        d_0 = d[0, :]
        return (p[None, :] + 1e-9) / (d + d_0[None, :] + d_0[:, None] + 1e-9)
    
    def gen_prize_ranked(p, d, ml):
        """H[i,j] = prize[j] / (dist[i,j] + dist[j,0]), with prize boosted by rank"""
        prize_rank = np.argsort(np.argsort(p)) / n
        boosted = p + 0.5 * prize_rank
        d_0j = d[0, :]
        return (boosted[None, :] + 1e-9) / (d + d_0j[None, :] + 1e-9)
    
    def gen_prize_sqrt(p, d, ml):
        """H[i,j] = sqrt(prize[j]) / (dist[i,j] + dist[j,0])"""
        d_0j = d[0, :]
        return (np.sqrt(p) + 1e-9) / (d + d_0j[None, :] + 1e-9)
    
    def gen_prize_log(p, d, ml):
        """H[i,j] = log(1+prize[j]) / (dist[i,j] + dist[j,0])"""
        d_0j = d[0, :]
        return (np.log1p(p) + 1e-9) / (d + d_0j[None, :] + 1e-9)
    
    def gen_prize_over_dist_minus(p, d, ml):
        """H[i,j] = prize[j] / max(dist[i,j] - dist[j,0] + dist[i,0], eps)"""
        # This encourages nodes that are "on the way" back to origin
        d_0 = d[0, :]
        denom = np.maximum(d + d_0[:, None] - d_0[None, :], 1e-9)
        return (p[None, :] + 1e-9) / denom
    
    candidates = [
        gen_prize_over_dist,
        gen_prize_over_total_dist,
        gen_prize_exp_decay,
        gen_prize_over_dist_sq,
        gen_prize_over_combined,
        gen_prize_ranked,
        gen_prize_sqrt,
        gen_prize_log,
        gen_prize_over_dist_minus,
    ]
    
    best_score = -np.inf
    best_matrix = None
    
    # Use a few seeds to estimate robustness
    test_seeds = [42, 123, 999]
    
    for gen in candidates:
        if time.time() > deadline:
            break
            
        try:
            h_mat = gen(prizes, distance_matrix, max_len)
            # Sanitize: ensure positive finite values
            h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
            
            scores = []
            for seed in test_seeds:
                if time.time() > deadline:
                    break
                score = _run_aco(prizes, distance_matrix, max_len, h_mat, seed)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_matrix = h_mat.copy()
        except Exception:
            continue
    
    if best_matrix is None:
        # Fallback to a simple heuristic
        best_matrix = gen_prize_over_total_dist(prizes, distance_matrix, max_len)
        best_matrix = np.where(np.isfinite(best_matrix) & (best_matrix > 0), best_matrix, 1e-9)
        
    return best_matrix
