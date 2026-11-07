import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment

def _two_opt_improve(tour, dist, max_iters=500):
    """
    Standard 2-opt local search improvement.
    """
    n = len(tour)
    if n < 3:
        return tour.copy()
    improved = True
    best_tour = tour.copy()
    
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u1 = tour[i]
                u2 = tour[(i + 1) % n]
                u3 = tour[j]
                u4 = tour[(j + 1) % n]
                
                # Check if the edges (u1, u2) and (u3, u4) exist in the current tour
                # We are replacing them with (u1, u3) and (u2, u4)
                # This is valid if the segment [i+1, j] is reversed
                delta = (dist[u1, u3] + dist[u2, u4]) - (dist[u1, u2] + dist[u3, u4])
                
                if delta < -1e-9:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    best_tour = tour.copy()
    
    return best_tour

def _nn_tour(dist, start=0):
    """Nearest neighbor tour construction."""
    n = dist.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    
    visited = np.zeros(n, dtype=bool)
    current = start
    tour = [start]
    visited[start] = True
    
    for _ in range(n - 1):
        d = dist[current].copy()
        d[visited] = np.inf
        next_node = np.argmin(d)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return np.array(tour)

def _construct_backbone_tour(dist):
    """
    Construct a high-quality tour using multi-restart NN + 2-Opt.
    """
    n = dist.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    
    best_tour = None
    best_len = float('inf')
    
    # Try multiple starting points for NN
    # For N=50, trying all 50 starts is cheap and often yields better results
    start_nodes = list(range(n))
    
    for start in start_nodes:
        tour = _nn_tour(dist, start)
        tour = _two_opt_improve(tour, dist, max_iters=200)
        tour_len = sum(dist[tour[i], tour[(i+1) % n]] for i in range(n))
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour.copy()
    
    # Final polish with more 2-opt iterations
    if best_tour is not None:
        best_tour = _two_opt_improve(best_tour, dist, max_iters=500)
    
    return best_tour if best_tour is not None else _nn_tour(dist, 0)

def _compute_mwpm_edges(dist):
    """
    Compute Maximum Weight Perfect Matching edges.
    We want to maximize weight, so we minimize -weight.
    Weight is typically 1/distance or similar. Here we use -dist to minimize total dist of matching.
    Actually, for TSP heuristics, we want edges that are SHORT to be attractive.
    So we find the Minimum Weight Perfect Matching on the distance matrix.
    """
    n = dist.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=bool)
    
    # linear_sum_assignment minimizes the sum of the selected entries.
    # So we pass the distance matrix directly.
    # It returns row_ind, col_ind such that (row_ind[i], col_ind[i]) are the matched pairs.
    row_ind, col_ind = linear_sum_assignment(dist)
    
    mst_mask = np.zeros((n, n), dtype=bool)
    for u, v in zip(row_ind, col_ind):
        if u != v:
            mst_mask[u, v] = True
            mst_mask[v, u] = True
            
    return mst_mask

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast ACO simulation to evaluate heuristic quality.
    Matches the contract: 20 ants, 30 iterations.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    rng = np.random.default_rng(seed)
    
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.integers(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= rho
        
        current_best = min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    n = distance_matrix.shape[0]
    dm = distance_matrix.copy()
    
    # 1. Construct a high-quality "backbone" tour with multi-restart
    backbone_tour = _construct_backbone_tour(dm)
    
    # 2. Compute Minimum Weight Perfect Matching (MWPM) edges
    # These edges are strong candidates for inclusion in the optimal tour.
    mwpm_mask = _compute_mwpm_edges(dm)
    
    # 3. Pre-compute the mask of backbone edges (symmetric)
    backbone_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        u = backbone_tour[i]
        v = backbone_tour[(i + 1) % n]
        backbone_mask[u, v] = True
        backbone_mask[v, u] = True
    
    # 4. Build continuous inverse-distance floor
    safe_dist = dm.copy()
    np.fill_diagonal(safe_dist, np.inf)
    safe_dist[safe_dist < 1e-9] = 1e-9
    inv_dist = 1.0 / safe_dist
    np.fill_diagonal(inv_dist, 0.0)
    max_inv = inv_dist.max()
    if max_inv > 0:
        inv_dist_norm = inv_dist / max_inv
    else:
        inv_dist_norm = inv_dist
    
    # 5. Test combined heuristics: 
    # H = inv_dist_norm + alpha * backbone_mask + beta * mwpm_mask
    # MWPM edges are structurally more critical (1-factor) than tour edges (2-factor).
    # We expect beta to be significant.
    
    alpha_candidates = [1.0, 5.0, 10.0, 20.0, 50.0]
    beta_candidates = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    best_score = float('inf')
    best_heuristic = None
    
    # We will iterate through a subset of combinations to stay within time limits
    # Total combos: 5 * 5 = 25. Each ACO sim is ~2-5s. 25 * 5s = 125s. Safe.
    
    for alpha in alpha_candidates:
        if time.time() > deadline - 60.0:
            break
        for beta in beta_candidates:
            if time.time() > deadline - 30.0:
                break
                
            h = inv_dist_norm.copy()
            h += alpha * backbone_mask.astype(float)
            h += beta * mwpm_mask.astype(float)
            
            np.fill_diagonal(h, 0.0)
            
            # Validate
            if not np.all(np.isfinite(h)) or h.shape != dm.shape:
                continue
            if np.any(h < 0):
                h = np.maximum(h, 1e-9)
            
            score = _fast_aco_simulate(dm, h, n_ants=20, n_iterations=30, seed=42)
            
            if score < best_score:
                best_score = score
                best_heuristic = h.copy()
                
    # Fallback: If something went wrong, use inverse distance
    if best_heuristic is None:
        best_heuristic = inv_dist_norm.copy()
    
    # Final sanity checks
    best_heuristic[best_heuristic <= 0] = 1e-9
    best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
    
    return best_heuristic
