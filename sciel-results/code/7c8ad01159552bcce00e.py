"""Heuristic for ACO-CVRP: Demand-Aware Delaunay with Full-Fidelity Adaptive Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def _build_demand_aware_delaunay_mask(coordinates, demands, capacity, distance_matrix):
    """
    Build a demand-aware Delaunay mask:
    1. Start with Delaunay triangulation edges
    2. Only keep edges (i,j) where demands[i] + demands[j] <= capacity
       (ensures at least two customers can share a route segment)
    3. For depot edges, always keep if demand fits
    4. Add high-savings edges that are demand-feasible
    """
    n = len(coordinates)
    mask = np.zeros((n, n), dtype=bool)
    
    # Step 1: Delaunay edges
    try:
        tri = Delaunay(coordinates)
        for simplex in tri.simplices:
            i, j, k = simplex
            mask[i, j] = True; mask[j, i] = True
            mask[j, k] = True; mask[k, j] = True
            mask[i, k] = True; mask[k, i] = True
    except Exception:
        try:
            tree = cKDTree(coordinates)
            _, indices = tree.query(coordinates, k=6)
            for i in range(n):
                for j in indices[i]:
                    mask[i, j] = True
        except Exception:
            mask.fill(True)

    # Step 2: Demand feasibility filter
    # For customer-customer edges: demand[i] + demand[j] <= capacity
    # For depot-customer edges: demand[j] <= capacity (always true given cap=50, demand<=9)
    D = demands
    # Create demand sum matrix
    demand_sum = D[:, np.newaxis] + D[np.newaxis, :]
    
    # Customer-customer edges (i,j) both >= 1: need demand_sum[i,j] <= capacity
    # Depot edges (i=0 or j=0): always feasible
    feasible = np.zeros((n, n), dtype=bool)
    # Depot row and column are always feasible
    feasible[0, :] = True
    feasible[:, 0] = True
    # Customer-customer: feasible if demand sum fits
    feasible[1:, 1:] = demand_sum[1:, 1:] <= capacity
    
    # Apply demand filter
    mask = mask & feasible
    
    # Step 3: Add high-savings edges that are demand-feasible
    d = distance_matrix.copy()
    np.fill_diagonal(d, 1e9)
    d_i0 = distance_matrix[:, 0]
    d_j0 = distance_matrix[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    
    # Top 15% savings edges that are demand-feasible
    if savings_pos.max() > 1e-9:
        threshold = np.percentile(savings_pos, 85)
        high_savings = savings_pos > threshold
        mask = mask | (high_savings & feasible)
    
    # Ensure symmetry
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    
    # Safety: ensure each customer has at least one feasible neighbor
    for i in range(1, n):
        if not mask[i, :].any():
            # Connect to nearest feasible node
            dists = distance_matrix[i, :].copy()
            dists[0] = 0  # depot always allowed
            dists[~feasible[i, :]] = np.inf
            nearest = np.argmin(dists)
            if nearest != i:
                mask[i, nearest] = True
                mask[nearest, i] = True
    
    return mask

def _compute_demand_proximity_penalty(distance_matrix, demands, capacity):
    """
    Compute a multiplicative penalty based on demand-density proximity.
    Edges between nodes in high-demand-density regions get boosted;
    long edges between sparse regions get penalized.
    """
    n = distance_matrix.shape[0]
    penalty = np.ones((n, n))
    
    try:
        # Local demand density: sum of demands of nearby nodes
        # Use k-nearest neighbors for density estimation
        tree = cKDTree(np.ones((n, 2)))  # placeholder, we'll use distance matrix directly
        
        # Simpler: for each node, compute average distance to k nearest neighbors
        # Short avg distance = dense region
        k = min(5, n - 1)
        avg_dist = np.zeros(n)
        for i in range(n):
            dists = distance_matrix[i, :].copy()
            dists[i] = np.inf
            sorted_dists = np.sort(dists)[:k]
            avg_dist[i] = np.mean(sorted_dists)
        
        # Normalize: dense nodes have small avg_dist
        mean_ad = np.mean(avg_dist)
        if mean_ad > 1e-9:
            avg_dist_norm = avg_dist / mean_ad
        else:
            avg_dist_norm = np.ones(n)
        
        # Penalty: boost edges between dense nodes, penalize long edges in sparse areas
        # ratio = d(i,j) / (avg_dist[i] * avg_dist[j])
        R_i = np.maximum(avg_dist[:, np.newaxis], 1e-6)
        R_j = np.maximum(avg_dist[np.newaxis, :], 1e-6)
        
        ratio = distance_matrix / (R_i * R_j)
        # Soft penalty: 1/(1+ratio^1.5) gives smoother falloff
        penalty = 1.0 / (1.0 + ratio**1.5)
        
        # Scale: make penalty range [0.5, 1.5]
        penalty = 0.5 + penalty * 1.0
        np.fill_diagonal(penalty, 1.0)
        
    except Exception:
        pass
        
    return penalty

def _compute_savings_heuristic(distance_matrix):
    """
    Compute the savings heuristic for CVRP.
    savings[i,j] = d[i,0] + d[0,j] - d[i,j]
    """
    n = distance_matrix.shape[0]
    d_i0 = distance_matrix[:, 0]
    d_j0 = distance_matrix[0, :]
    
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - distance_matrix
    
    # Clip negative savings to 0
    savings_pos = np.maximum(savings, 0.0)
    
    # Normalize to [0, 1]
    max_sav = np.max(savings_pos)
    if max_sav > 1e-9:
        savings_norm = savings_pos / max_sav
    else:
        savings_norm = np.ones_like(savings_pos)
        
    # Zero out diagonal
    np.fill_diagonal(savings_norm, 0.0)
    
    return savings_norm

def _build_savings_aware_delaunay_mask(coordinates, distance_matrix, savings_heuristic, k_nearest=3):
    """
    Build a mask that includes Delaunay edges AND high-savings edges.
    (Legacy - kept for compatibility but not used in main path)
    """
    n = len(coordinates)
    mask = np.zeros((n, n), dtype=bool)
    
    try:
        tri = Delaunay(coordinates)
        for simplex in tri.simplices:
            i, j, k = simplex
            mask[i, j] = True; mask[j, i] = True
            mask[j, k] = True; mask[k, j] = True
            mask[i, k] = True; mask[k, i] = True
    except Exception:
        try:
            tree = cKDTree(coordinates)
            _, indices = tree.query(coordinates, k=k_nearest)
            for i in range(n):
                for j in indices[i]:
                    mask[i, j] = True
        except Exception:
            mask.fill(True)
            np.fill_diagonal(mask, False)
            return mask

    threshold = np.percentile(savings_heuristic, 90)
    high_savings_mask = savings_heuristic > threshold
    mask = mask | high_savings_mask
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    
    return mask

def _compute_voronoi_penalty_matrix(coordinates, distance_matrix, demands, capacity):
    """
    Legacy Voronoi penalty - replaced by _compute_demand_proximity_penalty.
    Kept for interface compatibility.
    """
    n = len(coordinates)
    penalty = np.ones((n, n))
    return penalty

def _fast_aco_sim(distance_matrix, demands, capacity, alpha, beta, gamma, mask, n_ants=15, n_iter=8, seed=42):
    """
    Run a fast ACO simulation to evaluate heuristic parameters.
    Uses vectorized probability computation for speed.
    Returns the best cost found.
    """
    n = distance_matrix.shape[0]
    dm = distance_matrix
    
    # Precompute heuristic components
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    d_inv = 1.0 / (d + 1e-9)
    
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_heur = np.ones_like(savings)
    else:
        sav_heur = savings_pos / max_sav
    
    # Combine with log-space for numerical stability
    log_h = alpha * np.log(d_inv + 1e-9) + beta * np.log(sav_heur + 1e-9) + gamma * np.log(np.maximum(d_inv * sav_heur, 1e-9) + 1e-9)
    h = np.exp(log_h)
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
    
    # Apply mask
    h[~mask] = 1e-9
    np.fill_diagonal(h, 0.0)
    
    # Precompute feasible sets for each (node, remaining_cap) state
    # Since capacity is fixed and demands are small integers, we can precompute
    demand_int = np.round(demands).astype(int)
    
    P = np.ones((n, n))
    best_cost = float('inf')
    rng = np.random.default_rng(seed)
    
    for _ in range(n_iter):
        iteration_costs = np.zeros(n_ants)
        
        for ant in range(n_ants):
            node = 0
            remaining_cap = capacity
            cost = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                # Find feasible neighbors: unvisited, demand fits
                feasible_mask = (~visited) & (demand_int <= remaining_cap)
                feasible_mask[0] = False  # Don't go to depot until returning
                
                if not feasible_mask.any():
                    cost += dm[node, 0]
                    break
                
                # Vectorized probability computation
                probs = P[node, :] * h[node, :]
                probs[~feasible_mask] = 0.0
                
                total_prob = probs.sum()
                if total_prob <= 1e-12:
                    # Uniform among feasible
                    feasible_indices = np.where(feasible_mask)[0]
                    next_node = feasible_indices[rng.integers(len(feasible_indices))]
                else:
                    probs_norm = probs / total_prob
                    next_node = rng.choice(n, p=probs_norm)
                    if not feasible_mask[next_node]:
                        # Fallback (shouldn't happen)
                        feasible_indices = np.where(feasible_mask)[0]
                        next_node = feasible_indices[rng.integers(len(feasible_indices))]
                
                cost += dm[node, next_node]
                remaining_cap -= demand_int[next_node]
                visited[next_node] = True
                node = next_node
            
            iteration_costs[ant] = cost
        
        best_iter = float(np.min(iteration_costs))
        if best_iter < best_cost:
            best_cost = best_iter
        
        # Update Pheromone
        P *= 0.9
        P += 1.0 / best_iter
    
    return best_cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    Uses demand-aware Delaunay masking and adaptive parameter selection
    via full-fidelity local ACO simulation.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # If time is running out, return a safe default
    if _remaining() < 25.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Build demand-aware Delaunay mask
    savings_norm = _compute_savings_heuristic(dm)
    try:
        mask = _build_demand_aware_delaunay_mask(coords, D, cap, dm)
    except Exception:
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
    
    # Compute demand-proximity penalty
    try:
        proximity_penalty = _compute_demand_proximity_penalty(dm, D, cap)
    except Exception:
        proximity_penalty = np.ones((n, n))
    
    # Grid search over alpha (distance), beta (savings), gamma (interaction)
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    gammas = [0.0, 0.5, 1.0]
    
    best_score = float('inf')
    best_alpha = 2.0
    best_beta = 0.5
    best_gamma = 0.0
    
    # Phase 1: Coarse grid search
    for a in alphas:
        for b in betas:
            if _remaining() < 40.0:
                break
            for g in gammas:
                if _remaining() < 40.0:
                    break
                try:
                    score = _fast_aco_sim(dm, D, cap, a, b, g, mask, n_ants=12, n_iter=6, seed=42)
                    if score < best_score:
                        best_score = score
                        best_alpha = a
                        best_beta = b
                        best_gamma = g
                except Exception:
                    continue
        if _remaining() < 40.0:
            break
    
    # Phase 2: Fine-tune around best with more ants/iterations
    if _remaining() > 60.0:
        fine_alphas = [max(0.5, best_alpha - 0.5), best_alpha, min(4.0, best_alpha + 0.5)]
        fine_betas = [max(0.0, best_beta - 0.5), best_beta, min(2.5, best_beta + 0.5)]
        fine_gammas = [max(0.0, best_gamma - 0.5), best_gamma, min(1.5, best_gamma + 0.5)]
        
        for a in fine_alphas:
            for b in fine_betas:
                if _remaining() < 50.0:
                    break
                for g in fine_gammas:
                    if _remaining() < 50.0:
                        break
                    try:
                        score = _fast_aco_sim(dm, D, cap, a, b, g, mask, n_ants=18, n_iter=10, seed=123)
                        if score < best_score:
                            best_score = score
                            best_alpha = a
                            best_beta = b
                            best_gamma = g
                    except Exception:
                        continue
            if _remaining() < 50.0:
                break
    
    # Compute the final heuristic with the best parameters
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    d_inv = 1.0 / (d + 1e-9)
    
    # Combine in log-space
    log_h = (best_alpha * np.log(d_inv + 1e-9) + 
             best_beta * np.log(savings_norm + 1e-9) + 
             best_gamma * np.log(np.maximum(d_inv * savings_norm, 1e-9) + 1e-9))
    h = np.exp(log_h)
    
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
        
    # Apply proximity penalty
    h = h * proximity_penalty
    
    # Apply demand-aware mask
    h[~mask] = 1e-9
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 1e-9)
    
    # Final safety: ensure all values are finite and positive
    h[~np.isfinite(h)] = 1e-9
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
