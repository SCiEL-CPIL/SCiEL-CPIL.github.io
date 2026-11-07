# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to enforce wall-clock limit across multiple calls
_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # Safety margin before 330s kill

def _safe_update(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Strategy:
    1. Infer round number from edge_n_used max value.
    2. Apply adaptive penalty that increases with rounds.
    3. Extra penalty on edges in the current local-opt tour.
    4. Ensure the resulting matrix is finite and positive.
    """
    # Check time limit
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Infer round: max edge usage indicates how many rounds have passed
    max_used = edge_n_used.max()
    if max_used > 0:
        round_num = int(max_used)  # each round adds 1 to used edges
    else:
        round_num = 0
    
    # Normalize edge usage to [0, 1]
    if max_used > 0:
        norm_used = edge_n_used.astype(np.float64) / max_used
    else:
        norm_used = np.zeros_like(edge_n_used, dtype=np.float64)
    
    # Build a mask for edges in the current tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(len(local_opt_tour) - 1):
        u = local_opt_tour[i]
        v = local_opt_tour[i + 1]
        tour_mask[u, v] = 1.0
        tour_mask[v, u] = 1.0
    # Close the tour
    if len(local_opt_tour) > 1:
        u = local_opt_tour[-1]
        v = local_opt_tour[0]
        tour_mask[u, v] = 1.0
        tour_mask[v, u] = 1.0
    
    # Adaptive alpha: increase penalty strength with rounds to escape deeper local optima
    # Start moderate, ramp up
    base_alpha = 0.2 + 0.04 * round_num
    base_alpha = min(base_alpha, 1.2)  # Cap to avoid destroying structure
    
    # Extra penalty for edges in current tour
    tour_alpha = 0.15 + 0.02 * round_num
    tour_alpha = min(tour_alpha, 0.6)
    
    # Power law on normalized usage to emphasize heavy hitters
    gamma = 1.5
    p_norm = np.power(norm_used, gamma)
    
    # Combined penalty multiplier
    penalty_multiplier = 1.0 + base_alpha * p_norm
    
    # Add extra penalty for tour edges
    penalty_multiplier += tour_alpha * tour_mask
    
    # Clip to reasonable range
    penalty_multiplier = np.clip(penalty_multiplier, 1.0, 5.0)
    
    # Apply small random perturbation to break near-ties (seeded by round for reproducibility)
    rng = np.random.default_rng(seed=round_num * 1000 + 42)
    perturbation = rng.uniform(0.0, 0.01, size=(n, n))
    new_distances = edge_distance * penalty_multiplier + perturbation
    new_distances = (new_distances + new_distances.T) / 2.0
    np.fill_diagonal(new_distances, 0.0)
    
    # Ensure finiteness
    if not np.all(np.isfinite(new_distances)):
        new_distances = np.nan_to_num(new_distances, nan=1.0, posinf=1.0, neginf=1.0)
        
    return new_distances
    
    new_distances = edge_distance * penalty_multiplier
    
    # Ensure symmetry
    new_distances = (new_distances + new_distances.T) / 2.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(new_distances, 0.0)
    
    # Ensure finiteness
    if not np.all(np.isfinite(new_distances)):
        new_distances = np.nan_to_num(new_distances, nan=1.0, posinf=1.0, neginf=1.0)
        
    return new_distances

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    try:
        return _safe_update(edge_distance, local_opt_tour, edge_n_used)
    except Exception:
        return edge_distance.copy()
# EVOLVE-BLOCK-END
