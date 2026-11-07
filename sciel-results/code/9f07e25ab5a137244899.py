"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Reshape if 1D, though contract says (5, 100)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # prizes: (5, 100), weights: (5, 100)
    n_constraints, n_items = weights.shape
    
    # Structural Change: Per-constraint normalized efficiency aggregation
    # Instead of a single bottleneck metric, compute per-constraint efficiency
    # and aggregate them to capture multi-dimensional trade-offs.
    
    # 1. Normalize weights per constraint to [0, 1]
    # This makes the cost comparable across constraints
    w_min = np.min(weights, axis=1, keepdims=True)
    w_max = np.max(weights, axis=1, keepdims=True)
    w_range = w_max - w_min
    w_range = np.where(w_range > 0, w_range, 1.0)
    normalized_weights = (weights - w_min) / w_range  # shape (5, 100)
    
    # 2. Normalize prizes per constraint to [0, 1]
    p_min = np.min(prizes, axis=1, keepdims=True)
    p_max = np.max(prizes, axis=1, keepdims=True)
    p_range = p_max - p_min
    p_range = np.where(p_range > 0, p_range, 1.0)
    normalized_prizes = (prizes - p_min) / p_range  # shape (5, 100)
    
    # 3. Compute per-constraint efficiency: normalized prize / (normalized weight + eps)
    eps = 1e-9
    per_constraint_eff = normalized_prizes / (normalized_weights + eps)  # shape (5, 100)
    
    # 4. Aggregate per-constraint efficiencies
    # Use geometric mean to require good efficiency across ALL constraints
    # This is more robust than max/min because it balances all dimensions
    # log-geometric-mean = mean of logs
    # Clip to avoid log of negative/zero
    per_constraint_eff_clipped = np.clip(per_constraint_eff, eps, None)
    log_eff = np.log(per_constraint_eff_clipped)  # shape (5, 100)
    log_geo_mean = np.mean(log_eff, axis=0)  # shape (100,)
    geo_mean_eff = np.exp(log_geo_mean)  # shape (100,)
    
    # 5. Also compute a complementary metric: sum of normalized prizes / sum of normalized weights
    # This captures items that are good on average across constraints
    sum_prizes_norm = np.sum(normalized_prizes, axis=0)  # shape (100,)
    sum_weights_norm = np.sum(normalized_weights, axis=0) + eps  # shape (100,)
    avg_eff = sum_prizes_norm / sum_weights_norm  # shape (100,)
    
    # 6. Combine geometric mean and average efficiency
    # Geometric mean is more sensitive to the worst constraint
    # Average efficiency is more sensitive to overall value
    # Blend them: geometric mean * sqrt(avg_eff) to balance
    combined_score = geo_mean_eff * np.sqrt(avg_eff + eps)
    
    # 7. Handle non-finite values
    combined_score = np.where(np.isfinite(combined_score), combined_score, 0.0)
    
    # 8. Rank-based transformation to smooth the heuristic landscape
    # This prevents extreme outliers from dominating ant selection
    ranks = np.arange(1, n_items + 1)
    order = np.argsort(-combined_score)
    
    # Assign inverse ranks (best item gets highest heuristic)
    heuristic_scores = np.zeros_like(combined_score)
    heuristic_scores[order] = 1.0 / ranks
    
    # 9. Power transformation to adjust spread
    # Slight flattening to let pheromone guide more
    power = 0.85
    heuristic_scores = heuristic_scores ** power
    
    # 10. Safety checks
    heuristic_scores = np.where(np.isfinite(heuristic_scores) & (heuristic_scores > 0), heuristic_scores, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        pass  # Return whatever we have
        
    return heuristic_scores
# EVOLVE-BLOCK-END
