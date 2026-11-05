import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Enhanced Survival Probability + Optimized Local Search
    1. Initialize scores using structural features (degree, cluster density, second-neighbor density).
    2. Use vectorized greedy simulation for speed.
    3. Local search with smart swap candidates focused on decision boundary.
    4. Multiple restarts with adaptive noise.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Enhanced Survival Probability Model
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    cluster_density = neighbor_deg_sum / safe_deg
    
    # Second-neighbor density
    second_deg = adj @ (adj @ deg)
    safe_second = second_deg + 1.0
    second_density = second_deg / safe_second
    
    # Combine features
    scores = -deg - 0.3 * cluster_density - 0.1 * second_density
    
    # Normalize
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    def simulate_fast(s):
        """Vectorized greedy simulation."""
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    # Vectorized deactivation
                    active[nbrs] = False
        return count
    
    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # 3. Local Search with time guard
    max_time = 118.0  # Safety margin before 120s limit
    rng = np.random.default_rng(42)
    
    # Precompute score ordering
    score_order = np.argsort(-scores)
    
    # Try multiple restarts
    for restart in range(10):
        if time.time() - start_time > max_time - 3:
            break
        
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Adaptive noise: start larger, decrease
            noise_scale = max(0.02, 0.08 - 0.005 * restart)
            noise = rng.normal(0, noise_scale, n)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        no_improve = 0
        max_no_improve = 2000 + 300 * restart
        
        recent_swaps = set()
        
        while time.time() - start_time < max_time - 1 and no_improve < max_no_improve:
            # Smart swap: focus on vertices near the selection boundary
            # Vertices with similar scores are more likely to swap selection status
            if rng.random() < 0.7:
                # Pick a random position and swap with a nearby vertex
                base_idx = rng.integers(0, n - 1)
                offset = rng.integers(1, min(15, n - 1))
                i = score_order[base_idx]
                j = score_order[min(base_idx + offset, n - 1)]
            else:
                # Random swap
                i, j = rng.integers(0, n, size=2)
            
            if i == j:
                continue
            
            swap_key = (min(i, j), max(i, j))
            if swap_key in recent_swaps:
                continue
            recent_swaps.add(swap_key)
            if len(recent_swaps) > 1000:
                recent_swaps.clear()
            
            # Swap scores
            si, sj = current_scores[i], current_scores[j]
            current_scores[i], current_scores[j] = sj, si
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                current_scores[i], current_scores[j] = si, sj
                no_improve += 1
        
        # Early exit if no progress
        if current_val <= best_val and restart > 3:
            break
    
    return best_scores
