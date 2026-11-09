"""Optimized heuristics for ACO CVRP using a SEARCH-based approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses a search approach: generate multiple candidate heuristic matrices
    based on different weighting strategies and return the one that
    minimizes expected route cost (estimated via a fast greedy simulation).
    """
    deadline = time.time() + 290.0  # wall-clock budget with safety margin
    n = distance_matrix.shape[0]
    
    # Validate inputs
    if distance_matrix.shape[0] != n or distance_matrix.shape[1] != n:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Precompute useful arrays
    demand_ratio = demands / max(demands.max(), 1e-9)
    demand_vec = demands.copy()
    demand_vec[0] = 0.0
    
    # Candidate strategies for constructing the heuristic matrix
    candidates = []
    
    # Strategy 1: Pure inverse distance (baseline)
    candidates.append(base_heuristic.copy())
    
    # Strategy 2: Inverse distance with demand-aware scaling
    demand_scale = np.ones((n, n))
    for j in range(1, n):
        demand_scale[:, j] = 1.0 / (1.0 + 0.1 * demand_ratio[j])
    candidates.append(base_heuristic * demand_scale)
    
    # Strategy 3: Inverse distance squared (stronger preference for short edges)
    candidates.append(1.0 / (distance_matrix ** 2 + 1e-9))
    
    # Strategy 4: Inverse distance to the power of 0.5 (softer preference)
    candidates.append(1.0 / (np.sqrt(distance_matrix) + 1e-9))
    
    # Strategy 5: Combined - inverse distance with slight demand consideration
    candidates.append(1.0 / (distance_matrix ** 0.8 + 1e-9) * demand_scale ** 0.5)
    
    # Strategy 6: Nearest-neighbor bias
    nn_bias = np.ones((n, n))
    for i in range(n):
        if i == 0:
            continue
        dists = distance_matrix[i, 1:].copy()
        nearest_3 = np.argpartition(dists, 3)[:3]
        for j_idx in nearest_3:
            nn_bias[i, j_idx + 1] = 2.0
    candidates.append(base_heuristic * nn_bias)
    
    # Strategy 7: Depot-aware - boost edges from/to depot
    depot_boost = np.ones((n, n))
    depot_boost[0, :] = 1.2
    depot_boost[:, 0] = 1.2
    candidates.append(base_heuristic * depot_boost)
    
    # Strategy 8: Balanced - moderate exponent with small demand scaling
    candidates.append(1.0 / (distance_matrix ** 0.9 + 1e-9) * (1.0 / (1.0 + 0.05 * demand_ratio)))
    
    # Strategy 9: Stronger nearest-neighbor bias (top-5)
    nn_bias5 = np.ones((n, n))
    for i in range(n):
        if i == 0:
            continue
        dists = distance_matrix[i, 1:].copy()
        nearest_5 = np.argpartition(dists, 5)[:5]
        for j_idx in nearest_5:
            nn_bias5[i, j_idx + 1] = 1.5
    candidates.append(base_heuristic * nn_bias5)
    
    # Strategy 10: Distance-based with coordinate-aware detour penalty
    # Penalize edges that create large detours relative to direct depot distance
    coord = coordinates
    dist_to_depot = np.sqrt((coord[:, 0] - coord[0, 0])**2 + (coord[:, 1] - coord[0, 1])**2)
    detour_factor = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and i != 0 and j != 0:
                # Detour penalty: if going i->j increases total path vs going i->depot->j
                direct = dist_to_depot[i] + dist_to_depot[j]
                via = distance_matrix[i, j]
                if via > direct:
                    detour_factor[i, j] = 1.0 / (1.0 + 0.3 * (via - direct))
    candidates.append(base_heuristic * detour_factor)
    
    # Strategy 11: Exponent 1.5 with demand scaling
    candidates.append(1.0 / (distance_matrix ** 1.5 + 1e-9) * (1.0 / (1.0 + 0.15 * demand_ratio)))
    
    # Strategy 12: Exponent 0.7 with stronger depot boost
    depot_boost2 = np.ones((n, n))
    depot_boost2[0, :] = 1.5
    depot_boost2[:, 0] = 1.5
    candidates.append(1.0 / (distance_matrix ** 0.7 + 1e-9) * depot_boost2)
    
    # Strategy 13: Angular heuristic (S-shape)
    # Prefer edges that do not change direction abruptly
    # Calculate angle between vector (depot->i) and (i->j)
    # cos(theta) = ( (depot->i) . (i->j) ) / (|depot->i| * |i->j|)
    # We want high cos(theta) -> small angle -> straight line
    # Heuristic = 1 / dist * (1 + cos(theta))
    
    # Vector from depot to i
    vec_depot_i = coord - coord[0]
    # Vector from i to j (needs matrix of vectors)
    # vec_ij = coord[j] - coord[i]
    
    # Efficiently compute dot products and norms
    # Dot product matrix: (coord[j] - coord[i]) . vec_depot_i
    # This is tricky to vectorize fully without huge memory, but n=50 is small.
    
    # Let's compute a matrix of angles or cosines
    # For each i, we have a vector v_i = coord[i] - coord[0]
    # For each j, we have a vector w_ij = coord[j] - coord[i]
    
    # We can compute cos_matrix[i, j] = dot(v_i, w_ij) / (norm(v_i) * norm(w_ij))
    
    # v_i is (n, 2)
    # w_ij is (n, n, 2) -> too big? 50*50*2 = 5000 floats, fine.
    
    # Let's just loop over i for clarity and speed is still okay for n=50
    cos_matrix = np.ones((n, n))
    for i in range(1, n):
        v_i = coord[i] - coord[0]
        norm_v_i = np.linalg.norm(v_i)
        if norm_v_i < 1e-6: continue
        for j in range(1, n):
            if i == j: continue
            w_ij = coord[j] - coord[i]
            norm_w_ij = np.linalg.norm(w_ij)
            if norm_w_ij < 1e-6: continue
            dot = np.dot(v_i, w_ij)
            cos_val = dot / (norm_v_i * norm_w_ij)
            # Clamp to [-1, 1]
            cos_val = np.clip(cos_val, -1.0, 1.0)
            cos_matrix[i, j] = cos_val
            
    # Heuristic: boost if cos is high (straight), penalize if cos is low (turn)
    # Factor = 1 + alpha * cos
    alpha = 0.5
    angular_factor = 1.0 + alpha * cos_matrix
    # Ensure positive
    angular_factor = np.maximum(angular_factor, 0.1)
    candidates.append(base_heuristic * angular_factor)
    
    # Strategy 14: Inverse distance with angular penalty
    candidates.append(1.0 / (distance_matrix ** 1.2 + 1e-9) * angular_factor)
    
    def simulate_cost(heuristic_mat, n_sims=100):
        """
        Fast simulation to estimate route cost for a given heuristic.
        Simulates random ant-like routes using the heuristic as guidance.
        Returns the mean total distance.
        Optimized with vectorized feasibility checks.
        """
        rng = np.random.RandomState(42)
        total_costs = np.zeros(n_sims)
        
        # Precompute demand feasibility masks for each possible load?
        # No, load changes. But we can precompute demand array.
        
        for sim in range(n_sims):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            load = 0.0
            total_dist = 0.0
            remaining = n - 1
            
            while remaining > 0:
                # Find feasible next customers
                # Vectorized check:
                # 1. Not visited
                # 2. load + demands[j] <= capacity
                
                feasible_mask = ~visited
                feasible_mask[0] = False # Depot is not a customer to visit
                feasible_mask = feasible_mask & (load + demands <= capacity)
                
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    # Return to depot
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    load = 0.0
                    continue
                
                # Get heuristic values for feasible nodes
                probs = heuristic_mat[current, feasible_indices]
                probs = np.maximum(probs, 1e-9)
                probs /= probs.sum()
                
                # Select next node
                next_idx = rng.choice(len(feasible_indices), p=probs)
                next_j = feasible_indices[next_idx]
                
                total_dist += distance_matrix[current, next_j]
                visited[next_j] = True
                load += demands[next_j]
                current = next_j
                remaining -= 1
            
            total_costs[sim] = total_dist
        
        return np.mean(total_costs)
    
    # Evaluate each candidate and pick the best
    best_candidate = None
    best_cost = float('inf')
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        try:
            # Adjust number of simulations based on time remaining
            # If we have lots of time, do more sims for accuracy
            # If we have little time, do fewer
            time_left = deadline - 30 - time.time()
            if time_left > 60:
                n_sims = 200
            elif time_left > 30:
                n_sims = 100
            else:
                n_sims = 50
                
            cost = simulate_cost(cand, n_sims=n_sims)
            if cost < best_cost:
                best_cost = cost
                best_candidate = cand
        except Exception:
            continue
    
    # If no valid candidate, fallback to base heuristic
    if best_candidate is None:
        best_candidate = base_heuristic
    
    # Ensure all entries are positive
    result = np.maximum(best_candidate, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
