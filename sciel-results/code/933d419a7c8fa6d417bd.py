"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _quick_aco_sim(dm, demands, capacity, H, n_ants=10, n_iterations=10, seed=42, time_limit=2.0):
    """
    Fast simulation of the evaluator's ACO to estimate heuristic quality.
    Matches the evaluator's update rule: tau = 0.9*tau + 1/best.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    
    # Pre-compute valid masks? No, capacity depends on path.
    # But we can pre-compute demand array for speed.
    demands = np.asarray(demands, dtype=np.float64)
    
    for _ in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
            
        iteration_best = np.inf
        
        for _ in range(n_ants):
            if time.time() - start_time > time_limit:
                break
                
            curr = 0
            rem_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            dist = 0.0
            
            while True:
                # Find valid next nodes
                # Vectorized check is faster for small n, but explicit loop might be clearer.
                # For n=50, vectorized is fine.
                valid = ~visited
                valid[0] = False
                # Capacity constraint
                valid &= (demands <= rem_cap)
                
                if not np.any(valid):
                    # Return to depot
                    dist += dm[curr, 0]
                    rem_cap = capacity
                    curr = 0
                    if np.all(visited):
                        break
                    continue
                
                # Calculate weights
                # prob ~ tau[curr, :] * H[curr, :]
                w = tau[curr, :] * H[curr, :]
                w[~valid] = 0.0
                
                total_w = np.sum(w)
                if total_w <= 0:
                    # Fallback: random valid
                    valid_idx = np.where(valid)[0]
                    nxt = rng.choice(valid_idx)
                else:
                    p = w / total_w
                    nxt = rng.choice(n, p=p)
                
                dist += dm[curr, nxt]
                visited[nxt] = True
                rem_cap -= demands[nxt]
                curr = nxt
                
                if np.all(visited):
                    dist += dm[curr, 0]
                    break
                    
        # Track best of iteration for update
        # In the simulation, we don't strictly need to update tau perfectly if we are just ranking heuristics,
        # but to be faithful to the "refinement machinery", we should mimic the dynamics.
        # However, updating tau for every ant in a short sim is expensive.
        # The evaluator updates tau once per iteration based on the BEST ant of that iteration.
        # Let's collect the best cost of this iteration.
        
    # We need to return the best cost found.
    # The above loop structure is slightly off for the "best of iteration" tracking if we break early.
    # Let's rewrite the sim to be cleaner and strictly track the min cost found.
    
    return best_cost # This will be inf if not updated properly in the loop above. Let's fix the loop.

def _quick_aco_sim_fixed(dm, demands, capacity, H, n_ants=10, n_iterations=10, seed=42, time_limit=2.0):
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    global_best = np.inf
    demands = np.asarray(demands, dtype=np.float64)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
            
        iter_costs = []
        
        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break
                
            curr = 0
            rem_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            dist = 0.0
            
            while True:
                valid = ~visited
                valid[0] = False
                valid &= (demands <= rem_cap)
                
                if not np.any(valid):
                    dist += dm[curr, 0]
                    rem_cap = capacity
                    curr = 0
                    if np.all(visited):
                        break
                    continue
                
                w = tau[curr, :] * H[curr, :]
                w[~valid] = 0.0
                
                total_w = np.sum(w)
                if total_w <= 0:
                    valid_idx = np.where(valid)[0]
                    nxt = rng.choice(valid_idx)
                else:
                    p = w / total_w
                    nxt = rng.choice(n, p=p)
                
                dist += dm[curr, nxt]
                visited[nxt] = True
                rem_cap -= demands[nxt]
                curr = nxt
                
                if np.all(visited):
                    dist += dm[curr, 0]
                    break
                    
            iter_costs.append(dist)
            
        if iter_costs:
            iter_best = min(iter_costs)
            if iter_best < global_best:
                global_best = iter_best
            
            # Update Pheromone: tau = 0.9 * tau + 1 / iter_best
            # This matches the evaluator's rule.
            tau = 0.9 * tau + (1.0 / iter_best)
            
    return global_best

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Uses a hybrid of inverse distance and Clarke-Wright savings.
    Tunes parameters via local ACO simulation.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin for the 330s limit
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]
    
    # Base components
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Pre-compute savings terms
    d_depot = dm[0, :]
    d_i0 = d_depot[:, np.newaxis]
    d_j0 = d_depot[np.newaxis, :]
    sum_d_to_depot = d_i0 + d_j0
    sum_d_safe = sum_d_to_depot + 1e-9
    savings = sum_d_to_depot - dm
    savings_ratio = np.clip(savings / sum_d_safe, 0.0, 1.0)
    
    best_H = None
    best_sim_cost = np.inf
    
    # Search space for (gamma, alpha)
    # gamma: exponent for distance (1.0 is standard, >1 is more local)
    # alpha: weight for savings term (0.0 is pure distance, >0 adds structure)
    gammas = [0.8, 1.0, 1.2]
    alphas = [0.0, 0.5, 1.0, 1.5]
    
    # We will test a subset of these combinations to stay within time limits.
    # 50 nodes, 10 ants, 10 iterations is very fast (< 0.1s per test usually).
    # So we can test all 12 combinations comfortably.
    
    for g in gammas:
        for a in alphas:
            if time.time() > deadline - 10.0:
                break
                
            # Construct Heuristic
            # H = (1/d^g) * (1 + a * savings_ratio)
            dist_term = 1.0 / (dm_safe ** g)
            H_candidate = dist_term * (1.0 + a * savings_ratio)
            
            # Ensure positivity and finiteness
            H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)
            
            # Simulate ACO with this heuristic
            sim_cost = _quick_aco_sim_fixed(dm, demands, capacity, H_candidate, 
                                            n_ants=10, n_iterations=10, seed=42, 
                                            time_limit=1.0)
            
            if sim_cost < best_sim_cost:
                best_sim_cost = sim_cost
                best_H = H_candidate
                
    # Fallback if simulation failed or no valid H found
    if best_H is None:
        # Default to simple inverse distance with savings
        dist_term = 1.0 / dm_safe
        best_H = dist_term * (1.0 + 1.0 * savings_ratio)
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
        
    return best_H

# EVOLVE-BLOCK-END
