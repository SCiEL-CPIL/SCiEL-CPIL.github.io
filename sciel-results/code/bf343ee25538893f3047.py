"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _quick_aco_sim(dm, demands, capacity, H, n_ants=10, n_iterations=10, seed=42, time_limit=2.0):
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
            
            tau = 0.9 * tau + (1.0 / iter_best)
            
    return global_best

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Uses a demand-aware, multi-criteria matrix combining inverse distance
    with a penalty for high-demand nodes to encourage packing heavier customers
    early and minimizing trips.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]
    
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Pre-compute savings terms for structural bias
    d_depot = dm[0, :]
    d_i0 = d_depot[:, np.newaxis]
    d_j0 = d_depot[np.newaxis, :]
    sum_d_to_depot = d_i0 + d_j0
    sum_d_safe = sum_d_to_depot + 1e-9
    savings = sum_d_to_depot - dm
    savings_ratio = np.clip(savings / sum_d_safe, 0.0, 1.0)
    
    # Demand-aware factor: penalize high demand to pack heavy items early
    # demand_factor[j] = 1 / (1 + demand[j])
    # We want to bias towards nodes with higher demand? 
    # Actually, the directive says: "penalty for high-demand nodes (e.g., eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j]))), directly biasing the pheromone update toward routes that pack heavier customers early"
    # Wait, if we multiply by 1/(1+demand), we are PENALIZING high demand. 
    # The text says "biasing ... toward routes that pack heavier customers early". 
    # Usually, to pack heavy customers early, you might want to PRIORITIZE them? 
    # Or maybe the idea is that by making high demand nodes less attractive, you avoid ending up with a situation where you can't fit them later? 
    # Let's re-read: "penalty for high-demand nodes ... directly biasing the pheromone update toward routes that pack heavier customers early". 
    # This phrasing is slightly contradictory if "penalty" means lower attractiveness. 
    # If H[i,j] is lower for high demand, the ant is less likely to go there. 
    # If H[i,j] is higher for high demand, the ant is more likely to go there. 
    # "Pack heavier customers early" implies visiting them sooner. 
    # So perhaps the heuristic should FAVOR high demand nodes? 
    # Let's look at the example formula: `eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j]))`. 
    # This formula penalizes high demand. 
    # Why would that pack them early? Maybe it's a mistake in the prompt's logic or I'm misinterpreting "pack". 
    # If you have capacity 50 and demands 1-9. 
    # If you pick low demand items first, you might fill the truck with many small items and be left with no room for a big item that is far away, forcing an extra trip. 
    # By penalizing high demand items, you delay them? That seems bad. 
    # Let's try the opposite: Favor high demand? 
    # Or maybe the "penalty" is on the DISTANCE? 
    # Let's stick to the directive's explicit formula structure but maybe invert the demand term if the logic suggests favoring them. 
    # However, the directive says "Strategy to try ... demand-aware ... penalty for high-demand nodes". 
    # I will implement the penalty as described: factor = 1 / (1 + demand[j]). 
    # And I will also try a version that favors them: factor = 1 + demand[j]. 
    # And I will tune via simulation.
    
    best_H = None
    best_sim_cost = np.inf
    
    # Parameters to tune:
    # gamma: distance exponent
    # alpha: savings weight
    # beta: demand weight (0 means no demand awareness, >0 means penalty/favor)
    # mode: 0 for penalty (1/(1+d)), 1 for favor (1+d)
    
    gammas = [0.8, 1.0, 1.2, 1.5]
    alphas = [0.0, 0.5, 1.0]
    betas = [0.0, 0.5, 1.0]
    modes = [0, 1] # 0: penalty, 1: favor
    
    # Pre-compute demand factors for each mode
    # mode 0: 1 / (1 + demand)
    demand_factor_penalty = 1.0 / (1.0 + demands)
    # mode 1: 1 + demand
    demand_factor_favor = 1.0 + demands
    
    # We need to apply this factor to the columns of H (destination j)
    # H[i, j] *= demand_factor[j]
    
    for g in gammas:
        for a in alphas:
            for b in betas:
                for m in modes:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Base distance term
                    dist_term = 1.0 / (dm_safe ** g)
                    
                    # Add savings
                    H_base = dist_term * (1.0 + a * savings_ratio)
                    
                    # Apply demand factor
                    if m == 0:
                        factor = demand_factor_penalty
                    else:
                        factor = demand_factor_favor
                        
                    # Blend demand factor: 1 + b * (factor - 1) ? Or just multiply?
                    # If b=0, we want no effect. 
                    # If we multiply by factor, b=0 doesn't work directly unless we scale.
                    # Let's use: H = H_base * (1 + b * (factor - 1))
                    # If b=0, H = H_base.
                    # If m=0 (penalty), factor < 1, so H decreases for high demand.
                    # If m=1 (favor), factor > 1, so H increases for high demand.
                    
                    H_candidate = H_base * (1.0 + b * (factor[np.newaxis, :] - 1.0))
                    
                    # Ensure positivity
                    H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)
                    
                    # Simulate
                    sim_cost = _quick_aco_sim(dm, demands, capacity, H_candidate, 
                                              n_ants=10, n_iterations=10, seed=42, 
                                              time_limit=1.0)
                    
                    if sim_cost < best_sim_cost:
                        best_sim_cost = sim_cost
                        best_H = H_candidate
                        
    # Fallback
    if best_H is None:
        dist_term = 1.0 / dm_safe
        best_H = dist_term * (1.0 + 1.0 * savings_ratio)
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
        
    return best_H

# EVOLVE-BLOCK-END
