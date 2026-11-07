"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_aco_sim(dm, demands, capacity, H, n_ants=10, n_iterations=10, seed=42, time_limit=2.0):
    """
    Fast simulation of the evaluator's ACO to estimate heuristic quality.
    Mirrors the evaluator: 20 ants, 30 iters, tau=0.9*tau+1/best.
    We use fewer ants/iters for speed, but the relative ranking should hold.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)

    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                row_tau = tau[current_node]
                row_eta = H[current_node]

                weights = np.zeros(n)
                weights[valid_mask] = row_tau[valid_mask] * row_eta[valid_mask]

                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for ACO.
    Uses a fast ACO simulation to tune the heuristic parameters.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s kill

    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]
    
    # Pre-compute savings matrix for the hybrid option
    # Savings S_ij = d(i,0) + d(j,0) - d(i,j)
    d_depot = dm[0, :]
    sum_d_to_depot = d_depot[:, np.newaxis] + d_depot[np.newaxis, :]
    savings = sum_d_to_depot - dm
    # Normalize savings to [0, 1] roughly to keep heuristic values sane
    # Max possible savings is 2*max_d, min is 0 (or negative)
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_norm = np.clip(savings / max_savings, 0.0, 1.0)
    else:
        savings_norm = np.zeros_like(savings)

    # Replace 0 distances with a large number to avoid inf in 1/d^gamma
    dm_safe = np.where(dm > 1e-10, dm, 1e10)

    best_avg_cost = np.inf
    best_H = None

    # Candidate parameters to test
    # We test a range of gamma values for the power law
    # and a few lambda values for the savings boost
    gammas = np.linspace(0.5, 3.0, 11)
    lambdas = [0.0, 0.5, 1.0]

    for g in gammas:
        if time.time() > deadline - 30.0:
            break
        
        base_eta = 1.0 / (dm_safe ** g)
        
        for lam in lambdas:
            if time.time() > deadline - 10.0:
                break
            
            if lam == 0.0:
                H_candidate = base_eta
            else:
                H_candidate = base_eta * (1.0 + lam * savings_norm)
            
            # Ensure validity
            H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)
            
            # Evaluate with 2 seeds to reduce noise
            costs = []
            for s in [1, 2]:
                cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                                     n_ants=15, n_iterations=15, seed=s, time_limit=1.5)
                if cost < np.inf:
                    costs.append(cost)
            
            if costs:
                avg_cost = np.mean(costs)
                if avg_cost < best_avg_cost:
                    best_avg_cost = avg_cost
                    best_H = H_candidate

    # Fallback if no valid heuristic was found
    if best_H is None:
        best_H = 1.0 / dm_safe
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    return best_H
# EVOLVE-BLOCK-END
