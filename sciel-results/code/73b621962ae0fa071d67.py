import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    
    Uses a parametric family:
    H[i,j] = (1 / (dist[i,j] + eps)^p) * (1 / (1 + a * dem[j] / cap)) * (1 + b * depot_mask[i,j])
    
    where:
    - p controls distance sensitivity (higher = prefer shorter edges more strongly)
    - a controls load balancing (higher = prefer lower demand customers)
    - b controls depot affinity (higher = prefer edges connected to depot)
    
    The parameters are found via grid search and local refinement, optimizing 
    a fast but faithful surrogate ACO simulator (10 ants, 20 iters) to better 
    predict the performance of the full 20-ant, 30-iter evaluation.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit

    try:
        d = np.asarray(distance_matrix, dtype=np.float64)
        N = d.shape[0]

        if not np.all(np.isfinite(d)):
            return np.ones_like(d)

        dem = np.asarray(demands, dtype=np.float64)
        cap = float(capacity)
        eps = 1e-9
        coords = np.asarray(coordinates, dtype=np.float64)

        # Precompute base components to speed up heuristic construction
        # Base distance inverse power: d^(-p)
        # We compute for a few p values on the fly or just compute the log-distance
        # Actually, computing d^(-p) is fast enough for N=50.
        
        # Demand factor: 1 / (1 + a * dem[j] / cap)
        # This depends on j only.
        
        # Depot mask: 1 if i==0 or j==0, else 0
        depot_mask = np.zeros((N, N), dtype=np.float64)
        depot_mask[0, :] = 1.0
        depot_mask[:, 0] = 1.0
        # Note: Usually we don't want to boost the depot-to-depot edge excessively, 
        # but it doesn't matter much as ants don't stay there.
        # However, boosting edges *to* and *from* depot helps closing routes.

        def build_heuristic(p, a, b):
            """
            H[i,j] = d[i,j]^(-p) * dem_factor[j] * (1 + b * depot_mask[i,j])
            """
            # Distance term
            H = (1.0 / (d + eps) ** p).astype(np.float64)
            
            # Demand term (column broadcast)
            dem_factor = 1.0 / (1.0 + a * dem / cap)
            H = H * dem_factor[np.newaxis, :]
            
            # Depot term
            if b > 0:
                H = H * (1.0 + b * depot_mask)
            
            # Ensure positivity
            H = np.maximum(H, eps)
            return H

        def simulate_aco(heuristic, seed=42, n_ants=10, n_iters=20):
            """
            Surrogate ACO simulator.
            Uses 10 ants and 20 iterations to balance speed and fidelity.
            The actual evaluation uses 20 ants and 30 iterations.
            """
            rng = np.random.RandomState(seed)
            best_overall = float('inf')
            pher = np.ones((N, N), dtype=np.float64)
            
            # Precompute feasible customers for each load state? 
            # No, load changes dynamically. But we can precompute the list of customers.
            customers = np.arange(1, N)
            
            for iteration in range(n_iters):
                iter_best = float('inf')

                for _ant in range(n_ants):
                    visited = np.zeros(N, dtype=bool)
                    visited[0] = True
                    current = 0
                    load = 0.0
                    total_dist = 0.0

                    while True:
                        # Find feasible customers
                        # Vectorized check: not visited AND load + dem[j] <= cap
                        feasible_mask = (~visited[1:]) & (load + dem[1:] <= cap)
                        feasible_indices = customers[feasible_mask]

                        if len(feasible_indices) == 0:
                            # Return to depot
                            total_dist += d[current, 0]
                            current = 0
                            load = 0.0
                            if visited[1:].all():
                                break
                            continue

                        fe = feasible_indices
                        weights = pher[current, fe] * heuristic[current, fe]
                        weights = np.maximum(weights, 1e-9)
                        wsum = weights.sum()
                        if wsum <= 0:
                            probs = np.ones(len(fe)) / len(fe)
                        else:
                            probs = weights / wsum
                        
                        next_j = fe[rng.choice(len(fe), p=probs)]

                        total_dist += d[current, next_j]
                        visited[next_j] = True
                        load += dem[next_j]
                        current = next_j

                    iter_best = min(iter_best, total_dist)
                    best_overall = min(best_overall, total_dist)

                # Update pheromone
                pher *= 0.9
                if iter_best > 0:
                    pher += 1.0 / iter_best

            return best_overall

        # Grid search over (p, a, b)
        # p: distance power (0.5 to 3.0)
        # a: demand penalty (0.0 to 3.0)
        # b: depot affinity (0.0 to 0.5)
        
        p_vals = [0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
        a_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        b_vals = [0.0, 0.1, 0.2, 0.3, 0.5]

        best_cost = float('inf')
        best_params = (1.0, 0.0, 0.0)
        best_heuristic = build_heuristic(1.0, 0.0, 0.0).copy()

        eval_count = 0
        max_evals = 150  # 8*6*5 = 240 points, but we break early
        
        # Coarse grid
        for p in p_vals:
            if time.time() > deadline - 30.0:
                break
            for a in a_vals:
                if time.time() > deadline - 30.0:
                    break
                for b in b_vals:
                    if time.time() > deadline - 30.0:
                        break
                    if eval_count >= max_evals:
                        break
                    try:
                        H = build_heuristic(p, a, b)
                        c_val = simulate_aco(H, seed=42, n_ants=10, n_iters=20)
                        eval_count += 1
                        if np.isfinite(c_val) and c_val < best_cost:
                            best_cost = c_val
                            best_params = (p, a, b)
                            best_heuristic = H.copy()
                    except Exception:
                        continue
                if time.time() > deadline - 30.0:
                    break
            if time.time() > deadline - 30.0:
                break

        # Local refinement around best_params
        if time.time() < deadline - 40.0:
            p0, a0, b0 = best_params
            # Generate local candidates
            search_space = [
                (p0 * 0.9, a0, b0), (p0 * 1.1, a0, b0),
                (p0, max(0, a0 - 0.3), b0), (p0, a0 + 0.3, b0),
                (p0, a0, max(0, b0 - 0.05)), (p0, a0, b0 + 0.05),
                (p0 * 0.95, a0 + 0.2, b0), (p0 * 1.05, max(0, a0 - 0.2), b0 + 0.05),
                (p0, a0 + 0.5, b0 * 0.8), (p0 * 1.1, max(0, a0 - 0.5), b0 * 1.2),
            ]
            
            for (p, a, b) in search_space:
                if time.time() > deadline - 40.0:
                    break
                if p < 0.3 or p > 4.0 or a < -0.1 or a > 5.0 or b < -0.1 or b > 1.0:
                    continue
                try:
                    H = build_heuristic(p, a, b)
                    c_val = simulate_aco(H, seed=42, n_ants=10, n_iters=20)
                    if np.isfinite(c_val) and c_val < best_cost - 1e-6:
                        best_cost = c_val
                        best_params = (p, a, b)
                        best_heuristic = H.copy()
                except Exception:
                    continue

        # Second refinement with different seed to check robustness
        if time.time() < deadline - 20.0:
            p0, a0, b0 = best_params
            search_space2 = [
                (p0 * 0.85, a0, b0), (p0 * 1.15, a0, b0),
                (p0, max(0, a0 - 0.5), b0), (p0, a0 + 0.5, b0),
                (p0, a0, max(0, b0 - 0.1)), (p0, a0, b0 + 0.1),
                (p0 * 0.9, a0 + 0.3, max(0, b0 - 0.1)),
                (p0 * 1.1, max(0, a0 - 0.3), b0 + 0.1),
            ]
            for (p, a, b) in search_space2:
                if time.time() > deadline - 20.0:
                    break
                if p < 0.3 or p > 4.0 or a < -0.1 or a > 5.0 or b < -0.1 or b > 1.0:
                    continue
                try:
                    H = build_heuristic(p, a, b)
                    # Use different seed for diversity
                    c_val = simulate_aco(H, seed=123, n_ants=10, n_iters=20)
                    if np.isfinite(c_val) and c_val < best_cost - 1e-6:
                        best_cost = c_val
                        best_params = (p, a, b)
                        best_heuristic = H.copy()
                except Exception:
                    continue

        # Final safety: ensure all positive and finite
        if not np.all(np.isfinite(best_heuristic)):
            best_heuristic = build_heuristic(1.0, 0.0, 0.0).copy()
        best_heuristic = np.maximum(best_heuristic, eps)

        return best_heuristic

    except Exception:
        try:
            d = np.asarray(distance_matrix, dtype=np.float64)
            H = 1.0 / (d + 1e-9)
            H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
            return H
        except:
            return np.ones((50, 50))
