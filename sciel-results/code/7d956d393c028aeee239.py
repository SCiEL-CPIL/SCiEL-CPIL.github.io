import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for the ACO-CVRP instances.
    Strategy: Geometric angle-based heuristic combined with demand-awareness.
    We search for optimal weights for distance and angle smoothness.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Precompute geometry features
    # Angle matrix: A[i, j] represents how "straight" the turn is if we came from k to i and go to j?
    # Actually, standard ACO heuristics are static. Angle smoothness is path-dependent.
    # However, we can approximate a "smoothness" metric for edges.
    # A common static approximation is to penalize edges that are "sharp" relative to the depot or general flow.
    # But a better structural change: Use a heuristic that is NOT just 1/d^gamma.
    # Let's use: eta[i, j] = (1/d_ij)^alpha * (1 / (1 + demand[j]))^beta * (1 + cos_angle_term)^gamma
    # Since we can't know the previous node in a static matrix, we can't use dynamic angles directly in the matrix.
    
    # Alternative Structural Change:
    # Instead of tuning a single gamma for distance, we tune a combination of:
    # 1. Inverse distance
    # 2. Demand penalty (bias towards heavy nodes early? or avoid them?)
    # 3. "Cluster affinity" - how well does node j fit with the neighbors of i?
    
    # Let's try a different family: 
    # eta[i, j] = (1/d_ij)^a * (1/d_j_depot)^b
    # This biases towards nodes that are close to the depot (good for returning) or far?
    # Actually, in CVRP, you want to visit nodes that are close to each other.
    # Let's compute a "local density" or "neighborhood similarity" score.
    
    # Precompute: For each node i, what is the average distance to its k-nearest neighbors?
    # Or, simpler: eta[i, j] = 1 / (d_ij * (1 + |d_i_depot - d_j_depot|))
    # This encourages visiting nodes that are at similar distances from the depot (same "ring").
    
    d_depot = D[0, :]
    
    def make_eta(a, b, c):
        """
        eta[i, j] = (1/d_ij)^a * (1 / (1 + |d_i_depot - d_j_depot|))^b * (1 / (1 + demand[j]))^c
        """
        # Avoid division by zero
        D_safe = np.where(D > 0, D, 1.0)
        
        term_dist = (1.0 / D_safe) ** a
        
        # Ring similarity: small difference in distance to depot is good
        diff_ring = np.abs(d_depot[:, None] - d_depot[None, :])
        term_ring = (1.0 / (1.0 + diff_ring)) ** b
        
        # Demand penalty: heavier nodes are harder to place, so maybe less attractive?
        # Or more attractive to get them done?
        # Let's assume heavier nodes should be visited earlier in the route, but since we don't know the step,
        # we just bias against them slightly to balance the load? Or bias towards them?
        # Let's try biasing towards lighter nodes to fill capacity efficiently?
        # Actually, high demand nodes are "expensive" to serve in terms of capacity.
        # Let's make them less attractive to encourage spreading them out?
        # Or more attractive to ensure they are visited?
        # Let's use: (1 + demand[j])^(-c)
        demand_term = (1.0 + demands[None, :]) ** (-c)
        
        eta = term_dist * term_ring * demand_term
        
        # Diagonal should be low (or 0, but we need positive)
        np.fill_diagonal(eta, 1e-9)
        
        # Ensure positivity and finiteness
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        return eta

    def run_aco(eta, num_ants, num_iters, seed):
        """
        Run a small ACO to evaluate the heuristic.
        Returns the best cost found.
        """
        rng = np.random.RandomState(seed)
        best_cost = float('inf')
        tau = np.ones_like(D)

        for it in range(num_iters):
            iter_best = float('inf')

            for ant in range(num_ants):
                current = 0
                current_load = 0.0
                total_distance = 0.0
                unvisited = list(range(1, n))

                while unvisited:
                    rem_cap = capacity - current_load
                    # Find feasible nodes
                    feasible = [j for j in unvisited if demands[j] <= rem_cap]

                    if not feasible:
                        total_distance += D[current, 0]
                        break

                    if len(feasible) == 1:
                        next_node = feasible[0]
                    else:
                        f_arr = np.array(feasible)
                        weights = tau[current, f_arr] * eta[current, f_arr]
                        total_w = np.sum(weights)
                        if total_w <= 0:
                            next_node = f_arr[rng.randint(len(f_arr))]
                        else:
                            probs = weights / total_w
                            next_node = f_arr[rng.choice(len(f_arr), p=probs)]

                    total_distance += D[current, next_node]
                    current_load += demands[next_node]
                    unvisited.remove(next_node)
                    current = next_node

                    if not unvisited:
                        total_distance += D[current, 0]
                        break

                if total_distance < iter_best:
                    iter_best = total_distance

            if iter_best < best_cost:
                best_cost = iter_best

            # Pheromone update
            tau *= 0.9
            if iter_best > 0:
                tau += 1.0 / iter_best

        return best_cost

    # Search parameters: a (dist exp), b (ring sim exp), c (demand exp)
    # Ranges:
    # a: 0.5 to 3.0
    # b: 0.0 to 2.0
    # c: 0.0 to 1.0
    
    best_params = (1.0, 0.5, 0.5)
    best_val = float('inf')
    
    # Coarse Grid
    a_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    b_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
    c_vals = [0.0, 0.5, 1.0]
    
    for a in a_vals:
        if time.time() > deadline - 60.0:
            break
        for b in b_vals:
            if time.time() > deadline - 60.0:
                break
            for c in c_vals:
                if time.time() > deadline - 60.0:
                    break
                eta = make_eta(a, b, c)
                # Use small ACO for speed
                cost = run_aco(eta, 5, 5, seed=int(a*100 + b*10 + c))
                if cost < best_val:
                    best_val = cost
                    best_params = (a, b, c)
    
    # Fine Grid around best
    a0, b0, c0 = best_params
    a_fine = np.arange(max(0.1, a0 - 0.5), min(4.0, a0 + 0.5), 0.25)
    b_fine = np.arange(max(0.0, b0 - 0.5), min(3.0, b0 + 0.5), 0.25)
    c_fine = np.arange(max(0.0, c0 - 0.25), min(1.5, c0 + 0.25), 0.25)
    
    for a in a_fine:
        if time.time() > deadline - 30.0:
            break
        for b in b_fine:
            if time.time() > deadline - 30.0:
                break
            for c in c_fine:
                if time.time() > deadline - 30.0:
                    break
                eta = make_eta(a, b, c)
                cost = run_aco(eta, 8, 8, seed=int(a*100 + b*10 + c))
                if cost < best_val:
                    best_val = cost
                    best_params = (a, b, c)
    
    # Final Polish
    a0, b0, c0 = best_params
    a_polish = np.arange(max(0.1, a0 - 0.2), min(4.0, a0 + 0.2), 0.1)
    b_polish = np.arange(max(0.0, b0 - 0.2), min(3.0, b0 + 0.2), 0.1)
    c_polish = np.arange(max(0.0, c0 - 0.1), min(1.5, c0 + 0.1), 0.1)
    
    for a in a_polish:
        if time.time() > deadline - 10.0:
            break
        for b in b_polish:
            if time.time() > deadline - 10.0:
                break
            for c in c_polish:
                if time.time() > deadline - 10.0:
                    break
                eta = make_eta(a, b, c)
                cost = run_aco(eta, 10, 10, seed=int(a*100 + b*10 + c))
                if cost < best_val:
                    best_val = cost
                    best_params = (a, b, c)
    
    # Generate final heuristic
    a_final, b_final, c_final = best_params
    final_eta = make_eta(a_final, b_final, c_final)
    
    # Safety checks
    if final_eta.shape != D.shape:
        final_eta = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_eta, 1e-9)
        
    final_eta = np.where(np.isfinite(final_eta) & (final_eta > 0), final_eta, 1e-9)
    
    return final_eta
