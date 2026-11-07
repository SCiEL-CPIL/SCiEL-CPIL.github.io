import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for 3 fixed instances.
    Uses a two-phase search: coarse grid over gamma, then fine refinement.
    Proxy uses fast greedy construction with heuristic-guided selection.
    """
    start_time = time.time()
    deadline = start_time + 320.0
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    def compute_eta(gamma):
        """Compute heuristic matrix for given gamma."""
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(D > 0, D ** (-gamma), 1e-9)
        return np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    def greedy_tour(eta, rng_seed=42):
        """
        Fast greedy tour construction using the heuristic matrix.
        Picks the most attractive feasible neighbor at each step.
        Returns total tour distance.
        """
        current = 0
        current_load = 0.0
        total_distance = 0.0
        unvisited = set(range(1, n))
        
        while unvisited:
            rem_cap = capacity - current_load
            # Find feasible neighbors
            feasible = [j for j in unvisited if demands[j] <= rem_cap]
            
            if not feasible:
                total_distance += D[current, 0]
                break
            
            # Pick the one with highest heuristic value (most attractive)
            best_j = max(feasible, key=lambda j: eta[current, j])
            
            total_distance += D[current, best_j]
            current_load += demands[best_j]
            unvisited.remove(best_j)
            current = best_j
            
            if not unvisited:
                total_distance += D[current, 0]
                break
        
        return total_distance
    
    def evaluate_gamma(gamma, num_runs=20):
        """
        Evaluate gamma by running multiple greedy tours with slight randomization.
        Returns mean cost (lower is better).
        """
        eta = compute_eta(gamma)
        costs = []
        rng = np.random.RandomState(int(gamma * 1000) % (2**31))
        
        for _ in range(num_runs):
            # Add small random perturbation to simulate ACO diversity
            if rng.random() < 0.3:
                # Perturb: multiply by random factors in [0.8, 1.2]
                perturbation = 0.8 + 0.4 * rng.random(eta.shape)
                eta_pert = eta * perturbation
                eta_pert = np.where(np.isfinite(eta_pert) & (eta_pert > 0), eta_pert, 1e-9)
            else:
                eta_pert = eta
            
            cost = greedy_tour(eta_pert)
            costs.append(cost)
        
        return np.mean(costs)
    
    # Phase 1: Coarse grid search
    candidate_gammas = np.arange(0.5, 3.5, 0.25)
    
    best_gamma = 1.0
    best_cost = float('inf')
    
    for gamma in candidate_gammas:
        if time.time() > deadline - 60.0:
            break
        
        cost = evaluate_gamma(gamma, num_runs=20)
        
        if cost < best_cost:
            best_cost = cost
            best_gamma = gamma
    
    # Phase 2: Fine refinement around best_gamma
    if time.time() < deadline - 30.0:
        fine_gammas = np.arange(best_gamma - 0.2, best_gamma + 0.2, 0.05)
        fine_gammas = fine_gammas[(fine_gammas > 0.1) & (fine_gammas < 5.0)]
        
        for gamma in fine_gammas:
            if time.time() > deadline - 10.0:
                break
            
            cost = evaluate_gamma(gamma, num_runs=30)
            
            if cost < best_cost:
                best_cost = cost
                best_gamma = gamma
    
    # Phase 3: Ultra-fine polish
    if time.time() < deadline - 15.0:
        ultra_fine_gammas = np.arange(best_gamma - 0.05, best_gamma + 0.05, 0.01)
        ultra_fine_gammas = ultra_fine_gammas[(ultra_fine_gammas > 0.1) & (ultra_fine_gammas < 5.0)]
        
        for gamma in ultra_fine_gammas:
            if time.time() > deadline - 5.0:
                break
            
            cost = evaluate_gamma(gamma, num_runs=50)
            
            if cost < best_cost:
                best_cost = cost
                best_gamma = gamma
    
    # Construct final heuristic matrix
    final_heuristic = compute_eta(best_gamma)
    
    # Ensure shape is correct
    if final_heuristic.shape != D.shape:
        final_heuristic = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_heuristic, 1e-9)
    
    return final_heuristic
