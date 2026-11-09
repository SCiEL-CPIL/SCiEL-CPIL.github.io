# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    N = 10  # number of nodes/edges
    D = 15  # number of demands
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute edge incidence matrices for each demand
    # For each demand, precompute which edges are used if routed clockwise vs counter-clockwise
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod N)
    # Counter-clockwise uses all other edges
    
    cw_matrix = np.zeros((D, N), dtype=np.int8)  # 1 if edge is used clockwise
    ccw_matrix = np.zeros((D, N), dtype=np.int8)  # 1 if edge is used counter-clockwise
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges
        if s <= t:
            for e in range(s, t):
                cw_matrix[i, e] = 1
        else:
            for e in range(s, N):
                cw_matrix[i, e] = 1
            for e in range(0, t):
                cw_matrix[i, e] = 1
        
        # Counter-clockwise edges (complement of clockwise)
        ccw_matrix[i] = 1 - cw_matrix[i]
    
    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)
    
    def compute_max_load(routing):
        """Compute the maximum edge load given a routing decision vector."""
        # Vectorized computation using matrix operations
        base_loads = cw_matrix.T @ weights  # shape (N,)
        diff_matrix = ccw_matrix - cw_matrix  # shape (D, N)
        w_r = weights * routing  # shape (D,)
        additional_loads = diff_matrix.T @ w_r  # shape (N,)
        loads = base_loads + additional_loads
        return np.max(loads)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 580 seconds, leaving 20s safety margin
    
    best_routing = np.zeros(D)
    best_score = compute_max_load(best_routing)
    
    # Strategy 1: Exhaustive search over all 2^15 = 32768 combinations
    # This is fast enough to do completely
    all_combinations = list(product([0, 1], repeat=D))
    
    for combo in all_combinations:
        routing = np.array(combo, dtype=float)
        score = compute_max_load(routing)
        if score < best_score:
            best_score = score
            best_routing = routing.copy()
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # If we finished exhaustive search, we have the optimal solution
    # Otherwise, do some local search to improve
    
    # Local search: try flipping individual bits and pairs of bits
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(D):
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            score = compute_max_load(trial)
            if score < best_score - 1e-9:
                best_score = score
                best_routing = trial
                improved = True
        
        # Try flipping pairs of bits
        for i in range(D):
            for j in range(i + 1, D):
                trial = best_routing.copy()
                trial[i] = 1.0 - trial[i]
                trial[j] = 1.0 - trial[j]
                score = compute_max_load(trial)
                if score < best_score - 1e-9:
                    best_score = score
                    best_routing = trial
                    improved = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Smooth surrogate optimization: use log-sum-exp approximation of max
    # to find a potentially better solution via coordinate descent on soft objective
    if time.time() < deadline - 60:  # leave 60s buffer
        def soft_max_load(routing, beta):
            """Compute soft-max approximation of max edge load."""
            base_loads = cw_matrix.T @ weights
            diff_matrix = ccw_matrix - cw_matrix
            w_r = weights * routing
            additional_loads = diff_matrix.T @ w_r
            loads = base_loads + additional_loads
            scaled = beta * loads
            max_scaled = np.max(scaled)
            if max_scaled > 500:  # avoid overflow
                return max_scaled / beta
            log_sum = np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled
            return log_sum / beta
        
        # Try different beta values for smooth approximation
        for beta in [1.0, 5.0, 10.0, 50.0]:
            if time.time() > deadline - 60:
                break
            
            current = best_routing.copy()
            current_soft = soft_max_load(current, beta)
            
            for iteration in range(1000):
                if time.time() > deadline - 60:
                    break
                
                improved_soft = False
                for i in range(D):
                    trial = current.copy()
                    trial[i] = 1.0 - trial[i]
                    trial_soft = soft_max_load(trial, beta)
                    if trial_soft < current_soft - 1e-10:
                        current = trial
                        current_soft = trial_soft
                        improved_soft = True
                
                if not improved_soft:
                    break
            
            discrete_score = compute_max_load(current)
            if discrete_score < best_score - 1e-9:
                best_score = discrete_score
                best_routing = current.copy()
    
    return best_routing

# EVOLVE-BLOCK-END
