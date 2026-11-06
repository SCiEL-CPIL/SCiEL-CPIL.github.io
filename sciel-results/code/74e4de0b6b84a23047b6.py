"""Improved program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            # Update available mask for neighbors
            # adjacency_bool is a boolean matrix
            available[adjacency_bool[v]] = False
    return count


def _evaluate_order(adjacency_bool, order):
    """Evaluate a priority vector by running the greedy decoder."""
    return _greedy_size(adjacency_bool, order)


def _power_iteration_eigenvector(M, n, iters=50, tol=1e-8):
    """
    Fast power iteration for the principal eigenvector of a symmetric matrix.
    Returns the eigenvector (normalized to unit L2 norm).
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        new_v = M @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-12:
            break
        new_v /= norm
        if np.abs(np.dot(new_v, v)) > 1 - tol:
            v = new_v
            break
        v = new_v
    return v


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    # Compute spectral scores using power iteration on adjacency matrix
    # Vertices with low principal eigenvector values are often peripheral
    A_float = A.astype(float)
    principal = _power_iteration_eigenvector(A_float, n, iters=50)
    principal = np.abs(principal)
    
    # Normalize spectral scores
    p_min = principal.min()
    p_max = principal.max()
    if p_max > p_min:
        spectral_norm = (principal - p_min) / (p_max - p_min)
    else:
        spectral_norm = np.zeros(n)
        
    # Generate candidate orders
    rng = np.random.default_rng(42)
    
    # Heuristic 1: Ascending Degree
    order1 = np.argsort(deg, kind='stable')
    
    # Heuristic 2: Spectral (low spectral score first, i.e., peripheral)
    order2 = np.argsort(spectral_norm, kind='stable')
    
    # Heuristic 3: Combined Degree and Spectral
    # We want low degree and low spectral centrality
    score_h3 = deg / (deg.max() + 1e-6) * 0.5 + spectral_norm * 0.5
    order3 = np.argsort(score_h3, kind='stable')
    
    # Heuristic 4: Random
    order4 = rng.permutation(n)
    
    # Heuristic 5: Degree with noise
    noise = rng.normal(0, 0.5, size=n)
    score_h5 = deg + noise
    order5 = np.argsort(score_h5, kind='stable')
    
    candidates = [order1, order2, order3, order4, order5]
    
    best_size = -1
    best_order = None
    
    # Evaluate initial candidates
    for order in candidates:
        if time.time() > deadline:
            break
        size = _evaluate_order(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
            
    # Local Search: Multi-start Hill Climbing
    # We will perform several restarts. In each restart, we start from a perturbed version
    # of the current best or a random heuristic, and do hill climbing.
    
    # Estimate time per evaluation. N=200, greedy is fast.
    # Let's assume ~1-2ms per eval. 115s allows ~50k-100k evals.
    
    num_restarts = 40
    max_iters_per_restart = 1500
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Initialize current order for this restart
        if r == 0:
            current_order = best_order.copy()
        elif r % 3 == 1:
            # Perturb the best order found so far
            current_order = best_order.copy()
            for _ in range(10):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 3 == 2:
            # Start from a random heuristic candidate
            current_order = candidates[r % len(candidates)].copy()
            for _ in range(5):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        else:
            # Start from a new random order
            current_order = rng.permutation(n)
                
        current_size = _evaluate_order(A, current_order)
        
        # If this restart starts better than global best, update
        if current_size > best_size:
            best_size = current_size
            best_order = current_order.copy()
            
        no_improve_count = 0
        max_no_improve = 150
        
        for i in range(max_iters_per_restart):
            if time.time() > deadline - 5:
                break
                
            if no_improve_count > max_no_improve:
                break
                
            # Move: Pick a random vertex and move it to a random position (Insertion)
            v_idx = rng.integers(0, n)
            target_pos = rng.integers(0, n)
            
            if v_idx == target_pos:
                continue
                
            v = current_order[v_idx]
            
            new_order = current_order.copy()
            new_order = np.delete(new_order, v_idx)
            if v_idx < target_pos:
                insert_idx = target_pos - 1
            else:
                insert_idx = target_pos
            new_order = np.insert(new_order, insert_idx, v)
            
            new_size = _evaluate_order(A, new_order)
            
            if new_size > current_size:
                current_order = new_order
                current_size = new_size
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve_count += 1
                
    # Final polish: try adjacent swaps around the best order
    if time.time() < deadline - 2:
        polish_order = best_order.copy()
        polish_size = _evaluate_order(A, polish_order)
        # Use swaps for speed in the final phase
        for _ in range(1000):
            if time.time() > deadline - 1:
                break
            i = rng.integers(0, n - 1)
            j = i + 1
            polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
            swap_size = _evaluate_order(A, polish_order)
            if swap_size > polish_size:
                polish_size = swap_size
                if polish_size > best_size:
                    best_size = polish_size
                    best_order = polish_order.copy()
            else:
                polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
                
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # So vertex at index 0 in order gets highest priority.
    # priority[v] = n - index(v)
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
