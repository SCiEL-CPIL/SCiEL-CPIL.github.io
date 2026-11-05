import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0
    
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Initial scores based on degree
    # Lower degree is better for greedy MIS
    degrees = A.sum(axis=1)
    
    # Initial order: sort by degree ascending (lowest degree first)
    # Ties broken by index for determinism
    initial_order = np.argsort(degrees)
    
    # Function to evaluate IS size for a given order
    def get_is_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        # We iterate through the order
        for v in order:
            if not selected[v]:
                # Check if any neighbor is selected
                # A[v] is a boolean mask of neighbors
                # selected[A[v]] gets the selected status of neighbors
                # If any are True, we cannot select v
                if not np.any(selected[A[v]]):
                    selected[v] = True
                    count += 1
        return count

    best_order = initial_order
    best_size = get_is_size(best_order)
    
    # Local Search: Hill Climbing
    # We try to swap pairs of elements in the order
    # To make it faster, we can use a list for order and modify in place
    current_order = best_order.copy()
    current_size = best_size
    
    # Random number generator
    rng = np.random.default_rng(42)
    
    no_improve_count = 0
    max_no_improve = 5000 # Restart if stuck
    
    while time.time() < DEADLINE:
        if no_improve_count > max_no_improve:
            # Restart with a different heuristic or random perturbation
            # Simple restart: random shuffle
            current_order = np.arange(n)
            rng.shuffle(current_order)
            current_size = get_is_size(current_order)
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
            no_improve_count = 0
            max_no_improve = 5000
            
        # Pick two random indices to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate
        new_size = get_is_size(current_order)
        
        if new_size > current_size:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order.copy()
            no_improve_count = 0
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_count += 1
            
    # Convert best_order to scores
    # The decoder sorts by score descending.
    # So the first element in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
