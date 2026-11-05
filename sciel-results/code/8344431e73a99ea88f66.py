"""Improved priority for maximum independent set via simulated annealing on permutations."""
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for greedy independent set selection.
    Uses simulated annealing on vertex orderings to maximize the greedy independent set size.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Precompute neighbors as boolean arrays for fast checking
    neighbors = adjacency.astype(bool)
    
    def greedy_size(order):
        """Compute the size of the independent set produced by greedy selection in given order."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                # Check if v is already blocked (neighbor of selected)
                if selected[neighbors[v]].any():
                    continue
                selected[v] = True
                count += 1
        return count
    
    # Initialize with degree-based ordering (low degree first)
    degrees = adjacency.sum(axis=1).astype(float)
    # Start with a random permutation perturbed by degree
    current_order = np.argsort(degrees + np.random.rand(n) * 0.1)
    current_size = greedy_size(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # Simulated annealing parameters
    start_time = time.time()
    deadline = 120.0  # Keep safety margin
    max_iterations = 100000
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iterations):
        if time.time() - start_time > deadline - 5.0:
            break
        if T < T_min:
            break
        
        # Propose a swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_size = greedy_size(new_order)
        
        delta = new_size - current_size
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_order = new_order
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        
        T *= cooling
    
    # Now derive scores from best_order: earlier vertices get higher priority
    # Use inverse position as score
    scores = np.zeros(n, dtype=float)
    for rank, v in enumerate(best_order):
        scores[v] = float(n - rank)
    
    return scores
