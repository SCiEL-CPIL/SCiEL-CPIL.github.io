import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate 2 * L(v) where L(v) = number of edges in N(v)
    B = A @ A
    two_L = np.sum(A * B, axis=1)
    L = two_L / 2.0
    
    # Initial Score: L(v) - d(v)
    # This heuristic favors vertices whose neighbors are densely connected (less "wasted" blocking)
    # and penalizes high degree.
    scores = L - degrees
    
    # Helper function to evaluate the size of the greedy independent set
    def eval_scores(s):
        order = np.argsort(s)[::-1]
        selected = np.zeros(n, dtype=float)
        count = 0
        # A is a dense float matrix
        for v in order:
            # Check if any neighbor of v is already selected
            # np.dot(selected, A[v]) computes the sum of selected neighbors
            if np.dot(selected, A[v]) == 0.0:
                selected[v] = 1.0
                count += 1
        return count

    best_score = eval_scores(scores)
    best_scores = scores.copy()
    
    rng = np.random.default_rng(42)
    no_improve_count = 0
    max_no_improve = 2000
    
    # Local Search: Hill Climbing with Random Swaps
    while time.time() - start_time < deadline:
        # Pick two distinct vertices to swap
        i, j = rng.choice(n, 2, replace=False)
        
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        # Evaluate new configuration
        new_score = eval_scores(scores)
        
        if new_score > best_score:
            best_score = new_score
            best_scores = scores.copy()
            no_improve_count = 0
        else:
            # Revert swap
            scores[i], scores[j] = scores[j], scores[i]
            no_improve_count += 1
            
            # If stuck for a while, perturb the best solution to escape local optima
            if no_improve_count > max_no_improve:
                # Add small random noise to best scores to create a new starting point
                noise = rng.normal(0, 0.5, n)
                scores = best_scores + noise
                # Re-evaluate the perturbed state to set a new baseline for the hill climb
                current_score = eval_scores(scores)
                if current_score > best_score:
                    best_score = current_score
                    best_scores = scores.copy()
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.5) # Increase patience slightly

    return best_scores
