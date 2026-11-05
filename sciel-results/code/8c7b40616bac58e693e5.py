import numpy as np
import time

# Global state to track time across multiple calls
_start_time = None
_total_budget = 110.0  # 10s margin from 120s limit

def priority(adjacency):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    remaining_time = _total_budget - elapsed
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # If time is critically low, fall back to fast heuristic
    if remaining_time < 2.0:
        degrees = adj.sum(axis=1)
        return -degrees.astype(float)
    
    # 1. Initial Heuristic
    # Using a weighted degree heuristic as a better start
    degrees = adj.sum(axis=1)
    # Sum of degrees of neighbors
    # adj is bool, degrees is int
    # weighted_deg[i] = sum(degrees[j] for j in neighbors(i))
    weighted_deg = adj @ degrees
    
    # Score: We want vertices with low degree and low neighbor-degree to be picked first
    # Greedy picks descending score.
    # So score = - (degree + 0.5 * weighted_deg)
    alpha = 0.5
    current_scores = -(degrees + alpha * weighted_deg).astype(float)
    
    # Helper function to evaluate the independent set size
    def evaluate(scores):
        # Argsort descending
        order = np.argsort(scores)[::-1]
        selected = np.zeros(n, dtype=bool)
        count = 0
        # Pre-compute neighbor indices for faster access? 
        # adj[v] is a boolean mask. selected[adj[v]] is fast.
        for v in order:
            if not selected[v]:
                # Check if any neighbor is selected
                # selected[adj[v]] returns a boolean array of neighbors' status
                # .any() checks if any are True
                if not selected[adj[v]].any():
                    selected[v] = True
                    # Mark all neighbors as selected/blocked
                    selected[adj[v]] = True
                    count += 1
        return count

    best_size = evaluate(current_scores)
    best_scores = current_scores.copy()
    
    # 2. Local Search (Hill Climbing with Random Restarts)
    rng = np.random.default_rng(42)
    
    # We will run until time runs out or we are stuck for a long time
    # Since we don't know how many graphs are left, we just use the remaining time.
    
    no_improve_count = 0
    max_no_improve = 2000 # Adjust based on speed
    
    while True:
        elapsed = time.time() - _start_time
        if time.time() - _start_time > _total_budget - 1.0:
            break
            
        # If stuck, perturb
        if no_improve_count > max_no_improve:
            # Randomly swap a few pairs in best_scores to escape local optimum
            for _ in range(10):
                i, j = rng.choice(n, 2, replace=False)
                best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
            current_scores = best_scores.copy()
            best_size = evaluate(current_scores)
            no_improve_count = 0
            continue
        
        # Propose a move: Swap two random vertices
        i, j = rng.choice(n, 2, replace=False)
        
        # Apply move
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        # Evaluate
        new_size = evaluate(current_scores)
        
        if new_size > best_size:
            best_size = new_size
            best_scores = current_scores.copy()
            no_improve_count = 0
        else:
            # Revert move
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            no_improve_count += 1
            
    return best_scores
