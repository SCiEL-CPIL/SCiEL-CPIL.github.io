import numpy as np
import time
import random
from scipy.optimize import minimize

def _greedy_set_size_from_scores(adj, scores):
    """
    Run the fixed greedy decoder given continuous scores.
    Vertices are processed in descending order of score.
    """
    n = adj.shape[0]
    # Get indices sorted by score descending
    # Tie-breaking: random permutation of indices for stability if scores are equal?
    # Actually, for optimization stability, we should handle ties consistently.
    # However, Nelder-Mead moves continuously, so exact ties are rare unless we hit symmetries.
    # Using np.argsort with kind='stable' or similar might be good, but standard quicksort is fine.
    order = np.argsort(-scores)
    
    chosen = set()
    blocked = np.zeros(n, dtype=bool)
    count = 0
    
    for v in order:
        v = int(v)
        if not blocked[v]:
            chosen.add(v)
            count += 1
            # Block v and its neighbors
            blocked[v] = True
            nbrs = np.nonzero(adj[v])[0]
            if len(nbrs) > 0:
                blocked[nbrs] = True
                
    return -count # Negate for minimization

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Deadline guard: 140s to be safe against 150s limit
    deadline = start_time + 140.0
    
    # Initial scores: Low degree -> High priority (High score)
    # We want low degree vertices to be picked first.
    # So score should be inversely related to degree.
    deg = adj.sum(axis=1)
    # Normalize degrees to [0, 1]
    if deg.max() > 0:
        norm_deg = deg / deg.max()
    else:
        norm_deg = np.zeros(n)
    
    # Initial guess: scores = 1 - norm_deg + small random noise
    # This ensures low degree vertices have higher scores.
    init_scores = 1.0 - norm_deg + np.random.normal(0, 0.01, n)
    
    best_scores = init_scores.copy()
    best_val = _greedy_set_size_from_scores(adj, best_scores) # This is -size
    
    # Define objective for scipy
    def objective(x):
        return _greedy_set_size_from_scores(adj, x)
        
    # Nelder-Mead parameters
    # maxiter: we don't know how many it will take, but we have a time limit.
    # We can use a wrapper that checks time, but scipy doesn't natively support time-based stopping in objective easily without global state.
    # Instead, we run multiple restarts with limited iterations, checking time between them.
    
    # Strategy: Multiple random restarts of Nelder-Mead.
    # Each run has a max_iter limit.
    # We check time after each run.
    
    max_iter_per_run = 2000
    tol = 1e-5
    
    runs = 0
    max_runs = 20
    
    while time.time() < deadline and runs < max_runs:
        # Random restart: perturb initial scores
        if runs == 0:
            x0 = init_scores.copy()
        else:
            # Start from best known or random?
            # Starting from best known is usually better for exploitation.
            # Adding noise allows exploration.
            x0 = best_scores + np.random.normal(0, 0.1, n)
            
        # Run optimization
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': max_iter_per_run, 'xatol': 1e-4, 'fatol': 1e-4})
        except Exception:
            break
            
        if result.success or result.fun < best_val:
            if result.fun < best_val:
                best_val = result.fun
                best_scores = result.x.copy()
                
        runs += 1
        
        # Check if we have enough time for another run
        # Estimate time per run: if first run took 5s, we can do ~28 runs in 140s.
        # If it took 10s, ~14 runs.
        # The loop condition handles this naturally.
        
    # Convert best_scores to final priority scores
    # The greedy decoder sorts by score descending.
    # So we just return the optimized scores directly?
    # Yes, the interface asks for "one score per vertex".
    # The evaluator sorts by these scores.
    
    # Ensure scores are distinct to avoid tie-breaking issues in evaluator?
    # The evaluator likely uses a stable sort or specific tie-break.
    # Adding a tiny epsilon based on index can ensure distinctness without changing order significantly.
    # However, if the optimizer found a good order, we should preserve it.
    # Let's add a very small noise to break ties deterministically based on original index?
    # Or just return the scores. If ties occur, the evaluator's behavior is fixed.
    # To be safe, let's add a tiny perturbation to make all scores unique.
    final_scores = best_scores.copy()
    # Add tiny noise to break ties
    final_scores += np.random.uniform(0, 1e-9, n)
    
    return final_scores
