import numpy as np
import time

def _greedy_decode(adjacency, scores):
    """Decode scores into an independent set using greedy selection."""
    n = len(scores)
    order = np.argsort(-scores, kind='stable')
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            selected[v] = True
            count += 1
            # Mark neighbors as unavailable
            if adjacency[v].any():
                available[adjacency[v]] = False
    return selected, count

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Initialize scores with degree-based heuristic
    degrees = adj.sum(axis=1).astype(float)
    nbr_deg_sums = adj @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
    scores = -degrees + avg_nbr_deg
    
    # Get initial solution
    selected, best_count = _greedy_decode(adj, scores)
    
    # Local search: try swapping pairs of vertices to improve count
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5.0:
        # Try random swaps
        improved = False
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            
            i, j = rng.choice(n, size=2, replace=False)
            
            # Skip if either is already selected in a way that would break independence
            # Actually, we just swap scores and re-decode
            old_si, old_sj = scores[i], scores[j]
            scores[i], scores[j] = old_sj, old_si
            
            new_selected, new_count = _greedy_decode(adj, scores)
            
            if new_count > best_count:
                best_count = new_count
                selected = new_selected
                improved = True
            else:
                # Revert
                scores[i], scores[j] = old_si, old_sj
        
        # If no improvement in a full pass, try a different strategy
        if not improved:
            # Try small perturbations: add noise to scores
            noise = rng.normal(0, 0.1, size=n)
            perturbed = scores + noise
            pert_selected, pert_count = _greedy_decode(adj, perturbed)
            if pert_count > best_count:
                best_count = pert_count
                selected = pert_selected
                scores = perturbed
            # Continue searching
            continue
    
    return scores
