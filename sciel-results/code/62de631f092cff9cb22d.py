"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_is_size(adjacency, scores):
    """Simulate the greedy decoder: pick vertices in descending score order, skip neighbors."""
    n = len(scores)
    order = np.argsort(-scores)  # descending score
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not selected[v] and not blocked[v]:
            selected[v] = True
            count += 1
            # Block neighbors
            nbrs = np.where(adjacency[v])[0]
            blocked[nbrs] = True
    return count


def priority(adjacency):
    """Return a score per vertex for the greedy independent set decoder."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Compute neighbor average degree for each vertex
    # For vertex v, mean_deg_nbrs[v] = sum(degrees[u] for u in neighbors(v)) / degree(v)
    # Handle isolated vertices
    sum_nbr_deg = (adjacency @ degrees).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_nbr_deg = np.where(degrees > 0, sum_nbr_deg / degrees, 0.0)
    
    # Also compute: for each vertex, the minimum degree among its neighbors
    # This captures: if all neighbors are high-degree, v is safe to pick
    min_nbr_deg = np.full(n, np.inf)
    for v in range(n):
        nbrs = np.where(adjacency[v])[0]
        if len(nbrs) > 0:
            min_nbr_deg[v] = degrees[nbrs].min()
        else:
            min_nbr_deg[v] = 0.0
    
    # Try several scoring formulas and pick the best via self-evaluation
    candidates = []
    
    # Candidate 1: -degree
    candidates.append(-degrees)
    
    # Candidate 2: -degree + alpha * mean_nbr_deg
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        candidates.append(-degrees + alpha * mean_nbr_deg)
    
    # Candidate 3: -degree + alpha * min_nbr_deg
    for alpha in [0.1, 0.3, 0.5]:
        candidates.append(-degrees + alpha * min_nbr_deg)
    
    # Candidate 4: -degree * (1 + beta * mean_nbr_deg / (mean_degree + 1e-9))
    mean_deg = degrees.mean() + 1e-9
    for beta in [0.5, 1.0, 2.0]:
        candidates.append(-degrees * (1.0 + beta * mean_nbr_deg / mean_deg))
    
    # Candidate 5: -degree + alpha * (mean_nbr_deg - mean_deg)  (centered)
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        candidates.append(-degrees + alpha * (mean_nbr_deg - mean_deg))
    
    # Candidate 6: -degree * degree (quadratic penalty)
    candidates.append(-degrees * degrees)
    
    # Candidate 7: -degree + alpha * (sum_nbr_deg / (n - 1))
    for alpha in [0.1, 0.3, 0.5]:
        candidates.append(-degrees + alpha * sum_nbr_deg / max(n - 1, 1))
    
    # Evaluate all candidates and pick the best
    best_score = -1
    best_candidate = candidates[0]
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        size = _greedy_is_size(adjacency, cand)
        if size > best_score:
            best_score = size
            best_candidate = cand
    
    return best_candidate
# EVOLVE-BLOCK-END
