import numpy as np
import time

def _greedy_size(adjacency, order):
    """Simulate greedy MIS with given vertex order, return set size."""
    n = adjacency.shape[0]
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            nbrs = np.where(adjacency[v])[0]
            available[nbrs] = False
    return count

def _evaluate_order(adjacency, base_order, start_time, deadline, max_evals=50):
    """Evaluate base_order and perturbed versions, return best order."""
    n = adjacency.shape[0]
    best_size = _greedy_size(adjacency, base_order)
    best_order = base_order.copy()
    
    # Try swapping pairs to improve
    for _ in range(max_evals):
        if time.time() > deadline - 30.0:
            break
        
        # Randomly pick two positions to swap
        i, j = np.random.choice(n, size=2, replace=False)
        trial_order = best_order.copy()
        trial_order[i], trial_order[j] = trial_order[j], trial_order[i]
        
        trial_size = _greedy_size(adjacency, trial_order)
        if trial_size > best_size:
            best_size = trial_size
            best_order = trial_order
    
    return best_order

def _power_iteration_eigenvector(matrix, n_iter=50, tol=1e-8):
    """Compute principal eigenvector via power iteration."""
    n = matrix.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    
    for _ in range(n_iter):
        v_new = matrix @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-10:
            break
        v_new /= norm
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    return v

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s limit
    
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1)
    
    # Mean neighbor degree
    neighbor_deg_sum = adjacency @ deg
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Spectral component: use adjacency matrix eigenvector
    # For ER graphs, the principal eigenvector gives a good initial ordering
    eigvec = _power_iteration_eigenvector(adjacency.astype(float), n_iter=30)
    
    # Combine degree-based and spectral scores
    # Lower degree should be preferred, but spectral info also matters
    deg_norm = (deg - deg.min()) / (deg.max() - deg.min() + 1e-10)
    eig_norm = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min() + 1e-10)
    
    # Base score: prefer low degree and high spectral value
    base_score = -deg_norm * 0.7 + eig_norm * 0.3 + 0.1 * mean_neighbor_deg / (deg + 1e-10)
    
    # Initial order by base score
    base_order = np.argsort(-base_score)
    
    # Evaluate and improve the order
    if time.time() < deadline - 40.0:
        best_order = _evaluate_order(adjacency, base_order, start_time, deadline, max_evals=100)
    else:
        best_order = base_order
    
    # Compute final scores based on position in best order
    # Earlier in order = higher score
    pos = np.zeros(n, dtype=int)
    pos[best_order] = np.arange(n)
    
    # Score: n - position (so earlier positions get higher scores)
    final_score = (n - pos).astype(float)
    
    return final_score
