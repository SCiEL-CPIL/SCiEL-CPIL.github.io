"""Rewritten program using spectral centrality for Maximum Independent Set greedy decoder."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign scores to vertices for greedy MIS selection.
    Uses a combination of spectral centrality (principal eigenvector of adjacency)
    and degree to prioritize structurally peripheral, low-degree nodes.
    Lower score means higher priority (selected earlier in descending order of -score,
    but the decoder selects in descending score order, so we return negative priority).
    
    Wait: the decoder picks vertices in DESCENDING score order.
    So higher score = selected first.
    We want to select peripheral, low-degree nodes first.
    So we want LOW eigenvector centrality and LOW degree to have HIGH score.
    Thus: score = - (alpha * eigenvector_centrality + beta * degree)
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Degree
    degree = A.sum(axis=1)
    
    # Power iteration for principal eigenvector
    # Initialize with ones
    x = np.ones(n, dtype=float)
    x /= np.linalg.norm(x) + 1e-12
    
    max_iter = 1000
    tol = 1e-8
    prev_norm = 0.0
    
    for i in range(max_iter):
        # Check time
        if time.time() > deadline - 10.0:
            break
        
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            x_new = np.ones(n, dtype=float)
            norm = 1.0
        x_new /= norm
        
        # Check convergence
        diff = np.linalg.norm(x_new - x)
        x = x_new
        if diff < tol:
            break
    
    # Eigenvector centrality (principal component, all non-negative for ER graphs)
    # Ensure non-negative by taking absolute value or clipping
    eig_cent = np.abs(x)
    
    # Normalize both degree and eig_cent to [0, 1] for combination
    deg_min, deg_max = degree.min(), degree.max()
    deg_range = deg_max - deg_min + 1e-12
    deg_norm = (degree - deg_min) / deg_range
    
    eig_min, eig_max = eig_cent.min(), eig_cent.max()
    eig_range = eig_max - eig_min + 1e-12
    eig_norm = (eig_cent - eig_min) / eig_range
    
    # Combine: we want low degree and low eigenvector centrality to have high score
    # score = - (w1 * deg_norm + w2 * eig_norm)
    # Try a weighted combination
    w1 = 0.4  # weight for degree
    w2 = 0.6  # weight for eigenvector centrality
    
    combined = w1 * deg_norm + w2 * eig_norm
    score = -combined
    
    return score.astype(float)
# EVOLVE-BLOCK-END
