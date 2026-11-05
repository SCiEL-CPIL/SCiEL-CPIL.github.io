"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign scores to vertices for greedy independent set selection.
    
    Strategy: Model vertex priority via a smooth relaxation of the LP dual
    for maximum independent set / weighted vertex cover. We iteratively
    adjust weights to approximate the dual variables, which identify vertices
    with high 'structural independence' potential.
    
    The idea:
    - Start with uniform weights.
    - Iteratively update weights based on local density and independence potential.
    - Use a smooth surrogate (log-sum-exp style) to avoid discontinuities.
    - Add a deadline guard to ensure we finish within time limits.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Initial weights: uniform
    weights = np.ones(n)
    
    # Degrees
    degrees = A.sum(axis=1)
    
    # Number of iterations based on available time
    # Estimate cost per iteration and cap accordingly
    max_iter = 200
    min_iter = 10
    
    # Precompute some useful quantities
    # For each vertex, compute the "independence potential"
    # A vertex with few neighbors and neighbors that are themselves well-connected
    # is a good candidate for independent set.
    
    scores = weights.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute local density: how "clustered" the neighborhood is
        # For each vertex i, look at its neighbors' degrees
        # High neighbor degree means neighbors are "busy", so i is more independent
        
        # Neighbor average degree
        # For vertex i, mean degree of its neighbors
        # This is (A @ degrees) / (degrees + eps)
        eps = 1e-12
        neighbor_avg_deg = (A @ degrees) / (degrees + eps)
        
        # Independence potential: vertices with low degree and neighbors with high degree
        # are good candidates (they don't conflict much, and their neighbors conflict with others)
        # Smooth score: combine degree (low is good) and neighbor density (high is good)
        
        # Normalize for stability
        deg_norm = degrees / (degrees.max() + eps)
        nad_norm = neighbor_avg_deg / (neighbor_avg_deg.max() + eps)
        
        # Score: prefer low degree, high neighbor density
        # Use a smooth combination
        local_score = (1.0 - deg_norm) + 0.5 * nad_norm
        
        # Update weights with momentum
        alpha = 0.3  # learning rate
        weights = (1.0 - alpha) * weights + alpha * local_score
        
        # Optional: re-normalize to prevent drift
        # weights = weights / (weights.max() + eps)
        
        scores = weights.copy()
    
    # Final polish: add a small random tie-breaker to avoid degeneracy
    # Use deterministic tie-breaking based on vertex index to ensure reproducibility
    # Actually, for the greedy decoder, we just need distinct scores ideally
    # Add tiny index-based perturbation
    idx = np.arange(n)
    scores = scores + 1e-10 * idx
    
    return scores
# EVOLVE-BLOCK-END
