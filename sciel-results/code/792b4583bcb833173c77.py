"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def _log_energy_batch(points):
    """
    Compute logarithmic energy for a batch of configurations.
    points: array of shape (M, N, 3)
    returns: array of shape (M,) with energy values
    """
    # Compute pairwise distances
    # points shape: (M, N, 3)
    # We want (M, N, N)
    # diff = points[:, np.newaxis, :, :] - points[:, :, np.newaxis, :]
    # This is (M, N, 1, 3) - (M, 1, N, 3) -> (M, N, N, 3)
    # dist = np.linalg.norm(diff, axis=-1)
    
    # More memory efficient:
    # sq_norms = np.sum(points**2, axis=2) # (M, N)
    # dots = np.matmul(points, points.transpose(0, 2, 1)) # (M, N, N)
    # dist_sq = sq_norms[:, np.newaxis, :] + sq_norms[:, :, np.newaxis] - 2 * dots
    # np.maximum(dist_sq, 0) to handle numerical errors
    # dist = np.sqrt(dist_sq)
    
    M = points.shape[0]
    # Use the dot product method for speed
    sq_norms = np.sum(points**2, axis=2)
    dots = np.matmul(points, points.transpose(0, 2, 1))
    dist_sq = sq_norms[:, np.newaxis, :] + sq_norms[:, :, np.newaxis] - 2 * dots
    np.maximum(dist_sq, 0, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(N, dtype=bool)
    # We sum over all i,j. Since dist is symmetric, sum is 2 * sum_{i<j}.
    # Energy = sum_{i<j} log(1/d_ij) = 0.5 * sum_{i!=j} log(1/d_ij)
    
    # Add epsilon to avoid log(0) if any points are identical (shouldn't happen often with random init)
    dist_safe = dist + 1e-12
    
    log_terms = np.log(1.0 / dist_safe)
    
    # Set diagonal to 0
    log_terms[:, mask] = log_terms[:, mask] # just ensuring shape
    # Actually, easier to just zero out the diagonal in log_terms
    idx = np.arange(N)
    log_terms[:, idx, idx] = 0.0
    
    energies = np.sum(log_terms, axis=(1, 2)) * 0.5
    return energies

def construct():
    np.random.seed(42)
    
    # Parameters
    M = 30 # Number of particles
    iterations = 100
    
    # Initialize particles
    # Use random points on sphere
    # Or mix of Fibonacci and random?
    # Let's start with diverse random points.
    
    # Generate M sets of N points
    # For each particle, generate N random points on sphere
    # x = random.normal(0, 1, (M, N, 3))
    # x = x / norm(x)
    
    X = np.random.normal(0, 1, size=(M, N, 3))
    norms = np.linalg.norm(X, axis=2, keepdims=True)
    X = X / norms
    
    # Initialize velocities to 0
    V = np.zeros((M, N, 3))
    
    # Evaluate initial energies
    E = _log_energy_batch(X)
    
    # Personal bests
    P_best = X.copy()
    P_best_E = E.copy()
    
    # Global best
    g_best_idx = np.argmin(E)
    G_best = X[g_best_idx].copy()
    G_best_E = E[g_best_idx]
    
    # PSO Parameters
    w = 0.8 # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    for it in range(iterations):
        # Update learning rates? Maybe decrease w slightly
        # w = 0.9 - 0.4 * (it / iterations)
        
        # Random factors
        r1 = np.random.rand(M, N, 1)
        r2 = np.random.rand(M, N, 1)
        
        # Velocity update
        # V = w*V + c1*r1*(P_best - X) + c2*r2*(G_best - X)
        # G_best is (N, 3), needs broadcasting to (M, N, 3)
        V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best[np.newaxis, :, :] - X)
        
        # Position update
        X_new = X + V
        
        # Project to sphere
        norms = np.linalg.norm(X_new, axis=2, keepdims=True)
        norms[norms < 1e-15] = 1.0
        X_new = X_new / norms
        
        # Evaluate new energies
        E_new = _log_energy_batch(X_new)
        
        # Update personal bests
        improved = E_new < P_best_E
        # Use np.where to update
        P_best_E = np.where(improved, E_new, P_best_E)
        P_best = np.where(improved[:, np.newaxis, np.newaxis], X_new, P_best)
        
        # Update global best
        current_global_idx = np.argmin(E_new)
        if E_new[current_global_idx] < G_best_E:
            G_best_E = E_new[current_global_idx]
            G_best = X_new[current_global_idx].copy()
            
        # Update current state
        X = X_new
        E = E_new
        
    # Return the global best
    # Ensure it is a single array of shape (N, 3)
    return G_best
# EVOLVE-BLOCK-END
