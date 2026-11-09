"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assigns a score to each vertex for the greedy MIS decoder.
    Higher scores are picked first.
    Strategy: Combine local degree cost with global spectral position.
    1. Local Cost: Degree + Sum of Neighbor Degrees. Lower is better.
    2. Spectral Signal: Absolute value of the Fiedler vector (eigenvector 
       of the Laplacian corresponding to the second smallest eigenvalue).
       Higher absolute value indicates the vertex is further from the 
       graph's sparsest cut, which often corresponds to 'safer' picks 
       with fewer cross-cut conflicts.
    """
    A = np.array(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Compute Local Features
    degrees = A.sum(axis=1)
    # Sum of degrees of neighbors
    neighbor_deg_sum = A @ degrees
    # Combined local cost: we want to minimize this
    local_cost = degrees + neighbor_deg_sum
    
    # Normalize local cost to [0, 1] range (inverted so higher is better)
    min_lc = local_cost.min()
    max_lc = local_cost.max()
    if max_lc > min_lc:
        local_cost_norm = (local_cost - min_lc) / (max_lc - min_lc)
    else:
        local_cost_norm = np.zeros(n)
    
    # Base score: negative normalized cost
    scores = -local_cost_norm
    
    # 2. Compute Spectral Features
    # Construct Laplacian L = D - A
    D = np.diag(degrees)
    L = D - A
    
    try:
        # Compute eigenvalues and eigenvectors
        # eigh is optimized for symmetric/Hermitian matrices
        evals, evecs = np.linalg.eigh(L)
        
        # Check if the graph is effectively connected
        # If the second smallest eigenvalue is very close to 0, the graph is disconnected.
        # In that case, the Fiedler vector is not a meaningful 'cut' indicator for a single component.
        if evals[1] > 1e-6:
            fiedler = evecs[:, 1]
            # Use absolute value as a measure of 'distance' from the cut (where value is ~0)
            spectral_signal = np.abs(fiedler)
            
            # Normalize spectral signal to [0, 1]
            min_ss = spectral_signal.min()
            max_ss = spectral_signal.max()
            if max_ss > min_ss:
                spectral_norm = (spectral_signal - min_ss) / (max_ss - min_ss)
            else:
                spectral_norm = np.zeros(n)
            
            # Combine scores
            # We want to maximize spectral_norm (safe vertices)
            # We already have -local_cost_norm (cheap vertices)
            # Weight the spectral component
            alpha = 0.5
            scores = scores + alpha * spectral_norm
            
    except Exception:
        # If spectral computation fails for any reason, fall back to local cost only
        pass
        
    return scores
# EVOLVE-BLOCK-END
