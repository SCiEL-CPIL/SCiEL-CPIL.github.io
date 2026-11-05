"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a priority score to each vertex for the greedy independent-set decoder.
    Higher score means the vertex is considered earlier.
    
    Strategy: Use a spectral/peripheral heuristic combined with local structure.
    1. Compute the Laplacian L = D - A.
    2. Find the Fiedler vector (eigenvector for 2nd smallest eigenvalue).
       Vertices with large |Fiedler component| are at the 'periphery' of the
       main component, which often makes them safer to include (less central
       to connectivity).
    3. Combine with degree: lower degree is generally better for independent set
       (fewer conflicts). We use a weighted combination.
    
    To keep runtime bounded, we use a randomized power iteration or numpy's
    eigh on the small 200x200 matrix (which is very fast).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Degree vector
    deg = A.sum(axis=1)
    
    # Laplacian L = D - A
    D = np.diag(deg)
    L = D - A
    
    # We want the Fiedler vector. For a connected graph, this is the eigenvector
    # corresponding to the 2nd smallest eigenvalue.
    # For 200x200, np.linalg.eigh is fast enough.
    try:
        if time.time() > deadline:
            # Fallback: just use negative degree
            return -deg.astype(float)
        
        # Symmetrize to ensure numerical stability
        L = (L + L.T) / 2.0
        
        # Compute eigenvalues and eigenvectors
        # We only need the first few eigenvectors.
        # numpy.linalg.eigh computes all, but for n=200 it's ~ms.
        evals, evecs = np.linalg.eigh(L)
        
        # The smallest eigenvalue is 0 (for connected graph) with eigenvector all-ones.
        # The 2nd smallest is the Fiedler value.
        # If the graph is disconnected, there might be multiple zero eigenvalues.
        # We want the eigenvector corresponding to the smallest *non-zero* eigenvalue,
        # or generally the 2nd smallest if we assume connected.
        # Let's find the index of the first eigenvalue > 1e-8.
        nonzero_mask = evals > 1e-8
        if np.any(nonzero_mask):
            fiedler_idx = np.where(nonzero_mask)[0][0]
        else:
            # Graph is empty or single node?
            fiedler_idx = 0 if n > 0 else 0
            
        fiedler_vec = evecs[:, fiedler_idx]
        
    except Exception:
        # Fallback if eigen decomposition fails
        return -deg.astype(float)
    
    if time.time() > deadline:
        return -deg.astype(float)
    
    # Normalize the Fiedler vector components to [0, 1] range for easier combination
    # We use the absolute value, as the sign is arbitrary.
    fiedler_mag = np.abs(fiedler_vec)
    
    # Normalize fiedler_mag to [0, 1]
    f_min = fiedler_mag.min()
    f_max = fiedler_mag.max()
    if f_max - f_min > 1e-12:
        fiedler_norm = (fiedler_mag - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros_like(fiedler_mag)
    
    # Normalize degree to [0, 1]
    d_min = deg.min()
    d_max = deg.max()
    if d_max - d_min > 1e-12:
        deg_norm = (deg - d_min) / (d_max - d_min)
    else:
        deg_norm = np.zeros_like(deg)
    
    # We want:
    # 1. High fiedler_norm (peripheral) -> Good
    # 2. Low degree (fewer conflicts) -> Good, so we use (1 - deg_norm)
    
    # Weighted combination. 
    # Peripheral vertices are often low-degree in ER graphs? Not necessarily.
    # Let's try a simple sum.
    # Score = alpha * fiedler_norm + (1 - alpha) * (1 - deg_norm)
    # Since we want higher score for earlier selection:
    # High fiedler_norm is good.
    # Low degree is good -> (1 - deg_norm) is high when degree is low.
    
    alpha = 0.5
    scores = alpha * fiedler_norm + (1 - alpha) * (1 - deg_norm)
    
    # Add a small random tie-breaker to avoid symmetry issues? 
    # The task says "fixed greedy decoder", implying deterministic or seeded.
    # But adding a tiny deterministic noise based on index can help break ties.
    # Let's add a very small index-based term to ensure unique ordering.
    scores += np.arange(n) * 1e-10
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
