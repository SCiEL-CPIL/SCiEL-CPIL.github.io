"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 100s for safety
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Try spectral approach if time permits
    try:
        # Compute Laplacian L = D - A
        D = np.diag(degrees)
        L = D - adj
        
        # Get Fiedler vector (second smallest eigenvector)
        # Use scipy for efficient eigenvalue computation
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        L_sparse = csr_matrix(L)
        
        # Find the two smallest eigenvalues
        # For a connected graph, lambda_0 = 0, lambda_1 > 0 (Fiedler value)
        # We want the eigenvector for lambda_1
        
        # Use shift-invert mode for smallest eigenvalues
        vals, vecs = eigsh(L_sparse, k=2, sigma=0.0, which='LM')
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Fiedler vector is the second one
        fiedler = vecs[:, 1]
        
        # Normalize
        fiedler_norm = np.linalg.norm(fiedler)
        if fiedler_norm > 1e-10:
            fiedler = fiedler / fiedler_norm
        
        # Score based on:
        # 1. Low degree (primary)
        # 2. Distance from the "bottleneck" axis (Fiedler vector)
        #    Vertices far from zero in Fiedler space are on the "periphery"
        #    and likely in less constrained regions
        
        # Combine: low degree + high |Fiedler component|
        # The idea: vertices in sparse regions (low degree) that are also
        # far from the connectivity bottleneck (high |Fiedler|) are good candidates
        
        # Scale factors
        degree_scale = np.std(degrees) if np.std(degrees) > 1e-10 else 1.0
        fiedler_scale = np.std(fiedler) if np.std(fiedler) > 1e-10 else 1.0
        
        # Normalize degree to [0,1]
        deg_min, deg_max = np.min(degrees), np.max(degrees)
        if deg_max - deg_min > 1e-10:
            deg_norm = (degrees - deg_min) / (deg_max - deg_min)
        else:
            deg_norm = np.zeros(n)
        
        # Use |Fiedler| as a measure of distance from bottleneck
        fiedler_abs = np.abs(fiedler)
        fiedler_min, fiedler_max = np.min(fiedler_abs), np.max(fiedler_abs)
        if fiedler_max - fiedler_min > 1e-10:
            fiedler_norm_scaled = (fiedler_abs - fiedler_min) / (fiedler_max - fiedler_min)
        else:
            fiedler_norm_scaled = np.zeros(n)
        
        # Score: want low degree (negative) and high Fiedler distance (positive)
        # Weight: degree is more important, but Fiedler helps break ties
        score = -deg_norm + 0.3 * fiedler_norm_scaled
        
        # Check time
        if time.time() > deadline - 10:
            # Not enough time, fall back to simple degree
            return -degrees.astype(float)
        
        return score.astype(float)
    
    except Exception:
        # Fall back to degree-based scoring
        return -degrees.astype(float)
# EVOLVE-BLOCK-END
