"""Rewritten program using spectral gap heuristic for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def priority(adjacency):
    """
    Assign scores to vertices using a spectral gap heuristic.
    
    The idea:
    1. Compute the graph Laplacian L = D - A.
    2. Find the Fiedler vector (eigenvector of the second-smallest eigenvalue).
    3. Use the Fiedler vector to partition vertices into two sets (positive/negative).
    4. Vertices in the smaller partition are prioritized higher (assigned higher scores)
       because selecting from the smaller side of the sparsest cut minimizes edge conflicts
       and helps isolate a large independent set.
    5. Within each partition, break ties by degree (lower degree first) to further reduce conflicts.
    
    Returns:
        A numpy array of scores, one per vertex. Higher score = selected earlier.
    """
    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=float)
    
    # Ensure symmetry
    A = (A + A.T) / 2.0
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Handle case where graph is empty or trivial
    if n == 0:
        return np.array([], dtype=float)
    
    # Build Laplacian
    D = np.diag(degrees)
    L = D - A
    
    # For efficiency with larger graphs, use sparse representation
    L_sparse = sp.csr_matrix(L)
    
    # We want the second-smallest eigenvalue and its eigenvector (Fiedler vector)
    # Use shift-invert mode to find eigenvalues near 0
    try:
        # Find eigenvalues near 0 (smallest ones)
        # We need at least 2 eigenvalues; the smallest is 0 (for connected components)
        k = min(2, n)
        if n > 1:
            vals, vecs = spla.eigsh(L_sparse, k=k, sigma=0, which='LM')
        else:
            vals = np.array([0.0])
            vecs = np.array([[1.0]])
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
        if len(vals) >= 2:
            fiedler = vecs[:, 1]
        else:
            # Fallback: use degree-based scoring
            fiedler = -degrees
        
        # Normalize Fiedler vector
        fiedler_norm = np.linalg.norm(fiedler)
        if fiedler_norm > 1e-10:
            fiedler = fiedler / fiedler_norm
        
        # Partition vertices based on sign of Fiedler vector
        # Vertices with negative Fiedler values are in one partition, positive in another
        # We want to prioritize vertices in the smaller partition
        neg_mask = fiedler < 0
        pos_mask = fiedler >= 0
        
        neg_count = np.sum(neg_mask)
        pos_count = np.sum(pos_mask)
        
        # The smaller partition gets higher priority
        if neg_count <= pos_count:
            # Negative side is smaller or equal; prioritize negative side
            base_score = np.where(neg_mask, 1.0, 0.0)
        else:
            # Positive side is smaller; prioritize positive side
            base_score = np.where(pos_mask, 1.0, 0.0)
        
        # Within each partition, use negative degree as secondary score
        # Lower degree = higher score (less likely to conflict)
        degree_score = -degrees / (degrees.max() + 1e-10)
        
        # Combine: base_score (0 or 1) + small weight * degree_score
        # This ensures the partition choice dominates, but degree breaks ties
        weight = 0.1
        scores = base_score + weight * degree_score
        
        # Add a tiny random perturbation to break any remaining ties deterministically
        # Actually, let's avoid randomness for reproducibility; use index-based tie-breaking
        # Instead, just return the scores as is; the greedy decoder will handle ties
        
    except Exception:
        # Fallback to simple degree-based scoring if spectral computation fails
        scores = -degrees.astype(float)
    
    return scores
# EVOLVE-BLOCK-END
