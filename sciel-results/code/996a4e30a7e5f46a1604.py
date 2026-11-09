"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_decode(adjacency, scores):
    """
    Simulates the greedy decoder to evaluate a score vector.
    Returns the size of the independent set.
    """
    n = len(scores)
    order = np.argsort(-scores)  # Descending scores
    selected = np.zeros(n, dtype=bool)
    size = 0
    for v in order:
        if not selected[v]:
            selected[v] = True
            size += 1
            # Mark neighbors as selected (blocked)
            neighbors = np.where(adjacency[v])[0]
            selected[neighbors] = True
    return size

def priority(adjacency):
    """
    Computes priority scores for vertices.
    Strategy: Use the Fiedler vector (spectral gap) to identify a balanced bipartition.
    The vertices on one side of the partition (or both, depending on sign) form a large independent set.
    We return the values of the Fiedler vector as priorities.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Compute Laplacian L = D - A
    degrees = adjacency.sum(axis=1)
    # L = diag(degrees) - adjacency
    # We can compute eigenvalues/vectors of L directly or use the normalized Laplacian.
    # Standard Laplacian is sufficient for Fiedler vector.
    L = np.diag(degrees) - adjacency
    
    try:
        # Compute eigenvalues and eigenvectors
        # For n=200, this is very fast.
        evals, evecs = np.linalg.eigh(L)
        
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue.
        # eigh returns eigenvalues in ascending order.
        # Index 0 is usually the trivial eigenvector (all ones) for connected graphs.
        # If the graph is disconnected, there might be multiple zero eigenvalues.
        # The Fiedler vector is typically the one with the largest absolute values that is not constant.
        # Standard practice: take evecs[:, 1].
        fiedler_vec = evecs[:, 1]
        
        # Check time
        if time.time() > deadline:
             return -degrees.astype(float)
             
        # The sign of the Fiedler vector is arbitrary.
        # We want to pick the larger side of the bipartition or the side that yields a better independent set.
        # Actually, the greedy decoder will handle the selection.
        # The scores should reflect the "quality" of picking a vertex.
        # If we use the Fiedler vector directly, vertices with high positive values are in one partition,
        # and high negative values in the other.
        # Since the greedy decoder picks in descending order, it will pick the most positive vertices first.
        # This effectively picks one side of the bipartition.
        # To be robust, we can center the vector or just use it as is.
        # Using it as is, the greedy decoder will pick the "top" half of the spectrum.
        
        return fiedler_vec.astype(float)
        
    except Exception:
        # Fallback to degree-based if spectral fails
        return -degrees.astype(float)

# EVOLVE-BLOCK-END
