# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_cut


def partition(adjacency):
    # Use scipy's maximum_cut (spectral bisection) for a strong initial partition,
    # then perform several passes of local improvement (flip vertices if it increases cut size).
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert adjacency to a sparse matrix for scipy
    A = csr_matrix(np.asarray(adjacency, dtype=float))
    
    # Get initial partition from spectral maximum cut
    try:
        labels = maximum_cut(A, return_max=True)[0]
        # maximum_cut returns 0/1 labels
        labels = np.asarray(labels, dtype=int)
    except Exception:
        # Fallback to random initialization
        labels = np.random.randint(0, 2, size=n)
    
    # Local improvement: flip vertices if it increases the cut size
    # We do multiple passes until no improvement or max iterations
    for _ in range(10):
        improved = False
        for v in range(n):
            # Count edges to same side vs other side
            same_side = (labels == labels[v])
            other_side = (labels != labels[v])
            # Number of neighbors on same side
            same_count = np.sum(adjacency[v][same_side])
            # Number of neighbors on other side
            other_count = np.sum(adjacency[v][other_side])
            
            # If flipping v increases the cut, flip it
            # Currently, edges from v to other_side are in the cut.
            # After flipping, edges from v to same_side (excluding v itself, but v not connected to itself typically) would be in the cut.
            # Actually, we need to be careful: the cut size contribution from v is the number of neighbors on the opposite side.
            # If we flip v, the new contribution is the number of neighbors on the current same side.
            # So we flip if same_count > other_count.
            if same_count > other_count:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels
# EVOLVE-BLOCK-END
