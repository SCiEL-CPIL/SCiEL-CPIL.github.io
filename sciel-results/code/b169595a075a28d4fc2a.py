"""Spectral initialization + greedy local search for Maximum Cut."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh


def partition(adjacency):
    """
    Returns a 0/1 label array for each vertex, maximizing the number of edges
    crossing the partition.
    
    Strategy:
    1. Build the graph Laplacian.
    2. Compute the second smallest eigenvector (Fiedler vector) of the Laplacian.
    3. Initialize partition by thresholding the Fiedler vector at its median.
    4. Run multiple passes of greedy local search (flip vertex if it improves cut).
    5. Return the final labels.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Convert adjacency to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Ensure it's a boolean/int array for proper handling
    A = adjacency.astype(float)
    
    # Build Laplacian: L = D - A
    # D is the degree matrix
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    L = D - A
    
    # Compute the second smallest eigenvalue and eigenvector of L
    # Use eigsh for sparse eigenvalue problem
    # We want the smallest eigenvalues, so use which='SM'
    # For a connected graph, the smallest eigenvalue is 0 with eigenvector all ones.
    # The second smallest is what we want (Fiedler vector).
    
    try:
        # k=2 to get the two smallest eigenvalues
        # sigma=0 to shift for stability, or use which='SM'
        # eigsh requires the matrix to be symmetric
        k = min(2, n - 1)
        if k < 1:
            # Fallback to random
            labels = np.random.randint(0, 2, n)
        else:
            # Use eigsh to find the k smallest eigenvalues
            # For small n, eigsh might have issues, so use eigh for very small graphs
            if n <= 100:
                # Use dense eigenvalue decomposition for small graphs
                eigvals, eigvecs = np.linalg.eigh(L)
                # Sort by eigenvalue (eigh returns in ascending order)
                # The first eigenvector corresponds to the smallest eigenvalue (0)
                # The second corresponds to the Fiedler vector
                if len(eigvecs) > 1:
                    fiedler = eigvecs[:, 1]
                else:
                    fiedler = eigvecs[:, 0]
            else:
                # Use sparse eigensolver
                # Convert to sparse
                from scipy.sparse import csr_matrix
                L_sparse = csr_matrix(L)
                eigvals, eigvecs = eigsh(L_sparse, k=k, which='SM')
                # eigsh returns eigenvalues in arbitrary order, sort them
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                fiedler = eigvecs[:, 1]
            
            # Initialize labels based on the Fiedler vector
            # Split at the median
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(int)
            
            # Ensure both partitions are non-empty if possible
            if labels.sum() == 0 or labels.sum() == n:
                # Fallback to alternating
                labels = np.arange(n) % 2
    
    except Exception:
        # Fallback to alternating initialization
        labels = np.arange(n) % 2
    
    # Greedy local search: multiple passes
    # In each pass, for each vertex, check if flipping improves the cut
    # Cut size = sum over edges (u,v) of [label[u] != label[v]]
    
    # Precompute row sums for efficiency
    # For vertex v, the number of edges to same partition and different partition
    # same[v] = number of neighbors with same label
    # diff[v] = number of neighbors with different label
    # Flipping v changes the cut by: if same > diff, flipping increases cut by (same - diff)
    # Actually: current contribution of v to cut = diff[v]
    # After flip: contribution = same[v]
    # Change = same[v] - diff[v]
    # If same[v] > diff[v], flipping improves the cut
    
    A_bool = adjacency.astype(bool)
    
    # Run multiple passes until no improvement
    for _ in range(10):
        improved = False
        for v in range(n):
            # Count neighbors with same and different labels
            neighbors = A_bool[v]
            if not neighbors.any():
                continue
            
            same_count = (labels[neighbors] == labels[v]).sum()
            diff_count = (labels[neighbors] != labels[v]).sum()
            
            if same_count > diff_count:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels
# EVOLVE-BLOCK-END
