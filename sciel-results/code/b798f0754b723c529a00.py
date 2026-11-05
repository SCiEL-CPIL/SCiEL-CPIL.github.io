"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solves Maximum Cut using spectral initialization and vectorized local search.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Ensure we are working with a numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float32)
    
    # Deadline guard: 150s limit, keep 5s margin
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Spectral Initialization
    # The second eigenvector of the Laplacian L = D - A is a good heuristic for Max Cut.
    # L = D - A. We can compute eigenvectors of L.
    # For sparse graphs, scipy is better, but n=100 is small enough for dense numpy.
    # However, scipy is available and likely faster/more stable for eig.
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        # Convert to sparse for efficiency if possible, though n=100 is trivial
        # But let's stick to numpy dense if scipy import fails or for simplicity, 
        # but scipy is allowed and often faster for eig.
        # Actually, for n=100, np.linalg.eigh is very fast.
        # Let's use np.linalg.eigh for simplicity and robustness.
        
        # Degree matrix
        degrees = adj.sum(axis=1)
        L = np.diag(degrees) - adj
        
        # Compute eigenvalues and eigenvectors
        # We want the eigenvector corresponding to the 2nd smallest eigenvalue (index 1)
        # eigh returns eigenvalues in ascending order.
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # The eigenvector corresponding to eigvals[1]
        # It can be positive or negative, so we take absolute values or just use signs.
        # Standard approach: threshold at 0.
        v = eigvecs[:, 1]
        
        # If the vector is all zeros (unlikely for connected graph, possible for disconnected), 
        # fallback to random.
        if np.allclose(v, 0):
             labels = np.random.randint(0, 2, n).astype(np.int8)
        else:
             # Threshold at 0
             labels = (v > 0).astype(np.int8)
             
             # Ensure we have both labels if possible (avoid trivial solution)
             # If all 0 or all 1, flip half randomly
             if labels.sum() == 0 or labels.sum() == n:
                 labels = np.random.randint(0, 2, n).astype(np.int8)
    except Exception:
        # Fallback to alternating if spectral fails
        labels = np.arange(n) % 2
        labels = labels.astype(np.int8)

    # 2. Vectorized Local Search
    # We iterate until no more improvements or deadline.
    # To make it fast, we calculate gains for ALL vertices at once.
    
    # Precompute row sums (degrees)
    degs = adj.sum(axis=1)
    
    # Function to calculate cut size
    def get_cut_size(labs):
        # Cut size = sum of adj[i,j] where labs[i] != labs[j]
        # This can be computed as:
        # sum(adj) / 2 is total edges.
        # Non-cut edges are those where labs[i] == labs[j].
        # Let's use a fast method:
        # cut = 0.5 * (sum(adj) - sum(adj * (labs[:, None] == labs[None, :])))
        # Or simpler:
        # For each node, count neighbors in same partition.
        # same_neighbors = (adj * (labels[None, :] == labels[:, None])).sum(axis=1)
        # cut_size = (degs - same_neighbors).sum() / 2
        
        # Even faster for small n:
        # diff = (labels[None, :] != labels[:, None])
        # return np.sum(adj * diff)
        
        # Let's use the matrix multiplication trick if possible, or just boolean mask.
        # For n=100, boolean mask is fine.
        diff = (labels[None, :] != labels[:, None])
        return np.sum(adj * diff)

    # Actually, let's optimize the gain calculation.
    # Gain for flipping node v:
    # current contribution of v to cut: edges from v to opposite side
    # new contribution: edges from v to same side (which become opposite)
    # Gain = (edges to same) - (edges to opp)
    # Let s_v = sum of adj[v, labels == labels[v]]
    # Let o_v = sum of adj[v, labels != labels[v]]
    # deg_v = s_v + o_v
    # Gain = s_v - o_v = s_v - (deg_v - s_v) = 2*s_v - deg_v
    # We want to flip if Gain > 0.
    
    best_cut = get_cut_size(labels)
    
    while True:
        if time.time() > deadline:
            break
            
        # Calculate gains for all vertices simultaneously
        # labels is 1D array of 0/1.
        # same_mask[i, j] is True if labels[i] == labels[j]
        same_mask = (labels[None, :] == labels[:, None])
        
        # s_v for each v is sum of adj[v, :] where same_mask[v, :] is True
        # s = (adj * same_mask).sum(axis=1)
        
        s = (adj * same_mask).sum(axis=1)
        gains = 2 * s - degs
        
        # Find vertices with positive gain
        positive_mask = gains > 0
        if not np.any(positive_mask):
            break
            
        # Flip all vertices with positive gain simultaneously?
        # This is a "parallel" local search step. It can oscillate, but often converges quickly.
        # Alternatively, flip the single best one. But parallel is faster per iteration.
        # Let's try flipping all positive ones.
        labels[positive_mask] = 1 - labels[positive_mask]
        
        new_cut = get_cut_size(labels)
        if new_cut > best_cut:
            best_cut = new_cut
        else:
            # If it got worse (can happen with parallel flips), we might want to revert or just continue?
            # Usually, if we flip all positive gain nodes, the total cut should increase.
            # Proof: Sum of gains > 0 implies total cut increases?
            # Yes, the change in cut size is exactly sum of gains / 2?
            # Let's verify.
            # Change in cut = sum over flipped v of (new_contrib - old_contrib)
            # new_contrib - old_contrib = s_v - o_v = gain_v
            # So total change = sum(gains).
            # If all gains > 0, total change > 0. So cut must increase.
            # So best_cut update is safe.
            pass

    return labels
# EVOLVE-BLOCK-END
