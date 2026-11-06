"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization + local search refinement.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph.
        
    Returns:
        np.array of 0/1 labels per vertex.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Deadline guard: 145 seconds (5s safety margin before 150s kill)
    start_time = time.time()
    DEADLINE = 145.0
    
    # Convert adjacency to boolean numpy array
    adj = np.array(adjacency, dtype=bool)
    
    # Build degree vector and Laplacian
    degree = adj.sum(axis=1)
    
    # Build sparse Laplacian for eigenvalue computation
    adj_sparse = csr_matrix(adj)
    L = laplacian(adj_sparse, normed=False)
    
    # Compute Fiedler vector (eigenvector for second-smallest eigenvalue)
    # For n=100, this is fast
    try:
        # Use scipy's eigh for symmetric matrices
        # We want the second-smallest eigenvalue's eigenvector
        # For small n, we can compute all eigenvalues
        L_dense = L.toarray()
        eigvals, eigvecs = eigh(L_dense, subset_by_index=[1, 1])
        fiedler = eigvecs[:, 0]
    except Exception:
        # Fallback: random initialization
        fiedler = np.random.randn(n)
    
    # Generate initial labeling from Fiedler vector
    # Threshold at median (or zero)
    median_val = np.median(fiedler)
    labels = (fiedler > median_val).astype(int)
    
    # Ensure both classes are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.arange(n) % 2
    
    # Helper: compute cut size
    def cut_size(labels):
        # Count edges between 0 and 1
        # cut = sum over edges (i,j) where labels[i] != labels[j]
        # Efficient: for each vertex v in class 0, count neighbors in class 1
        # Or: use matrix multiplication
        # cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # More efficient: 
        # For vertex v, gain from flipping = (neighbors in other class) - (neighbors in same class)
        # Current cut = sum over v in 0 of (neighbors in 1) + sum over v in 1 of (neighbors in 0)
        # But each edge counted once, so:
        # cut = sum over v in 0 of (neighbors in 1)
        # Since each crossing edge has one endpoint in 0
        in0 = labels == 0
        in1 = labels == 1
        cut = 0
        for v in range(n):
            if in0[v]:
                cut += adj[v][in1].sum()
            else:
                cut += adj[v][in0].sum()
        # Wait, this double counts. Let me think again.
        # An edge (u,v) with u in 0, v in 1: counted once when v=u (if u in 0) 
        # Actually, if we sum over all v in 0 the number of neighbors in 1, 
        # each crossing edge is counted exactly once (from the 0-side endpoint).
        # So the above is correct.
        return cut
    
    # Better: vectorized cut computation
    def cut_size_fast(labels):
        # labels: 0/1 array
        # Convert to +1/-1
        x = 2 * labels - 1  # +1 for 1, -1 for 0
        # cut = (1/4) * sum_{i,j} adj[i,j] * (1 - x[i]*x[j])
        #     = (1/4) * (sum_{i,j} adj[i,j] - sum_{i,j} adj[i,j]*x[i]*x[j])
        #     = (1/4) * (2*m - x^T * A * x)
        # where m = number of edges
        # So maximizing cut = minimizing x^T A x (since 2*m is constant)
        # Or: cut = m - (1/4) * x^T A x
        m = degree.sum() / 2.0
        Ax = adj @ x
        xAx = x @ Ax
        return int(round(m - xAx / 4.0))
    
    def flip_gain(v, labels):
        """Gain from flipping vertex v."""
        # Current contribution of v to cut:
        # If v in 0: contributes (neighbors in 1)
        # If v in 1: contributes (neighbors in 0)
        # After flip:
        # If v was in 0, now in 1: contributes (neighbors in 0)
        # Gain = new - old
        if labels[v] == 0:
            old_contrib = adj[v][labels == 1].sum()
            new_contrib = adj[v][labels == 0].sum()
        else:
            old_contrib = adj[v][labels == 0].sum()
            new_contrib = adj[v][labels == 1].sum()
        return new_contrib - old_contrib
    
    # Local search: repeatedly flip vertices with positive gain
    def local_search(labels, max_time=None):
        """1-flip local search. Returns improved labels."""
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if max_time is not None and time.time() > max_time:
                    break
                gain = flip_gain(v, labels)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
            if max_time is not None and time.time() > max_time:
                break
        return labels
    
    # Run local search from spectral initialization
    best_labels = labels.copy()
    best_cut = cut_size_fast(best_labels)
    
    # Local search pass 1
    labels = local_search(labels, max_time=start_time + DEADLINE - 5)
    cur_cut = cut_size_fast(labels)
    if cur_cut > best_cut:
        best_labels = labels.copy()
        best_cut = cur_cut
    
    # Multi-start: try a few random perturbations and re-optimize
    rng = np.random.default_rng(42)
    num_starts = 5
    
    for s in range(num_starts):
        if time.time() > start_time + DEADLINE - 10:
            break
        
        # Perturb the best solution
        perturbed = best_labels.copy()
        # Flip a random subset of vertices (10-20%)
        num_flips = max(1, int(n * 0.15))
        flip_indices = rng.choice(n, size=num_flips, replace=False)
        perturbed[flip_indices] = 1 - perturbed[flip_indices]
        
        # Local search from perturbed solution
        perturbed = local_search(perturbed, max_time=start_time + DEADLINE - 5)
        cur_cut = cut_size_fast(perturbed)
        if cur_cut > best_cut:
            best_labels = perturbed.copy()
            best_cut = cur_cut
    
    # Final polish: one more local search pass on the best solution
    if time.time() < start_time + DEADLINE - 2:
        final_labels = best_labels.copy()
        final_labels = local_search(final_labels, max_time=start_time + DEADLINE - 1)
        final_cut = cut_size_fast(final_labels)
        if final_cut > best_cut:
            best_labels = final_labels
            best_cut = final_cut
    
    return best_labels
# EVOLVE-BLOCK-END
