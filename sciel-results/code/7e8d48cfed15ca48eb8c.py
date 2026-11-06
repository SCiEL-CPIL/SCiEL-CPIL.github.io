"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy:
    1. Convert adjacency to sparse matrix.
    2. Compute the graph Laplacian.
    3. Use power iteration / eigsh to find the Fiedler vector (second smallest eigenvector).
    4. Combine spectral information with local degree and neighborhood structure.
    5. Return scores that prioritize vertices in sparse regions with low blocking potential.
    """
    start_time = time.time()
    DEADLINE = 115  # Safety margin before 120s limit
    
    n = adjacency.shape[0]
    A = csc_matrix(adjacency)
    
    # Compute degree
    degrees = np.asarray(A.sum(axis=1)).flatten()
    
    # Compute Laplacian: L = D - A
    D = csc_matrix((degrees, (np.arange(n), np.arange(n))))
    L = D - A
    L = L.tocsr()
    
    # Try to get the Fiedler vector (2nd smallest eigenvector of L)
    # For disconnected graphs, there may be multiple zero eigenvalues
    # We'll use eigsh to get a few small eigenpairs
    try:
        # Get smallest eigenvalues and eigenvectors
        # k=3 to handle potential disconnected components
        if n > 10:
            vals, vecs = eigsh(L, k=min(3, n-1), which='SM', return_eigenvectors=True)
        else:
            # Fallback for small graphs
            L_dense = L.toarray()
            vals, vecs = np.linalg.eigh(L_dense)
            vals = vals[:min(3, n)]
            vecs = vecs[:, :min(3, n)]
        
        # The Fiedler vector is typically the eigenvector corresponding to the 
        # second smallest non-zero eigenvalue, but for our purpose, we can use
        # the eigenvector with the most "spread" or the one that best separates
        # the graph. Let's use the eigenvector corresponding to the smallest 
        # positive eigenvalue, or just the second eigenvector if it's non-trivial.
        
        # For independent set purposes, we want vertices that are "peripheral" 
        # or in sparse regions. The Fiedler vector tends to have larger magnitude 
        # for vertices that are more "central" in some sense, but we want the opposite.
        
        # Let's use the absolute value of the Fiedler vector as a centrality measure,
        # and prioritize vertices with LOW centrality (i.e., peripheral vertices).
        
        # Find the eigenvector with the smallest positive eigenvalue
        positive_mask = vals > 1e-10
        if np.any(positive_mask):
            # Get the eigenvector for the smallest positive eigenvalue
            pos_indices = np.where(positive_mask)[0]
            idx = pos_indices[np.argmin(vals[pos_indices])]
            fiedler_vec = vecs[:, idx]
        else:
            # All eigenvalues are near zero (disconnected with isolated vertices)
            # Use the last eigenvector as a fallback
            fiedler_vec = vecs[:, -1]
        
        # Normalize the Fiedler vector
        fiedler_abs = np.abs(fiedler_vec)
        fiedler_abs = fiedler_abs / (np.max(fiedler_abs) + 1e-12)
        
    except Exception:
        # Fallback if eigsh fails
        fiedler_abs = np.ones(n) / n
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Return simple degree-based score
        return -degrees.astype(float)
    
    # Compute neighborhood degree sum (sum of degrees of neighbors)
    # This measures how "blocking" a vertex is
    neighbor_deg_sum = np.asarray(A.dot(degrees)).flatten()
    
    # Compute clustering-like measure: fraction of edges among neighbors
    # For each vertex, count edges among its neighbors
    # This is expensive for large graphs, so we'll use a simpler proxy
    # Sum of degrees of neighbors already captures some of this
    
    # Compute a "local density" measure
    # Vertices in dense regions should be deprioritized
    # Use the ratio of neighbor degree sum to degree (if degree > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_neighbor_degree = np.where(degrees > 0, neighbor_deg_sum / degrees, 0)
    
    # Combine signals:
    # 1. Low degree is good (less blocking)
    # 2. Low Fiedler centrality is good (peripheral vertices)
    # 3. Low average neighbor degree is good (sparse neighborhood)
    
    # Normalize each signal to [0, 1]
    deg_norm = degrees / (np.max(degrees) + 1e-12)
    fiedler_norm = fiedler_abs  # Already in [0, 1]
    avg_deg_norm = avg_neighbor_degree / (np.max(avg_neighbor_degree) + 1e-12)
    
    # Weighted combination: prioritize low degree, low centrality, low neighbor density
    # We want HIGH score for vertices we want to pick early
    # So we negate the "bad" signals
    score = -(0.4 * deg_norm + 0.3 * fiedler_norm + 0.3 * avg_deg_norm)
    
    # Check time again
    if time.time() - start_time > DEADLINE:
        return score.astype(float)
    
    # Optional: refine with a local improvement pass
    # Try a few iterations of adjusting scores based on greedy selection feedback
    # This is a simple heuristic: simulate greedy selection and adjust scores
    
    # For now, return the combined score
    return score.astype(float)
# EVOLVE-BLOCK-END
