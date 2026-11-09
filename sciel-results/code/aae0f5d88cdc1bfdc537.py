"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Combined Heuristic:
    1. Primary: Prefer vertices with lower degree (fewer neighbors to block).
    2. Secondary: Prefer vertices whose neighbors have higher average degrees.
       Rationale: Picking a vertex 'v' blocks its neighbors. If the blocked
       neighbors are high-degree "hubs", it is beneficial because those hubs
       would have blocked many other vertices if they had been picked.
       By blocking them early, we 'save' their neighbors from being blocked.
       This is captured by maximizing the mean degree of neighbors.
    3. Tertiary (Spectral): Use the Fiedler vector (second smallest eigenvector
       of the Laplacian) to identify global structural partitioning. Vertices
       on one side of the sparsest cut may be prioritized to form a coherent
       independent set component, reducing fragmentation.
    
    The score is defined as:
    Priority(v) = -degree(v) + mean_neighbor_degree(v) + alpha * fiedler_component(v)
    
    We use a small alpha so that degree and local structure remain dominant,
    but the spectral term breaks ties and guides the global ordering.
    """
    start_time = time.time()
    
    # Convert to integer array for efficient arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate mean degree of neighbors
    # Add 1.0 to denominator to avoid division by zero and smooth the term
    mean_neighbor_deg = neighbor_degree_sums / (degrees + 1.0)
    
    # Base priority: -degree + mean_neighbor_deg
    # This balances the immediate cost (degree) against the future benefit.
    base_priorities = -degrees + mean_neighbor_deg
    
    # Deadline guard before potentially expensive spectral computation
    if time.time() - start_time > 100:
        return base_priorities.astype(float)
        
    # Spectral Heuristic:
    # Compute the Laplacian L = D - A
    # The Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)
    # provides a 1D embedding that approximates the sparsest cut.
    # We use the component of this vector as a tie-breaker/global guide.
    
    try:
        # Laplacian matrix
        D = np.diag(degrees)
        L = D - adj.astype(np.float64)
        
        # For N=200, eigendecomposition is fast but we guard against timeout
        if time.time() - start_time > 110:
            return base_priorities.astype(float)
            
        # Compute eigenvalues and eigenvectors
        # np.linalg.eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # The second smallest eigenvalue's eigenvector is the Fiedler vector
        # Index 1 (0 is the trivial eigenvalue 0 for connected components)
        # If the graph is disconnected, there might be multiple 0 eigenvalues.
        # We pick the first non-zero eigenvalue's eigenvector to be safe,
        # but typically index 1 is the Fiedler vector for connected graphs.
        
        # Find the index of the second smallest non-zero eigenvalue
        # For a connected graph, eigvals[0] is 0. eigvals[1] is the Fiedler value.
        # If disconnected, there are multiple 0s. We want the first eigenvector
        # that is not constant (which corresponds to the first non-zero eigenvalue).
        
        # Simple approach: use index 1. If graph is disconnected, this might be another 0 eigenvector.
        # To be robust, let's find the first eigenvalue > 1e-8.
        non_zero_idx = np.where(eigvals > 1e-8)[0]
        if len(non_zero_idx) > 0:
            fiedler_idx = non_zero_idx[0]
        else:
            fiedler_idx = 1  # Fallback
        
        fiedler_vec = eigvecs[:, fiedler_idx]
        
        # Normalize the Fiedler vector to have zero mean and unit variance
        # This makes it comparable across different graphs
        fiedler_vec = fiedler_vec - np.mean(fiedler_vec)
        std = np.std(fiedler_vec)
        if std > 1e-8:
            fiedler_vec = fiedler_vec / std
        else:
            fiedler_vec = np.zeros_like(fiedler_vec)
        
        # Weight for the spectral term.
        # The base_priorities are roughly in range [-200, 200].
        # The Fiedler vector is normalized to unit variance.
        # We want the spectral term to be a small perturbation,
        # so it doesn't override the local degree heuristic,
        # but helps in tie-breaking and global ordering.
        # Let's use a small weight like 0.1 or 0.5.
        # Since the Fiedler vector has unit variance, a weight of 1.0 adds ~1 unit of noise.
        # The degree term has a range of ~200. So 1.0 is small relative to degree.
        # Let's try alpha = 1.0.
        alpha = 1.0
        
        priorities = base_priorities + alpha * fiedler_vec
        
    except Exception:
        # If spectral computation fails, fall back to base priorities
        priorities = base_priorities
        
    # Final deadline guard
    if time.time() - start_time > 115:
        return base_priorities.astype(float)
        
    return priorities.astype(float)
# EVOLVE-BLOCK-END
