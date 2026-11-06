"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy:
    1. Compute degree and clustering-based penalties.
    2. Use spectral embedding (eigenvectors of normalized Laplacian) for global structure.
    3. Iteratively refine scores by simulating the greedy selection process,
       updating scores based on 'effective degree' (number of unblocked neighbors).
    4. Combine all signals into a final score.
    """
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Ensure adjacency is a numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Basic features
    degree = adj.sum(axis=1)
    
    # Clustering coefficient approximation: for each vertex, 
    # count edges among its neighbors / possible edges
    # Use a fast approximation: sum of degrees of neighbors
    neighbor_degree_sum = adj @ degree
    
    # Spectral embedding using normalized Laplacian
    # L = I - D^{-1/2} A D^{-1/2}
    sqrt_deg = np.sqrt(degree + 1e-10)
    inv_sqrt_deg = 1.0 / sqrt_deg
    
    # Normalized adjacency: D^{-1/2} A D^{-1/2}
    norm_adj = adj * inv_sqrt_deg[:, None] * inv_sqrt_deg[None, :]
    
    # Power iteration to find top eigenvectors of norm_adj
    # We want the smallest eigenvalues of L, which correspond to largest of norm_adj
    # Actually, for community detection, we want the Fiedler vector (2nd smallest eig of L)
    # which is the 2nd largest eig of norm_adj
    
    def power_iteration(matrix, k, num_iters=50, tol=1e-8):
        """Find top k eigenvectors using power iteration with deflation."""
        n = matrix.shape[0]
        eigenvectors = []
        residual = matrix.copy()
        
        for i in range(k):
            # Start with random vector
            v = np.random.randn(n)
            v /= (np.linalg.norm(v) + 1e-10)
            
            for _ in range(num_iters):
                v_new = residual @ v
                # Deflate against previous eigenvectors
                for prev_v in eigenvectors:
                    v_new -= np.dot(v_new, prev_v) * prev_v
                norm = np.linalg.norm(v_new)
                if norm < 1e-10:
                    v_new = np.random.randn(n)
                    norm = np.linalg.norm(v_new)
                v_new /= norm
                
                if np.linalg.norm(v_new - v) < tol:
                    v = v_new
                    break
                v = v_new
            
            eigenvectors.append(v)
            # Deflate the matrix
            residual = residual - np.outer(v, v) @ residual
        
        return eigenvectors
    
    try:
        # Get top 3 eigenvectors for spectral features
        eigvecs = power_iteration(norm_adj, 3, num_iters=30)
        spectral_scores = []
        for v in eigvecs:
            spectral_scores.append(v)
        spectral_matrix = np.column_stack(spectral_scores) if spectral_scores else np.zeros((n, 1))
    except Exception:
        spectral_matrix = np.zeros((n, 1))
    
    # Compute a 'community centrality' score from spectral embedding
    # Vertices with similar spectral coordinates are in the same community
    # We want to pick one representative per community
    if spectral_matrix.shape[1] > 0:
        # Distance to centroid of all points
        centroid = spectral_matrix.mean(axis=0)
        dist_to_centroid = np.linalg.norm(spectral_matrix - centroid, axis=1)
        # Vertices far from centroid are more 'central' in their community
        spectral_centrality = dist_to_centroid
    else:
        spectral_centrality = np.zeros(n)
    
    # Iterative refinement: simulate greedy selection multiple times
    # and track how often each vertex gets selected
    num_iterations = 20
    selection_count = np.zeros(n)
    blocked_count = np.zeros(n)
    
    for it in range(num_iterations):
        if time.time() > deadline - 10:
            break
        
        # Create initial scores
        # Combine: low degree, low neighbor degree sum, high spectral centrality
        base_score = (-degree * 2.0 
                     - neighbor_degree_sum * 0.5 
                     + spectral_centrality * 1.0)
        
        # Add some randomness to break ties
        base_score += np.random.randn(n) * 0.1
        
        # Simulate greedy selection
        selected = np.zeros(n, dtype=bool)
        order = np.argsort(-base_score)
        
        for v in order:
            if time.time() > deadline - 5:
                break
            if not selected[v]:
                selected[v] = True
                selection_count[v] += 1
                # Mark neighbors as blocked
                neighbors = np.where(adj[v] > 0.5)[0]
                for nb in neighbors:
                    if not selected[nb]:
                        blocked_count[nb] += 1
    
    # Final score: combine all signals
    # Vertices that are frequently selected should have high scores
    # Vertices that are frequently blocked should have low scores
    freq_selected = selection_count / (num_iterations + 1e-10)
    freq_blocked = blocked_count / (num_iterations + 1e-10)
    
    # Normalize components
    def normalize(x):
        if x.std() > 1e-10:
            return (x - x.mean()) / x.std()
        return x
    
    score_degree = normalize(-degree)
    score_neighbor_deg = normalize(-neighbor_degree_sum)
    score_spectral = normalize(spectral_centrality)
    score_freq_sel = normalize(freq_selected)
    score_freq_block = normalize(-freq_blocked)
    
    # Weighted combination
    final_score = (0.25 * score_degree 
                 + 0.15 * score_neighbor_deg 
                 + 0.10 * score_spectral 
                 + 0.35 * score_freq_sel 
                 + 0.15 * score_freq_block)
    
    return final_score
# EVOLVE-BLOCK-END
