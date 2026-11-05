import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Spectral Prioritization via Laplacian Eigenvectors.
    - Compute the graph Laplacian L = D - A.
    - Find the eigenvectors corresponding to the smallest non-zero eigenvalues.
    - These eigenvectors (Fiedler vectors) capture the coarse structure of the graph.
    - Vertices with large components in these eigenvectors are considered "spectrally peripheral".
    - In the context of Independent Set, peripheral vertices in dense graphs are often 
      located in sparser regions or on the boundary of clusters, making them 
      good candidates for selection as they may have fewer "competing" neighbors 
      in the immediate vicinity compared to core vertices.
    - We score vertices by their magnitude in the spectral embedding space defined 
      by the top k smallest non-zero eigenvectors.
    - This is a deterministic, structural heuristic that does not require iterative search.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Compute Degree matrix
    deg = adj.sum(axis=1)
    D = np.diag(deg)
    
    # Compute Laplacian L = D - A
    L = D - adj
    
    # We want eigenvectors for the smallest eigenvalues.
    # np.linalg.eigh returns eigenvalues in ascending order.
    # The first eigenvalue is 0 (for connected graphs) with eigenvector 1/sqrt(n).
    # We want the next few (Fiedler vectors).
    
    try:
        # For n=200, full eigendecomposition is fast.
        # If n were larger, we'd use scipy.sparse.linalg.eigsh.
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Select the top k eigenvectors corresponding to the smallest non-zero eigenvalues.
        # The first one is usually the constant vector. We skip it.
        k = min(10, n - 1) # Use up to 10 dimensions, but at least 1 if possible
        if k > 0:
            # eigenvectors[:, 1:k+1]
            spectral_coords = eigenvectors[:, 1:k+1]
            
            # Calculate the distance of each vertex from the origin in this spectral space.
            # This measures how "distinct" or "peripheral" the vertex is in the spectral structure.
            scores = np.sum(spectral_coords ** 2, axis=1)
            
            # Add a small degree-based component to break ties or reinforce the heuristic.
            # Low degree vertices are generally better for MIS.
            # We want high scores for low degree.
            # Normalize degree to [0, 1]
            if deg.max() > 0:
                deg_norm = deg / deg.max()
            else:
                deg_norm = np.zeros(n)
                
            # Combine: Spectral peripherality (high is good) + Low Degree (high is good)
            # Weights are heuristic. Spectral is the main structural signal.
            final_scores = scores - 0.1 * deg_norm
            
        else:
            # Fallback if n is very small or k=0
            final_scores = -deg
            
    except Exception:
        # Fallback to degree-based if spectral fails
        final_scores = -deg
        
    # Ensure unique scores to avoid tie-breaking issues in the greedy decoder
    # (though the decoder handles ties, distinct scores are safer for "priority")
    # We add a tiny random noise to ensure strict ordering if needed, 
    # but the problem says "descending score order", ties are usually stable or arbitrary.
    # To be safe and deterministic, we can just return the scores.
    
    return final_scores
