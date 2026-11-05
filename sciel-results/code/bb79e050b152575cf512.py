import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Convert adjacency to a boolean numpy array.
    2. Compute the eigenvalues and eigenvectors of the adjacency matrix.
    3. Use the components of the eigenvector corresponding to the smallest eigenvalue 
       (or a combination of eigenvalues) to determine vertex scores.
       The intuition is that vertices with high "independence" in the spectral sense 
       (often associated with the ground state of the Laplacian or similar structures) 
       should be picked first.
       
       Specifically, for an Erdos-Renyi graph, we can look at the eigenvector associated 
       with the smallest eigenvalue of the adjacency matrix (which is often related to 
       bipartiteness or clustering properties) or the largest eigenvalue of the Laplacian.
       
       A common heuristic for MIS is to pick vertices that are "far" from each other. 
       Spectral methods can capture global structure.
       
       Let's try using the eigenvector corresponding to the smallest eigenvalue of the 
       adjacency matrix. The magnitude of the components might indicate importance.
       Alternatively, we can use the Fiedler vector (2nd smallest eigenvalue of Laplacian).
       
       However, a simpler spectral heuristic for Independent Set is related to the 
       eigenvector of the largest eigenvalue of the adjacency matrix (Perron-Frobenius). 
       Vertices with large components in this eigenvector are "central". 
       But for MIS, we want to avoid central vertices if they connect to many others? 
       Actually, the greedy decoder skips neighbors. So if we pick a central vertex first, 
       we eliminate many candidates. This is bad for size.
       So we want to pick vertices with LOW centrality first?
       
       Let's test the hypothesis: Score = - (component in principal eigenvector).
       Or Score = - degree (which is current).
       
       Let's try a different spectral invariant: The eigenvalues of the adjacency matrix.
       We can compute a score based on the local density or the spectral radius of the 
       neighborhood.
       
       A robust spectral method for MIS is to use the eigenvector of the smallest 
       eigenvalue of the Laplacian (Fiedler vector) or simply the eigenvector of the 
       smallest eigenvalue of the adjacency matrix.
       
       Let's try:
       1. Compute eigenvalues and eigenvectors of A.
       2. Let v_min be the eigenvector for the smallest eigenvalue.
       3. Score = -abs(v_min) or just -v_min?
       
       Actually, for regular graphs, the smallest eigenvalue eigenvector is uniform.
       For ER graphs, it's not.
       
       Another idea: Use the eigenvalues to compute a "spectral degree".
       But let's stick to a direct spectral score.
       
       Let's try: Score = - (A @ v_max)[i] where v_max is the principal eigenvector.
       This is proportional to degree for regular graphs, but captures structure otherwise.
       Since we want to avoid high-degree nodes (to preserve options), we want low scores 
       for high "effective degree".
       
       Let's try: Score = - (v_max @ A)[i] = - (A @ v_max)[i].
       
       If this doesn't work well, we can fall back to -degree.
       
       To be safe and potentially better, we can combine:
       Score = - (alpha * degree + (1-alpha) * spectral_component)
       
       But the directive asks for a STRUCTURAL change.
       Let's implement the spectral score purely.
       
       Note: The greedy decoder picks in descending score order.
       So high score = picked first.
       We want to pick vertices that are "isolated" or "low impact" first?
       No, the greedy decoder picks a vertex, then removes its neighbors.
       If we pick a high-degree vertex first, we remove many vertices.
       This usually results in a SMALLER independent set.
       So we should pick LOW degree vertices first.
       Current strategy: -degree. High score = low degree.
       
       Spectral equivalent:
       The principal eigenvector v_max has positive components.
       Vertices with high v_max[i] are "central".
       So we should assign LOW scores to high v_max[i].
       Score = -v_max[i].
       
       Let's implement this.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    try:
        adj = np.asarray(adjacency, dtype=bool)
        n = adj.shape[0]
        
        if n < 2:
            return np.zeros(n)
        
        # Ensure symmetry (ER graphs are undirected)
        # adj = (adj | adj.T)
        
        # Compute eigenvalues and eigenvectors
        # For n=200, this is fast.
        # Use symmetric eigensolver if possible, but numpy.linalg.eigh is for symmetric.
        # adj is boolean, so it's symmetric if the graph is undirected.
        # Let's enforce symmetry to be safe for eigh.
        adj_sym = (adj | adj.T).astype(float)
        
        # Compute eigenvalues and eigenvectors
        # eigh returns eigenvalues in ascending order.
        eigvals, eigvecs = np.linalg.eigh(adj_sym)
        
        # The largest eigenvalue is the last one.
        # The principal eigenvector is the last column.
        v_max = eigvecs[:, -1]
        
        # We want to pick vertices with LOW centrality first.
        # So score should be inversely related to v_max.
        # Since v_max components are positive (Perron-Frobenius), we can use -v_max.
        
        # However, if the graph is disconnected, the principal eigenvector might not be unique or might have zeros.
        # But for ER graphs with n=200, it's likely connected or has one giant component.
        
        # Let's try Score = -v_max.
        scores = -v_max
        
        # Check time
        if time.time() - start_time > DEADLINE:
            # Fallback to degree if spectral is too slow (unlikely for n=200)
            deg = adj.sum(axis=1)
            return -deg
            
        return scores
        
    except Exception:
        # Fallback to simple degree score
        try:
            deg = np.sum(adjacency, axis=1)
            return -deg
        except:
            return np.zeros(adjacency.shape[0])
