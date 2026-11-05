"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy:
    1. Compute degree (primary negative signal).
    2. Compute average neighbor degree (secondary positive signal - 'hub clearing').
    3. Compute Fiedler vector (tertiary positive signal - 'peripheral safety').
    4. Combine with tuned weights.
    """
    start_time = time.time()
    
    # Ensure float matrix for linear algebra
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Degree
    deg = A.sum(axis=1)
    
    # 2. Average Neighbor Degree
    sum_nei_deg = A @ deg
    avg_nei_deg = np.zeros(N)
    mask = deg > 0
    avg_nei_deg[mask] = sum_nei_deg[mask] / deg[mask]
    
    # 3. Fiedler Vector (from Laplacian)
    # The Fiedler vector (eigenvector of the 2nd smallest eigenvalue of the Laplacian)
    # tends to separate the graph. Vertices with large |value| are often more peripheral
    # or in distinct clusters, which can be safer to pick in a greedy IS context.
    
    fiedler = np.zeros(N)
    try:
        # Compute Laplacian L = D - A
        # For efficiency, we don't need to explicitly form D as a matrix if we use properties,
        # but for N=200, forming it is fine.
        L = np.diag(deg) - A
        
        # Compute eigenvalues and eigenvectors
        # eigh is for symmetric matrices, returns sorted eigenvalues
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # The first eigenvector (index 0) corresponds to the smallest eigenvalue (0 for connected graph).
        # The second eigenvector (index 1) is the Fiedler vector.
        # Note: If the graph is disconnected, there might be multiple zero eigenvalues.
        # In that case, the "Fiedler" vector is usually the one associated with the smallest positive eigenvalue.
        # However, for ER graphs, they are typically connected.
        # If disconnected, eigvecs[:, 1] might still be a valid separating vector for the first component,
        # or we might need to find the first non-zero eigenvalue's vector.
        
        # Let's check for the first non-zero eigenvalue just in case.
        # For a connected graph, eigvals[0] is ~0, eigvals[1] is > 0.
        # For disconnected, eigvals[0..k-1] are ~0.
        
        # Simple heuristic: take the eigenvector corresponding to the smallest positive eigenvalue.
        # If the graph is connected, this is index 1.
        # If disconnected, it might be index k.
        
        # Let's just use index 1 as a standard approximation, or find the first significant one.
        # Given N=200 and ER graphs, index 1 is almost always the Fiedler vector.
        
        fiedler = eigvecs[:, 1]
        
    except Exception:
        # Fallback if eig fails
        fiedler = np.zeros(N)
        
    # Normalize features to Z-scores for consistent weighting
    def zscore(x):
        m = x.mean()
        s = x.std()
        if s < 1e-9:
            return np.zeros_like(x)
        return (x - m) / s
        
    z_deg = zscore(deg)
    z_avg = zscore(avg_nei_deg)
    z_fiedler = zscore(fiedler)
    
    # Composite Score
    # We want to:
    # - Minimize Degree (pick low degree nodes) -> -z_deg
    # - Maximize Avg Neighbor Degree (pick nodes connected to hubs to remove them) -> +z_avg
    # - Maximize |Fiedler|? Or just Fiedler? 
    #   Usually, we want to pick nodes that are "safe". 
    #   High |Fiedler| means far from the "center" of the partition.
    #   Let's use z_fiedler directly? Or abs?
    #   Inspiration 2 and 3 used abs(fiedler). Let's stick with that as it captures "peripherality".
    
    z_fiedler_abs = zscore(np.abs(fiedler))
    
    # Weights:
    # Degree is the most important factor.
    # Avg Neighbor Degree is a strong secondary factor.
    # Fiedler is a tertiary factor.
    
    # Based on inspiration 1 (score 1.0263): -deg + 1.0 * avg_nei_deg
    # Based on inspiration 2 (score 1.0226): -1.0 * deg_norm + 0.8 * nds_norm + 0.2 * fiedler_norm
    # Based on inspiration 3 (score 1.0188): -1.0 * deg_norm + 0.8 * nds_norm + 0.5 * fiedler_norm
    
    # Let's blend these. The raw scale of z-scores is comparable.
    # w_deg = 1.0
    # w_avg = 0.8  (slightly less than deg, but strong)
    # w_fiedler = 0.2 (smaller contribution)
    
    score = -1.0 * z_deg + 0.8 * z_avg + 0.2 * z_fiedler_abs
    
    # Safety check
    if time.time() - start_time > 110.0:
         # Fallback to simple degree
         return -deg.astype(float)
         
    return score
# EVOLVE-BLOCK-END
