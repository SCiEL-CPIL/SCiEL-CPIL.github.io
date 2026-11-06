"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: assume we have ~140s total for the script, 
    # but each call should be fast. Let's set a per-call limit of 20s to be safe 
    # if called multiple times, or just run as fast as possible.
    # Given N=100, this will run in milliseconds.
    
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Get eigenvectors
    # We only need the second smallest eigenvalue's vector
    # eigh returns eigenvalues in ascending order
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the 2nd column (index 1)
        # If graph is disconnected, there might be multiple zero eigenvalues.
        # We pick the first non-zero eigenvalue's vector? 
        # Standard Max-Cut spectral heuristic uses the Fiedler vector (2nd smallest).
        fiedler = eigvecs[:, 1]
    except Exception:
        # Fallback to random if eigh fails (unlikely)
        fiedler = np.random.randn(n)

    best_labels = None
    best_cut = -1

    # Number of restarts. Since N=100, we can do many.
    # Let's do 100 restarts to be safe and thorough.
    num_restarts = 100
    
    for r in range(num_restarts):
        # Create initial labels from Fiedler vector
        # Threshold at 0
        labels = (fiedler > 0).astype(int)
        
        # Perturbation: flip a small number of random vertices to escape basin
        if r > 0:
            num_flips = np.random.randint(1, 5)
            flip_indices = np.random.choice(n, num_flips, replace=False)
            labels[flip_indices] = 1 - labels[flip_indices]
        
        # Local Search
        # We track the cut size. 
        # Cut size = sum over edges (u,v) of (labels[u] != labels[v])
        # Alternatively, for each v, contribution is deg(v) - same_count(v).
        # Let's maintain the cut size dynamically.
        
        # Initial cut size calculation
        # cut = 0.5 * sum_v (deg(v) - same_v) ? No.
        # cut = sum_{u<v, adj[u,v]=1} (labels[u] != labels[v])
        
        # Efficient update:
        # For vertex v, let S_v be count of neighbors with same label.
        # Let O_v be count of neighbors with different label.
        # deg(v) = S_v + O_v.
        # Current cut contribution of edges incident to v is O_v.
        # If we flip v, new contribution is S_v.
        # Delta = S_v - O_v = S_v - (deg(v) - S_v) = 2*S_v - deg(v).
        
        # We can compute S_v for all v at once.
        # same_matrix = (labels[:, None] == labels[None, :])
        # S = (adj * same_matrix).sum(axis=1)
        
        # Let's do iterative local search.
        improved = True
        max_iters = 500 # Safety cap
        
        # Precompute degrees
        degs = adj.sum(axis=1)
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Compute S_v: number of neighbors with same label
            # labels is shape (n,)
            # adj is (n, n)
            # same = (labels[:, None] == labels[None, :]) -> (n, n) bool
            # S = (adj * same).sum(axis=1)
            
            # Optimization: 
            # S_v = sum_u adj[v,u] * (labels[u] == labels[v])
            
            # We can just loop through vertices and check.
            # For N=100, vectorized is faster.
            
            same = (labels[:, None] == labels[None, :])
            S = (adj * same).sum(axis=1)
            
            # Gain for flipping v: 2*S[v] - degs[v]
            gains = 2 * S - degs
            
            # Find vertices with positive gain
            # We can flip all positive gain vertices? 
            # Flipping one changes the state for neighbors. 
            # If we flip all at once, the gains are calculated based on the OLD state.
            # This is a "batch" update. It might be unstable but often works well.
            # Alternatively, flip one by one in random order.
            
            # Let's try flipping all with gain > 0 simultaneously.
            flip_mask = gains > 0
            if np.any(flip_mask):
                labels[flip_mask] = 1 - labels[flip_mask]
                improved = True
        
        # Calculate final cut size
        # cut = 0.5 * sum_v (degs[v] - S_v) ? 
        # No, cut = sum_{u<v} adj[u,v] (labels[u]!=labels[v])
        # cut = 0.5 * sum_v sum_u adj[v,u] (labels[v]!=labels[u])
        # sum_v sum_u adj[v,u] (labels[v]!=labels[u]) = sum_v (degs[v] - S_v_final)
        
        same_final = (labels[:, None] == labels[None, :])
        S_final = (adj * same_final).sum(axis=1)
        cut_size = np.sum(degs - S_final) / 2.0
        
        if cut_size > best_cut:
            best_cut = cut_size
            best_labels = labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
