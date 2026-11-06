"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solves Max-Cut using Spectral Initialization and Multi-Start Greedy Refinement.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin for 120s limit
    
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Helper to compute cut size
    def cut_size(labels):
        # labels: 0/1 array
        # cut = sum of edges where labels differ
        # Using matrix ops:
        # Let s = 2*labels - 1 (values in {-1, 1})
        # Cut = 0.5 * (sum(A) - s^T A s) is not quite right for 0/1 labels directly without care.
        # Simpler:
        # diff = (labels[:, None] != labels[None, :])
        # return np.sum(adj * diff)
        # This is O(N^2) memory and time. For N=100, it's 10k ops.
        # Faster:
        # s = 2.0 * labels - 1.0
        # total_edges = np.sum(adj) / 2.0
        # same_side_sum = np.sum(adj * (s[:, None] * s[None, :]))
        # cut = total_edges - same_side_sum / 2.0 ? No.
        # Let's stick to the direct sum for clarity and speed is fine.
        # Actually, vectorized diff check is fast enough for N=100.
        return np.sum(adj * (labels[:, None] != labels[None, :])) // 2 # Integer division? adj is int. 
        # Wait, if adj is symmetric, sum(adj * diff) counts each crossing edge twice.
        # So divide by 2.
        
    # Greedy Refinement using Gain Updates
    def greedy_refine(adj, init_labels):
        labels = init_labels.copy()
        n = len(labels)
        # Convert to signs for efficient gain calculation
        # s = 2*labels - 1
        s = 2.0 * labels - 1.0
        
        # Initial Gains G(v) = sum_u A_uv s_u s_v
        # G = s @ (adj @ s) ? No.
        # G[v] = sum_u adj[v,u] s[v] s[u] = s[v] * sum_u adj[v,u] s[u]
        # Let As = adj @ s
        # G = s * As
        
        As = adj @ s
        G = s * As
        
        # We need to find max G. If max G > 0, flip.
        # We iterate until no positive gain.
        
        # To avoid infinite loops in case of ties or zero gains, we track steps.
        # Max possible steps is bounded, but let's just cap it.
        max_steps = 10000
        
        for _ in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
                
            # Find vertex with max gain
            v = np.argmax(G)
            gain = G[v]
            
            if gain <= 1e-9:
                break
                
            # Flip v
            s[v] = -s[v]
            
            # Update G for neighbors of v
            # Neighbors are u where adj[v,u] != 0
            neighbors = np.nonzero(adj[v])[0]
            
            # Update formula derived earlier:
            # G(u) <- G(u) - 2 * A_uv * s_u * s_v_old
            # Note: s_v is now flipped. We need s_v_old.
            # s_v_old = -s[v] (current)
            # So delta = -2 * adj[v, u] * s[u] * (-s[v]) = 2 * adj[v, u] * s[u] * s[v]
            
            # Vectorized update for neighbors
            # G[neighbors] += 2 * adj[v, neighbors] * s[neighbors] * s[v]
            
            G[neighbors] += 2.0 * adj[v, neighbors] * s[neighbors] * s[v]
            
            # Also, G[v] itself changes?
            # G(v) = sum_u A_vu s_v s_u = s_v sum_u A_vu s_u
            # When we flip s_v, s_v becomes -s_v.
            # The sum part sum_u A_vu s_u does NOT change because s_u for u != v didn't change.
            # Wait, does G[v] depend on s[v]?
            # G[v] = s[v] * (adj @ s)[v]
            # (adj @ s)[v] is sum_u A_vu s_u. This term does not include s[v] (since A_vv=0).
            # So (adj @ s)[v] is constant during the flip of v.
            # Therefore G[v] flips sign.
            G[v] = -G[v]
            
            # Check if we should stop if time is tight
            if time.time() - start_time > DEADLINE:
                break

        # Convert back to 0/1 labels
        # s = 2*labels - 1 => labels = (s + 1) / 2
        final_labels = ((s + 1.0) // 2).astype(np.int8)
        return final_labels

    # 1. Spectral Initialization
    # Laplacian L = D - A
    degrees = np.sum(adj, axis=1)
    L = np.diag(degrees) - adj
    # Compute eigenvalues and eigenvectors
    # For N=100, eigh is fast
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Second smallest eigenvalue corresponds to index 1 (index 0 is 0 for connected graph)
        # If graph is disconnected, there might be multiple 0s.
        # We pick the eigenvector corresponding to the smallest non-zero eigenvalue.
        # Usually index 1 works for connected graphs.
        # To be safe, find index of smallest eigenvalue > 1e-10.
        non_zero_indices = np.where(eigenvalues > 1e-10)[0]
        if len(non_zero_indices) > 0:
            idx = non_zero_indices[0]
        else:
            idx = 1 if n > 1 else 0
            
        fiedler_vec = eigenvectors[:, idx]
        # Threshold at median
        median_val = np.median(fiedler_vec)
        spectral_labels = (fiedler_vec >= median_val).astype(np.int8)
    except Exception:
        # Fallback to random if eigen fails
        spectral_labels = np.random.randint(0, 2, n).astype(np.int8)

    best_labels = None
    best_cut = -1

    # 2. Multi-Start
    # Start 1: Spectral
    refined = greedy_refine(adj, spectral_labels)
    cut_val = cut_size(refined)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined

    # Start 2-10: Random
    # Number of random restarts depends on time
    # Each greedy run is ~1-10ms. We can do many.
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        random_labels = np.random.randint(0, 2, n).astype(np.int8)
        refined = greedy_refine(adj, random_labels)
        cut_val = cut_size(refined)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined

    if best_labels is None:
        # Fallback
        best_labels = np.zeros(n, dtype=np.int8)
        
    return best_labels
# EVOLVE-BLOCK-END
