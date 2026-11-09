"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure deadline is safely before the 120s limit
    deadline = time.time() + 110.0

    # Pre-compute degree
    degree = adj.sum(axis=1)
    
    # Helper to compute cut size efficiently
    # Cut size = 0.5 * sum_{i,j} A_{ij} (x_i != x_j)
    # In 0/1 labels: (x_i != x_j) is 1 if x_i != x_j.
    # We can compute: sum(A) - sum(A * (x_i == x_j))
    # sum(A * (x_i == x_j)) = sum_i sum_j A_{ij} (1 - |x_i - x_j|) ? No.
    # Let's use the gain formula for local search, and a direct calc for scoring.
    
    def _cut_size(labels):
        # labels is 0/1 array
        # cross = (labels[:, None] != labels[None, :])
        # return int((adj * cross).sum() // 2)
        # Faster:
        # Sum of edges within sets:
        # Set 0: adj[labels==0, :][:, labels==0].sum()
        # Set 1: adj[labels==1, :][:, labels==1].sum()
        # Total edges = adj.sum() / 2
        # Cut = Total - (Within0 + Within1) / 2
        # Within0 = adj[labels==0].sum() / 2 ? No, adj is symmetric.
        # adj[labels==0, :][:, labels==0].sum() counts each internal edge twice.
        # So Within0_count = 0.5 * adj[labels==0, :][:, labels==0].sum()
        
        mask0 = labels == 0
        mask1 = labels == 1
        within0 = adj[mask0, :][:, mask0].sum() / 2.0
        within1 = adj[mask1, :][:, mask1].sum() / 2.0
        total_edges = adj.sum() / 2.0
        return int(total_edges - (within0 + within1))

    def _local_search(labels):
        # Vectorized local search
        # gain[v] = degree[v] - 2 * same_count[v]
        # same_count[v] = sum_{u} adj[v,u] * (labels[u] == labels[v])
        # We can compute same_count efficiently using matrix ops if we map labels to -1/1?
        # Let s[v] = 2*labels[v] - 1 (so -1 or 1)
        # Then (labels[u] == labels[v]) is 1 if s[u]*s[v] == 1, 0 if -1.
        # (1 + s[u]*s[v]) / 2
        # same_count[v] = sum_u adj[v,u] * (1 + s[u]*s[v]) / 2
        #                = 0.5 * degree[v] + 0.5 * sum_u adj[v,u] s[u] s[v]
        # Let M = adj. Let s be vector.
        # term = M @ s
        # sum_u adj[v,u] s[u] s[v] = s[v] * (M @ s)[v]
        # So same_count[v] = 0.5 * degree[v] + 0.5 * s[v] * (M @ s)[v]
        # gain[v] = degree[v] - 2 * same_count[v]
        #          = degree[v] - (degree[v] + s[v]*(M@s)[v])
        #          = - s[v] * (M @ s)[v]
        
        # If gain > 0, flip.
        # Flip means s[v] -> -s[v].
        # We can flip all v where -s[v]*(M@s)[v] > 0, i.e., s[v]*(M@s)[v] < 0.
        
        s = 2.0 * labels - 1.0
        
        for _ in range(100): # Max iterations for one restart
            if time.time() > deadline:
                break
            
            # Compute M @ s
            # adj is float64
            Ms = adj @ s
            
            # Check for improvements
            # We want to flip v if gain > 0.
            # gain = -s[v] * Ms[v]
            # So flip if s[v] * Ms[v] < 0
            prod = s * Ms
            to_flip = prod < 0
            
            if not to_flip.any():
                break
            
            # Flip all at once
            s[to_flip] = -s[to_flip]
            
            # Convert back to labels for next check? 
            # We can keep s and just convert at the end, or convert each time.
            # Converting each time is cheap: (s+1)/2
            # But we need labels for the gain formula in the next iteration? 
            # The formula derived above works directly on s.
            # So we can just keep s.
            
            # However, we need to return labels.
            # And we need to check time.
            
        labels_out = ((s + 1.0) / 2.0).astype(np.int8)
        return labels_out

    best_labels = None
    best_cut = -1

    # Generate diverse initializations
    # 1. Spectral (Fiedler vector)
    inits = []
    try:
        L = np.diag(degree) - adj
        eigvals, eigvecs = np.linalg.eigh(L)
        # Handle disconnected graphs: find smallest non-zero eigenvalue
        # Usually index 1 is Fiedler. If graph is disconnected, there might be multiple 0s.
        # Let's just take the second one.
        fiedler = eigvecs[:, 1]
        labels_spec = (fiedler >= 0).astype(np.int8)
        if labels_spec.sum() > 0 and labels_spec.sum() < n:
            inits.append(labels_spec)
    except Exception:
        pass

    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        inits.append(labels_rand)

    # 3. Alternating
    labels_alt = np.arange(n) % 2
    inits.append(labels_alt)

    # Run local search on each initialization
    for init_labels in inits:
        if time.time() > deadline:
            break
        
        current_labels = _local_search(init_labels.copy())
        current_cut = _cut_size(current_labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()

    # If we have time left, try more random restarts
    while time.time() < deadline - 5.0:
        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        current_labels = _local_search(labels_rand)
        current_cut = _cut_size(current_labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()

    # Fallback if no labels found (should not happen)
    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
        best_labels[::2] = 1

    return best_labels
# EVOLVE-BLOCK-END
