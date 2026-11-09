"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency):
    """
    Generate initial labels using spectral partitioning.
    Uses the second smallest eigenvector of the graph Laplacian.
    """
    n = len(adjacency)
    # Convert to sparse matrix for efficiency
    A = csr_matrix(adjacency)
    
    # Compute degrees
    degrees = A.sum(axis=1).A1
    
    # Build Laplacian L = D - A
    # We can construct it as sparse matrices
    D = csr_matrix((degrees, (range(n), range(n))), shape=(n, n))
    L = D - A
    
    # We want the eigenvector corresponding to the second smallest eigenvalue (Fiedler vector).
    # scipy.sparse.linalg.eigsh finds eigenvalues.
    # k=2 to get the two smallest.
    # sigma=0 targets eigenvalues near 0.
    try:
        # n=100 is small, so this should be fast.
        # Use which='SM' (smallest magnitude) or 'SA' (smallest algebraic).
        # For Laplacian, eigenvalues are non-negative.
        # The smallest is 0 (for connected graph). The second smallest is what we want.
        vals, vecs = eigsh(L, k=2, which='SA', sigma=0, return_eigenvectors=True)
        
        # Sort eigenvalues to ensure we pick the correct one
        idx = np.argsort(vals)
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # which is index 1 after sorting (since index 0 is the constant vector for eigenvalue 0)
        fiedler_vec = vecs[:, idx[1]]
        
        # Split based on sign of the Fiedler vector
        # Threshold at 0
        labels = (fiedler_vec > 0).astype(np.int8)
        
        # Ensure both partitions are non-empty? 
        # Usually spectral partitioning works well, but if the graph is disconnected,
        # the second eigenvalue is also 0, and the eigenvector might be indicator of components.
        # For random graphs, it's likely connected.
        
        return labels
        
    except Exception:
        # Fallback to random if spectral fails
        return np.random.randint(0, 2, size=n)

def _local_search(adjacency, labels, deadline):
    """
    Hill climbing local search with random restarts handled externally or internally.
    Here we do a single pass of steepest ascent or best-improvement until no improvement.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    # Precompute degrees
    degrees = adjacency.sum(axis=1)
    
    # Compute gains
    # gain[v] = (neighbors with different label) - (neighbors with same label)
    # If we flip v, cut changes by gain[v].
    # other_count[v] = sum( (labels[u] != labels[v]) * adj[v,u] )
    # same_count[v] = sum( (labels[u] == labels[v]) * adj[v,u] )
    
    def compute_gains(labels):
        diff_mat = labels[None, :] != labels[:, None]
        other_counts = (adjacency & diff_mat).sum(axis=1)
        same_counts = (adjacency & ~diff_mat).sum(axis=1)
        return other_counts - same_counts

    gains = compute_gains(labels)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find the vertex with the maximum positive gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain > 0:
            v = max_gain_idx
            old_label = labels[v]
            labels[v] = 1 - old_label
            current_cut += max_gain
            
            # Update gains
            # v's gain becomes -old_gain
            gains[v] = -max_gain
            
            # Update neighbors
            neighbors = np.where(adjacency[v])[0]
            for u in neighbors:
                if labels[u] == 1 - old_label:
                    # u now has same label as v (new label)
                    # Before: u had different label from v (old label) -> u's 'other' included v
                    # After: u has same label as v -> u's 'same' includes v
                    # So 'other' decreases by 1, 'same' increases by 1.
                    # gain[u] = other - same. Change: -1 - (+1) = -2
                    gains[u] -= 2
                else:
                    # u now has different label from v (new label)
                    # Before: u had same label as v (old label) -> u's 'same' included v
                    # After: u has different label from v -> u's 'other' includes v
                    # So 'other' increases by 1, 'same' decreases by 1.
                    # gain[u] = other - same. Change: +1 - (-1) = +2
                    gains[u] += 2
            
            improved = True
            
    return labels, current_cut

def partition(adjacency):
    """
    Main partition function using Spectral Initialization + Local Search.
    """
    n = len(adjacency)
    deadline = time.time() + 145.0 # Safety margin (limit is 150s)
    
    best_labels = None
    best_cut = -1
    
    # Strategy:
    # 1. Get a good initial guess via spectral method.
    # 2. Refine it with local search.
    # 3. If time permits, try a few random restarts with local search to escape local optima.
    #    However, spectral init is often very strong.
    
    # Step 1: Spectral Init
    t0 = time.time()
    init_labels = _spectral_init(adjacency)
    
    # Step 2: Local Search on spectral init
    labels, cut = _local_search(adjacency, init_labels, deadline)
    
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Step 3: Random Restarts if time remains
    # We can try a few random starts, but local search on random starts is often slower to converge 
    # or finds worse optima than spectral. 
    # Let's try 5 random restarts if we have plenty of time.
    
    t1 = time.time()
    elapsed = t1 - t0
    # If spectral + LS took less than 10s, we have ~130s left.
    # Each random LS might take a few seconds.
    
    num_restarts = 0
    while time.time() < deadline - 5.0: # Leave 5s buffer
        # Random init
        rand_labels = np.random.randint(0, 2, size=n)
        
        # Run local search
        # To avoid getting stuck in the same local optimum, we could use a variant of LS
        # but standard LS is fast.
        
        # We need to be careful not to spend too much time on one restart.
        # Let's give each restart a sub-deadline.
        sub_deadline = min(deadline - 5.0, time.time() + 10.0)
        
        r_labels, r_cut = _local_search(adjacency, rand_labels, sub_deadline)
        
        if r_cut > best_cut:
            best_cut = r_cut
            best_labels = r_labels.copy()
        
        num_restarts += 1
        
        # Check if we are running out of time
        if time.time() > deadline - 10.0:
            break
            
        # If we've done a few and aren't improving, maybe stop?
        # Or just keep going until deadline.
        
    return best_labels
# EVOLVE-BLOCK-END
