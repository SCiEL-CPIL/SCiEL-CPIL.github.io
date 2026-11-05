"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Fixed safety margin so we return well before the 120 s wall-clock limit.
# We use 110s to be safe.
DEADLINE_SECONDS = 110.0


def _cut_size(labels, adj):
    """Compute the cut size for a given labeling."""
    # diff[i, j] is True if labels[i] != labels[j]
    # We want sum of adj[i, j] where diff[i, j] is True
    # Since adj is boolean/symmetric, we can just sum adj * diff
    # But adj * diff creates a full matrix. For n=100, this is 10k ops, very fast.
    diff = (labels[None, :] != labels[:, None])
    return int(np.sum(adj * diff))


def _local_search(labels, adj, row_sums, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain
    until no more improvements are possible or deadline is reached.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    
    # Precompute initial same-side counts for each vertex
    # same_count[v] = sum(adj[v, u] for u where labels[u] == labels[v])
    # We can compute this efficiently using matrix multiplication or boolean indexing.
    # For n=100, even O(n^2) per iteration is fine, but let's be smart.
    
    # Vectorized gain calculation:
    # gain[v] = 2 * (deg[v] - same_count[v]) - deg[v] = deg[v] - 2 * same_count[v]
    # Wait, if we flip v:
    # Old contribution of edges incident to v to cut: other_count[v]
    # New contribution: same_count[v]
    # Change: same_count[v] - other_count[v]
    # other_count[v] = deg[v] - same_count[v]
    # Change: same_count[v] - (deg[v] - same_count[v]) = 2 * same_count[v] - deg[v]
    # So we want to flip if 2 * same_count[v] - deg[v] > 0
    
    # Let's maintain same_count array.
    # Initially:
    # same_count[v] = sum(adj[v, :][labels == labels[v]])
    
    while True:
        # Check deadline
        if time.time() > deadline:
            break
            
        # Compute same_count for all vertices vectorized
        # labels is 0/1. 
        # For each v, same_count[v] = sum(adj[v, u] * (labels[u] == labels[v]))
        # This is equivalent to:
        # same_count = np.sum(adj * (labels[None, :] == labels[:, None]), axis=1)
        # This is O(n^2) per iteration. n=100, so 10k ops. Very fast.
        
        same_count = np.sum(adj * (labels[None, :] == labels[:, None]), axis=1)
        
        # Calculate gain for flipping each vertex
        # gain[v] = 2 * same_count[v] - row_sums[v]
        gains = 2 * same_count - row_sums
        
        # Find the vertex with the maximum positive gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain <= 0:
            # No more improvements possible
            break
        
        # Flip the vertex
        v = max_gain_idx
        labels[v] = 1 - labels[v]
        current_cut += int(max_gain)
        
        # Note: We don't need to update same_count incrementally if we recompute
        # vectorized each time, as n is small. This simplifies logic and avoids bugs.
        
    return labels, current_cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=bool)
    row_sums = adj.sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple restarts with different initializations
    # 1. Spectral initialization (Laplacian eigenvector)
    # 2. Random initializations
    
    # --- Spectral Initialization ---
    # Compute Laplacian L = D - A
    # Smallest eigenvector of L gives a good initial cut
    try:
        L = np.diag(row_sums) - adj
        # Use scipy if available for efficiency, but numpy is enough for n=100
        # We want the eigenvector corresponding to the smallest non-zero eigenvalue
        # Or just the smallest eigenvalue (which is 0 for connected graph, but we want the
        # Fiedler vector, i.e., 2nd smallest).
        # Actually, for Max Cut, the spectral method often uses the eigenvector of
        # the adjacency matrix or Laplacian. Let's use the eigenvector of the Laplacian
        # corresponding to the smallest eigenvalue > 0.
        
        # For n=100, full eigendecomposition is fine.
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Find index of smallest positive eigenvalue
        # Due to numerical errors, 0 might be slightly negative or positive.
        # We want the Fiedler vector (2nd smallest eigenvalue).
        # Sort eigenvalues
        sorted_idx = np.argsort(eigvals)
        
        # Find the first eigenvalue that is significantly > 0
        # If the graph is connected, the first eigenvalue is 0.
        # Let's just take the eigenvector corresponding to the 2nd smallest eigenvalue.
        # If the graph is disconnected, this might not be optimal, but it's a good start.
        
        # Actually, a simpler spectral heuristic:
        # Use the eigenvector of the adjacency matrix corresponding to the largest eigenvalue?
        # No, for Max Cut, the Laplacian is standard.
        
        # Let's try the eigenvector of the Laplacian for the smallest non-zero eigenvalue.
        # If the graph is connected, eigvals[0] ~ 0. So we use eigvecs[:, sorted_idx[1]].
        
        if n > 1:
            # Use the 2nd smallest eigenvalue's eigenvector
            fiedler_vec = eigvecs[:, sorted_idx[1]]
            # Threshold at median or mean
            threshold = np.median(fiedler_vec)
            init_labels = (fiedler_vec > threshold).astype(int)
        else:
            init_labels = np.zeros(n, dtype=int)
            
        labels, cut = _local_search(init_labels, adj, row_sums, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    except Exception:
        pass # Fallback to random if spectral fails
    
    # --- Random Restarts ---
    # Use remaining time for random restarts
    while time.time() < deadline:
        # Random initialization
        init_labels = np.random.randint(0, 2, n)
        
        labels, cut = _local_search(init_labels, adj, row_sums, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we have plenty of time, continue.
        # If we are close to deadline, break.
        # The loop condition handles this.
        
        # To avoid infinite loop if local search is instant, we just let it run.
        # For n=100, local search is very fast (microseconds to milliseconds).
        # So we can do many restarts.
        
        # Optional: Add a small sleep if we are doing too many? No, better to use CPU.
        
    if best_labels is None:
        # Fallback: return alternating labels
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
