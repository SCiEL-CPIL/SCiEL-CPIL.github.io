# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    # labels are 0/1
    # An edge (i,j) crosses if labels[i] != labels[j]
    # We can compute: for each i, sum adj[i, j] where labels[j] != labels[i]
    # But we need to count each edge once. Since adj is symmetric (presumably) or we count directed?
    # The task says "unweighted 100-node graphs". Usually adjacency matrix is symmetric for undirected.
    # "number of edges crossing" implies undirected edges.
    # If adj is symmetric, sum_{i,j} adj[i,j] * (labels[i] != labels[j]) counts each edge twice.
    # Let's check the seed program. It uses `adjacency[v][labels == labels[v]].sum()`.
    # This counts neighbors.
    # Let's compute cut size as: sum_{i} sum_{j > i} adj[i,j] * (labels[i] != labels[j])
    # Or simply: 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j]) if symmetric.
    # To be safe and fast for n=100:
    # diff = (labels[:, None] != labels[None, :]).astype(np.float64)
    # return np.sum(adj * diff) / 2.0  # if symmetric
    # Actually, let's just do the direct sum to avoid assuming symmetry if not guaranteed, 
    # but standard graph problems usually imply symmetric adjacency for undirected.
    # The seed program treats it as: for v, same = adj[v, labels==labels[v]].sum()
    # This counts the number of neighbors in the same set.
    # The cut size is total edges - edges inside sets.
    # Let's use the vectorized formula:
    diff = (labels[:, None] != labels[None, :]).astype(np.float64)
    # If the graph is undirected, adj is symmetric. The number of crossing edges is 
    # sum_{i<j} adj[i,j] * diff[i,j] = 0.5 * sum_{i,j} adj[i,j] * diff[i,j]
    # Let's assume symmetry.
    return np.sum(adj * diff) / 2.0


def _spectral_init(adj, n):
    """Generate initial partition using spectral methods (Fiedler vector)."""
    # Convert to sparse for faster eigendecomposition if needed, but n=100 is small.
    # Use dense numpy eig for simplicity and reliability on small n.
    # The Laplacian L = D - A.
    # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue.
    # We can also use the leading eigenvector of (A - I) or similar, but Fiedler is standard for Max Cut.
    # Actually, for Max Cut, the spectral method often uses the leading eigenvector of the adjacency matrix
    # or the Laplacian. Let's use the Fiedler vector of the Laplacian.
    # However, a simpler and often effective heuristic is to use the leading eigenvector of the adjacency matrix.
    # Let's try the Fiedler vector approach first.
    
    deg = np.sum(adj, axis=1)
    D = np.diag(deg)
    L = D - adj.astype(np.float64)
    
    # Compute eigenvalues and eigenvectors
    # For n=100, full eigendecomposition is fast.
    try:
        evals, evecs = np.linalg.eigh(L)
        # evecs are columns. Sort by eigenvalues.
        # eigh returns in ascending order.
        # The second smallest eigenvalue is index 1.
        # The corresponding eigenvector is evecs[:, 1].
        fiedler = evecs[:, 1]
    except np.linalg.LinAlgError:
        # Fallback to random
        return np.random.randint(0, 2, n)
    
    # Threshold the Fiedler vector to get 0/1 labels.
    # Median threshold is common.
    median = np.median(fiedler)
    labels = (fiedler > median).astype(int)
    
    # Ensure both partitions are non-empty (if possible)
    if np.all(labels == 0) or np.all(labels == 1):
        # Fallback: split in half
        labels[:n//2] = 0
        labels[n//2:] = 1
        
    return labels


def _greedy_refine(adj, labels, max_iterations=10):
    """
    Greedy local improvement (flip a node if it increases the cut).
    Returns improved labels and the final cut size.
    """
    n = len(labels)
    labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    
    for _ in range(max_iterations):
        improved = False
        # To make it more robust, we can iterate until no improvement or max iterations.
        # For n=100, this is very fast.
        for v in range(n):
            # Calculate the gain of flipping v
            # Current state: v is in group labels[v]
            # Neighbors in same group: same_count
            # Neighbors in other group: other_count
            # If we flip v, the cut changes by:
            # - other_count (edges that were crossing now become internal)
            # + same_count (edges that were internal now become crossing)
            # Gain = same_count - other_count
            # If gain > 0, flipping improves the cut.
            
            same_mask = (labels == labels[v])
            other_mask = (labels != labels[v])
            
            same_count = adj[v][same_mask].sum()
            other_count = adj[v][other_mask].sum()
            
            gain = same_count - other_count
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                best_cut += gain
                improved = True
        if not improved:
            break
            
    return labels, best_cut


def _sdp_init(adj, n, time_limit=2.0):
    """
    Attempt SDP-based initialization.
    This is computationally heavier. We use it if time permits.
    For n=100, a simple SDP solver might be too slow or complex to implement from scratch.
    Instead, we rely on spectral methods and multiple restarts.
    We can simulate diversity by perturbing the spectral vector or using random hyperplanes
    on a low-dimensional embedding.
    """
    # For now, we'll use spectral init with random perturbations for diversity.
    return _spectral_init(adj, n)


def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit
    
    adj = np.array(adjacency, dtype=np.float64)
    n = len(adj)
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral initialization
    labels_spec = _spectral_init(adj, n)
    labels_refined, cut_val = _greedy_refine(adj, labels_spec)
    best_labels = labels_refined
    best_cut = cut_val
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return best_labels
    
    # 2. Multiple random restarts with greedy refinement
    # We can also try random initializations.
    num_restarts = 20  # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        labels_rand = np.random.randint(0, 2, n)
        # Optionally, bias towards spectral solution?
        # Or just pure random.
        
        labels_ref, cut_val = _greedy_refine(adj, labels_rand)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels_ref
    
    # 3. More sophisticated local search: 2-opt or similar?
    # For now, greedy flip is O(n^2) per pass. 
    # We can try multiple passes or different orderings.
    
    # Let's try a few more restarts if time allows, focusing on perturbed spectral solutions.
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        # Try perturbing the spectral solution
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            # Flip a random subset of nodes in the spectral solution
            labels_pert = best_labels.copy()
            # Flip 5-10% of nodes
            num_flips = max(1, int(n * 0.1))
            flip_indices = np.random.choice(n, num_flips, replace=False)
            labels_pert[flip_indices] = 1 - labels_pert[flip_indices]
            
            labels_ref, cut_val = _greedy_refine(adj, labels_pert)
            if cut_val > best_cut:
                best_cut = cut_val
                best_labels = labels_ref

    return best_labels
# EVOLVE-BLOCK-END
