# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1  # convert 0/1 to -1/+1
    # cut = 0.5 * (sum(A) - s^T A s)
    # But for undirected symmetric A with 0 diagonal:
    # sum(A) = 2*E
    # s^T A s = sum_{i,j} s_i A_ij s_j
    # For an edge (i,j): if s_i == s_j, contributes 2; if s_i != s_j, contributes -2
    # So s^T A s = 2*(#same) - 2*(#cut) = 2*(2E - cut) - 2*cut = 4E - 4*cut
    # Wait, let me just compute directly for safety and speed at n=100
    # Direct: count pairs where labels differ and adjacency is True
    diff = labels[:, None] != labels[None, :]
    return int(np.sum(adj & diff))


def _greedy_refine(adj, labels, max_passes=100, time_limit=None, start_time=None):
    """Greedy local improvement: flip vertex if it increases cut size."""
    n = len(labels)
    labels = labels.copy()
    # Precompute current cut size
    current_cut = _cut_size(adj, labels)
    
    # For efficiency, track for each vertex: number of neighbors on same side vs other side
    # same_count[v] = sum of adj[v, u] where labels[u] == labels[v]
    # other_count[v] = sum of adj[v, u] where labels[u] != labels[v]
    # Flipping v changes cut by: other_count[v] - same_count[v]
    # (we gain other_count edges crossing, lose same_count edges crossing)
    
    same_count = np.zeros(n, dtype=np.int64)
    other_count = np.zeros(n, dtype=np.int64)
    
    for v in range(n):
        same_count[v] = np.sum(adj[v] & (labels == labels[v]))
        other_count[v] = np.sum(adj[v] & (labels != labels[v]))
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time_limit is not None:
            if start_time is not None and (time.time() - start_time) > time_limit:
                break
        improved = False
        passes += 1
        for v in range(n):
            # Gain from flipping v
            gain = other_count[v] - same_count[v]
            if gain > 0:
                # Flip vertex v
                labels[v] = 1 - labels[v]
                current_cut += gain
                # Update counts for neighbors of v
                neighbors = np.where(adj[v])[0]
                for u in neighbors:
                    # Before flip: u and v were on same side (since gain>0 means other>same, 
                    # but actually we need to think about what changes)
                    # After flipping v, for each neighbor u:
                    # If u was on same side as v before, now u is on other side
                    # If u was on other side as v before, now u is on same side
                    if labels[u] == labels[v]:
                        # u and v are now on same side, so they were on different sides before
                        # For v: same_count increases by 1, other_count decreases by 1
                        # For u: same_count increases by 1, other_count decreases by 1
                        same_count[v] += 1
                        other_count[v] -= 1
                        same_count[u] += 1
                        other_count[u] -= 1
                    else:
                        # u and v are now on different sides, so they were on same side before
                        # For v: other_count increases by 1, same_count decreases by 1
                        # For u: other_count increases by 1, same_count decreases by 1
                        other_count[v] += 1
                        same_count[v] -= 1
                        other_count[u] += 1
                        same_count[u] -= 1
                improved = True
    
    return labels, current_cut


def _spectral_init(adj, n, rng):
    """Initialize using the leading eigenvector of the graph Laplacian."""
    # L = D - A
    degrees = adj.sum(axis=1).astype(np.float64)
    D = np.diag(degrees)
    L = D - adj.astype(np.float64)
    
    # Use numpy's eigensolver for the smallest eigenvalue (or largest of -L)
    # Actually, we want the Fiedler vector (2nd smallest eigenvalue of L)
    # But for max-cut, the leading eigenvector of (D - A) works well
    # Let's use the eigenvector corresponding to the largest eigenvalue of -L
    # Or equivalently, the smallest eigenvalue of L (which is 0 with eigenvector ones)
    # So we want the 2nd smallest eigenvalue
    
    try:
        # For n=100, full eigendecomposition is fine
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector of the 2nd smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Convert to 0/1 labels based on sign
        labels = (fiedler > 0).astype(np.int64)
        return labels
    except:
        # Fallback to random
        return rng.integers(0, 2, size=n)


def _random_init(n, rng):
    """Random 0/1 initialization."""
    return rng.integers(0, 2, size=n)


def partition(adjacency):
    """Max-Cut partition using spectral init + greedy refinement + restarts."""
    adj = np.asarray(adjacency, dtype=bool)
    n = len(adj)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    # First try spectral initialization
    labels_spec = _spectral_init(adj, n, rng)
    refined_labels, cut_val = _greedy_refine(adj, labels_spec, max_passes=200, 
                                              time_limit=10, start_time=start_time)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels
    
    # Multiple random restarts
    num_restarts = 50  # Will be limited by time
    for r in range(num_restarts):
        if time.time() > deadline - 5:  # Stop with 5s margin
            break
        
        # Random initialization
        labels_rand = _random_init(n, rng)
        
        # Greedy refinement with time limit
        time_remaining = deadline - 5 - time.time()
        if time_remaining <= 0:
            break
        
        refined_labels, cut_val = _greedy_refine(adj, labels_rand, max_passes=200,
                                                  time_limit=max(1.0, time_remaining),
                                                  start_time=time.time())
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
    
    # Final polish: one more greedy pass on the best solution
    if best_labels is not None and time.time() < deadline - 2:
        polished_labels, final_cut = _greedy_refine(adj, best_labels, max_passes=100,
                                                     time_limit=2, start_time=time.time())
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = polished_labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
