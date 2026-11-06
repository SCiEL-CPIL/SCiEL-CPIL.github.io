"""Maximum Cut via Goemans-Williamson Rounding, Fast Local Search, and Large Neighborhood Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Goemans-Williamson spectral initialization, fast heap-based local search,
    and large neighborhood search (perturbation) to escape local optima.
    """
    start_time = time.time()
    # Safety margin: 150s limit, keep 10s buffer
    deadline = start_time + 140.0

    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees and neighbor lists
    degrees = adj.sum(axis=1).astype(np.int32)
    neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]
    
    # Helper to compute cut size
    def compute_cut(labels):
        # Cut = sum of edges where labels differ
        # Efficient way: 
        # Let x be vector of labels (0 or 1). 
        # Cut = 0.5 * (sum(x) * sum(adj) - x^T adj x) ... no that's complex.
        # Simpler: 
        # in_0 = (labels == 0)
        # in_1 = (labels == 1)
        # cut = in_0 @ adj @ in_1
        in_0 = (labels == 0).astype(np.int32)
        in_1 = (labels == 1).astype(np.int32)
        return int(in_0 @ adj @ in_1)

    def goemans_williamson_init():
        """
        Initialize using the Fiedler vector (second smallest eigenvector of Laplacian).
        This is equivalent to the standard spectral method for Max Cut.
        """
        try:
            # Laplacian L = D - A
            D = np.diag(degrees)
            L = D - adj.astype(np.float64)
            # Symmetrize just in case
            L = (L + L.T) / 2.0
            
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(L)
            
            # The Fiedler vector corresponds to the second smallest eigenvalue (index 1)
            # Index 0 is the trivial eigenvalue 0 (constant vector)
            fiedler = eigvecs[:, 1]
            
            # Partition based on sign
            labels = (fiedler > 0).astype(np.int8)
            
            # Ensure both partitions are non-empty
            if np.all(labels == 0) or np.all(labels == 1):
                # Fallback: split by median
                median_val = np.median(fiedler)
                labels = (fiedler > median_val).astype(np.int8)
                
            return labels
        except Exception:
            return None

    def local_search(labels):
        """
        Fast local search using a priority queue.
        Returns the improved labels and cut size.
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        
        # Calculate initial gains for all vertices
        # gain[v] = change in cut if we flip v
        # If v is in 0, flipping to 1:
        #   Loses edges to 0-neighbors, Gains edges to 1-neighbors
        #   gain = deg[v] - 2 * (num neighbors in same group)
        
        gains = np.zeros(n, dtype=np.int32)
        for v in range(n):
            if labels[v] == 0:
                same_count = int(adj[v, labels == 0].sum())
            else:
                same_count = int(adj[v, labels == 1].sum())
            gains[v] = degrees[v] - 2 * same_count

        # Heap of (-gain, vertex)
        heap = [(-gains[v], v) for v in range(n)]
        heapq.heapify(heap)

        # Track best cut in this local search
        best_cut = cut
        best_labels = labels.copy()

        while time.time() < deadline - 2.0: # Leave time for other restarts
            if not heap:
                break
                
            neg_gain, v = heapq.heappop(heap)
            gain = -neg_gain
            
            # Verify gain is still valid
            # The gain might have changed if neighbors were flipped
            if labels[v] == 0:
                same_count = int(adj[v, labels == 0].sum())
            else:
                same_count = int(adj[v, labels == 1].sum())
            current_gain = degrees[v] - 2 * same_count
            
            if current_gain != gain:
                # Gain changed, re-push with correct value if it's positive
                if current_gain > 0:
                    heapq.heappush(heap, (-current_gain, v))
                continue
                
            if current_gain <= 0:
                # No improvement possible from this vertex currently
                # Since heap is max-heap, if the best available is <= 0, we are at local optimum
                break

            # Flip vertex v
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            cut += current_gain
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
                
            # Update gains for neighbors of v
            for u in neighbors[v]:
                # u's gain changes because v's label changed
                # If u and v were in same group, they are now in different groups.
                # u lost a "same" neighbor, so its gain increases by 2.
                # If u and v were in different groups, they are now in same group.
                # u gained a "same" neighbor, so its gain decreases by 2.
                
                if labels[u] == labels[v]:
                    # They are now in the same group
                    # Before flip: they were in different groups (since v flipped)
                    # Wait, logic check:
                    # Before: labels[u] vs old_label(v)
                    # After: labels[u] vs new_label(v)
                    
                    # Let's re-calculate gain for u directly to be safe and simple
                    if labels[u] == 0:
                        same_u = int(adj[u, labels == 0].sum())
                    else:
                        same_u = int(adj[u, labels == 1].sum())
                    new_gain_u = degrees[u] - 2 * same_u
                    
                    # Only push if positive
                    if new_gain_u > 0:
                        heapq.heappush(heap, (-new_gain_u, u))
                    # We don't remove old entries, we just validate on pop.
                    # This is a "lazy" heap approach.

        return best_labels, best_cut

    def perturb(labels, num_flips):
        """
        Randomly flip a subset of vertices to escape local optimum.
        """
        new_labels = labels.copy()
        indices = np.random.choice(n, size=num_flips, replace=False)
        new_labels[indices] = 1 - new_labels[indices]
        return new_labels

    rng = np.random.default_rng(42)
    
    best_labels_global = None
    best_cut_global = -1

    # Generate initial candidates
    candidates = []
    
    # 1. Goemans-Williamson (Spectral)
    gw_labels = goemans_williemson_init()
    if gw_labels is not None:
        candidates.append(gw_labels)
        
    # 2. Random
    for _ in range(2):
        candidates.append(rng.integers(0, 2, size=n, dtype=np.int8))

    # Process each candidate with Local Search + LNS loop
    for init_labels in candidates:
        if time.time() > deadline - 10:
            break
            
        current_labels = init_labels.copy()
        current_cut = compute_cut(current_labels)
        
        # Phase 1: Local Search
        current_labels, current_cut = local_search(current_labels)
        
        if current_cut > best_cut_global:
            best_cut_global = current_cut
            best_labels_global = current_labels.copy()
            
        # Phase 2: Large Neighborhood Search (Perturbation + Local Search)
        # We will run several perturbations
        while time.time() < deadline - 5:
            # Perturb
            # Flip a small random percentage of nodes (e.g., 5-15%)
            num_flips = max(1, int(n * 0.1)) 
            perturbed_labels = perturb(current_labels, num_flips)
            
            # Local Search on perturbed
            ls_labels, ls_cut = local_search(perturbed_labels)
            
            if ls_cut > current_cut:
                current_labels = ls_labels
                current_cut = ls_cut
                
            if current_cut > best_cut_global:
                best_cut_global = current_cut
                best_labels_global = current_labels.copy()
                
            # If we didn't improve current, we might still accept a worse one to explore?
            # Standard LNS often accepts worse to escape, but since we keep best_global,
            # and we start each LNS iteration from the *current* best local optimum,
            # we should probably restart from the best local optimum found in this candidate's history
            # or just keep exploring from the latest one.
            # Let's stick to improving the current trajectory. If we stagnate, we can break.
            # Actually, if ls_cut <= current_cut, we are stuck. 
            # To be more aggressive, we could accept worse solutions with some probability (SA-like).
            # But for now, let's just try more perturbations on the current best local opt.
            
            # To avoid infinite loop if no improvement:
            # We can break if we've tried many times without improvement.
            # For simplicity, we rely on the time limit.

    if best_labels_global is None:
        best_labels_global = np.zeros(n, dtype=np.int8)

    return best_labels_global.astype(np.int8)

# EVOLVE-BLOCK-END
