"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        # labels: 0/1 int array
        # cut = 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
        # Equivalent: sum over i<j
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def tabu_search(initial_labels, tabu_length=10, max_iterations=10000):
        labels = initial_labels.copy()
        current_cut = compute_cut(labels)
        
        # Precompute degrees for O(1) gain calculation
        # gain(v) = 2 * (sum_neighbors_other - sum_neighbors_same)
        # sum_neighbors_other[v] = sum of adj[v, u] where labels[u] != labels[v]
        # sum_neighbors_same[v] = sum of adj[v, u] where labels[u] == labels[v]
        
        # Initialize gains
        gains = np.zeros(n, dtype=np.int32)
        for v in range(n):
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            gains[v] = 2 * (other - same)

        tabu_until = np.zeros(n, dtype=np.int64)
        current_iter = 0
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        # Diversification counter
        no_improve_count = 0
        max_no_improve = 1000
        
        while current_iter < max_iterations:
            if time.time() > deadline:
                break
                
            # Select best non-tabu move
            best_gain = -np.inf
            best_v = -1
            best_is_tabu = True
            
            # Vectorized selection of best gain, but we need to respect tabu
            # For efficiency, we can just iterate or use a heap.
            # n=100, so iterating is fast enough (100 ops).
            # To make it faster, we can pre-filter non-tabu nodes.
            non_tabu_mask = tabu_until <= current_iter
            if not np.any(non_tabu_mask):
                # All nodes are tabu, pick best overall (aspiration)
                best_v = np.argmax(gains)
                best_gain = gains[best_v]
            else:
                # Mask out tabu nodes
                gains_masked = np.where(non_tabu_mask, gains, -np.inf)
                best_v = np.argmax(gains_masked)
                best_gain = gains[best_v]
                
            if best_gain == -np.inf:
                break

            # Apply move
            v = best_v
            old_label = labels[v]
            labels[v] = 1 - old_label
            current_cut += best_gain // 2
            
            # Update tabu list
            tabu_until[v] = current_iter + tabu_length
            
            # Update gains for v and its neighbors
            # For v: gain changes sign? No, gain is calculated based on neighbors.
            # If we flip v, its neighbors' gains change, and v's gain changes.
            
            # Neighbors of v
            neighbors = np.where(adj[v] == 1)[0]
            
            # Update gains for neighbors
            # For a neighbor u of v:
            # Before flip: v had label L. u had label U.
            # If U == L, v contributed to same_count(u). Now v has 1-L, so v contributes to other_count(u).
            # Gain(u) = 2 * (other - same).
            # Change in other_count(u) = +1 (if v was same, now other) or -1 (if v was other, now same)?
            # Let's re-derive.
            # gain(u) = 2 * (sum_adj_other(u) - sum_adj_same(u))
            # When v flips:
            # If labels[u] == old_label(v):
            #   Before: v was same for u. After: v is other for u.
            #   sum_same(u) decreases by 1, sum_other(u) increases by 1.
            #   gain(u) changes by 2 * (1 - (-1)) = 4.
            # If labels[u] != old_label(v):
            #   Before: v was other for u. After: v is same for u.
            #   sum_other(u) decreases by 1, sum_same(u) increases by 1.
            #   gain(u) changes by 2 * (-1 - 1) = -4.
            
            same_as_old = (labels[neighbors] == old_label)
            gains[neighbors[same_as_old]] += 4
            gains[neighbors[~same_as_old]] -= 4
            
            # Update gain for v itself
            # v's label is now 1 - old_label.
            # Its neighbors are the same set.
            # The relative configuration of its neighbors hasn't changed, only v's label.
            # gain(v) = 2 * (sum_adj_other(v) - sum_adj_same(v))
            # Before flip: sum_same_old, sum_other_old. gain_old = 2*(sum_other_old - sum_same_old)
            # After flip: v is now 1-old_label.
            # Neighbors that were same (label old_label) are now other.
            # Neighbors that were other (label 1-old_label) are now same.
            # So new_sum_same = old_sum_other
            # new_sum_other = old_sum_same
            # new_gain = 2 * (old_sum_same - old_sum_other) = -old_gain
            # So gains[v] = -gains[v]
            gains[v] = -gains[v]

            current_iter += 1
            
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                break
                
        return best_local_labels, best_local_cut

    # Strategy: Multiple restarts with Tabu Search
    # 1. Spectral initialization
    try:
        # Use scipy for spectral if available, else random
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        A = csr_matrix(adj)
        # Laplacian L = D - A
        D = np.array(A.sum(axis=1)).flatten()
        L = csr_matrix((D[:, None] - adj), shape=(n, n))
        
        # Smallest eigenvalue is 0, next is Fiedler vector
        # eigsh finds eigenvalues in the middle of the spectrum by default
        # We want the smallest non-zero eigenvalue.
        # Use which='SM' might be slow. Use shift-invert or just 'SA' (smallest algebraic)?
        # Actually, for connected graphs, lambda_0=0, lambda_1>0.
        # eigsh(L, k=2, which='SA') might return 0 and lambda_1.
        
        vals, vecs = eigsh(L, k=2, which='SA')
        # Sort by value
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        # Fiedler vector is the second smallest eigenvalue's eigenvector
        fiedler = vecs[:, 1]
        
        # Partition based on sign of Fiedler vector
        labels = (fiedler > 0).astype(np.int8)
        # Ensure balance? No, max cut doesn't require balance, but spectral usually gives near-balance.
        # If all same, flip some
        if np.all(labels == 0) or np.all(labels == 1):
             labels = np.random.randint(0, 2, size=n)
             
        tabu_labels, tabu_cut = tabu_search(labels, tabu_length=10, max_iterations=5000)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
            
    except Exception:
        pass

    # 2. Random restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        labels = np.random.randint(0, 2, size=n)
        
        # Adaptive tabu length? Start small, increase if stuck?
        # Or just fixed.
        tabu_len = 10 + (i % 10)
        
        tabu_labels, tabu_cut = tabu_search(labels, tabu_length=tabu_len, max_iterations=3000)
        
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()

    # 3. Final polish with 1-flip local search on the best found so far
    if best_labels is not None and time.time() < deadline:
        labels = best_labels.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order
            order = list(range(n))
            np.random.shuffle(order)
            for v in order:
                if time.time() > deadline:
                    break
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        final_cut = compute_cut(labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = labels

    if best_labels is None:
        return np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
