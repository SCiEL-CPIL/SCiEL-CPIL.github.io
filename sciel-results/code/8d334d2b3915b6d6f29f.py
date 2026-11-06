"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Helper to calculate cut size
    def _cut_size(labels):
        # Vectorized cut size calculation
        # diff[i, j] is 1 if labels[i] != labels[j]
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(adj * diff) // 2)
    
    # Helper for 1-flip local search (vectorized)
    def _local_search_1flip(labels):
        # labels is 0/1 array
        # Gain for flipping vertex v:
        # If labels[v] == 0:
        #   Current contribution to cut from v: edges to 1s (adj[v, labels==1].sum())
        #   New contribution: edges to 0s (adj[v, labels==0].sum())
        #   Gain = new - old = adj[v, labels==0].sum() - adj[v, labels==1].sum()
        # If labels[v] == 1:
        #   Current contribution: edges to 0s
        #   New contribution: edges to 1s
        #   Gain = adj[v, labels==1].sum() - adj[v, labels==0].sum()
        
        # We can compute this uniformly:
        # Let S0 = sum of adj[v, labels==0], S1 = sum of adj[v, labels==1]
        # If label is 0, gain = S0 - S1
        # If label is 1, gain = S1 - S0
        # Note: S0 + S1 = degree[v]
        # So gain = 2*S0 - degree[v] if label is 0, and degree[v] - 2*S0 if label is 1?
        # Actually, simpler:
        # gain = (2 * adj[v, labels == (1 - labels[v])].sum()) - degree[v]
        # Wait, let's derive carefully.
        # Cut size C = sum_{i<j, labels[i]!=labels[j]} adj[i,j]
        # Change in C when flipping v:
        # Edges (v, u) where labels[u] == labels[v] were NOT in cut, now ARE in cut. (+1 for each)
        # Edges (v, u) where labels[u] != labels[v] WERE in cut, now are NOT in cut. (-1 for each)
        # Delta = count_same - count_diff
        # count_same = adj[v, labels == labels[v]].sum()
        # count_diff = adj[v, labels != labels[v]].sum()
        # Delta = count_same - count_diff
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Vectorized calculation of gains for all vertices
            # same_counts[v] = adj[v, labels == labels[v]].sum()
            # diff_counts[v] = adj[v, labels != labels[v]].sum()
            
            # Efficiently compute counts
            # For each vertex v, we want sum of adj[v, u] where labels[u] == labels[v]
            # This is equivalent to:
            # sum_same = np.sum(adj * (labels[:, None] == labels[None, :]), axis=1)
            # sum_diff = np.sum(adj * (labels[:, None] != labels[None, :]), axis=1)
            # But this is O(n^2) per iteration. With n=100, it's 10,000 ops, very fast.
            
            mask_same = (labels[:, None] == labels[None, :]).astype(np.int8)
            mask_diff = (labels[:, None] != labels[None, :]).astype(np.int8)
            
            sum_same = np.sum(adj * mask_same, axis=1)
            sum_diff = np.sum(adj * mask_diff, axis=1)
            
            gains = sum_same - sum_diff
            
            # Find vertices with positive gain
            flip_mask = gains > 0
            if np.any(flip_mask):
                labels[flip_mask] = 1 - labels[flip_mask]
                improved = True
        return labels

    # Helper for 2-flip local search (optimized)
    # Gain of flipping i and j simultaneously (assuming labels[i] != labels[j]):
    # Let L_i, L_j be labels.
    # Change in cut:
    # 1. Edge (i, j): was in cut (if diff), now not in cut (if same). Delta = -1 if adj[i,j]
    # 2. Edge (i, u) for u != i, j:
    #    If labels[u] == L_i: was not in cut, now is (since L_i flips to 1-L_i, which is L_j? No, L_i flips to 1-L_i. If L_i=0, L_j=1. 1-L_i=1=L_j. So u (0) vs i (1) -> cut. u (0) vs j (1) -> cut.
    #    This is getting complicated. Let's use the gain formula.
    #    Gain(i) = sum_same_i - sum_diff_i
    #    Gain(j) = sum_same_j - sum_diff_j
    #    But they interact.
    #    The interaction term is:
    #    If we flip i, the gain for j changes because i's label changed.
    #    Simpler approach: Calculate exact cut change for pair (i, j).
    #    This is O(n) per pair if we precompute sums.
    #    With n=100, n^2 pairs = 10,000 pairs. O(n) per pair -> 1,000,000 ops. Very fast.
    
    def _local_search_2flip(labels):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Precompute sums for efficiency?
            # For each pair (i, j) with labels[i] != labels[j]:
            # Calculate gain of flipping both.
            # Let's just iterate and check.
            
            # To speed up, we can precompute the "current cut contribution" vectors.
            # But for n=100, brute force O(n^3) might be too slow if many iterations.
            # O(n^2 * n) = 10^6 per pass. If 100 passes, 10^8. Python is slow.
            # Vectorized 2-flip is better.
            
            # Let's stick to 1-flip first, then a simpler 2-flip if time permits.
            # Or use a randomized 2-flip.
            
            # Actually, 1-flip is very effective. Let's do more 1-flip passes with random restarts.
            pass
        return labels

    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple restarts with spectral init and random init, followed by 1-flip local search.
    # Since 1-flip is fast, we can do many restarts.
    
    # 1. Spectral Initialization
    try:
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        init_labels = (fiedler > 0).astype(np.int8)
        
        labels = _local_search_1flip(init_labels.copy())
        cut = _cut_size(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    except Exception:
        pass
    
    # 2. Random Restarts
    # Determine number of restarts based on time
    start_time = time.time()
    num_restarts = 0
    while time.time() < deadline - 5.0: # Leave 5s for final check
        # Random initialization
        labels = np.random.randint(0, 2, n).astype(np.int8)
        
        # Local search
        labels = _local_search_1flip(labels)
        
        cut = _cut_size(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        num_restarts += 1
        
        # If we are doing very fast restarts, maybe add a small perturbation to best solution?
        # Or just keep going.
        
        # Check time frequently
        if time.time() > deadline - 2.0:
            break

    # Final Polish: Try 2-flip on the best solution found
    if best_labels is not None and time.time() < deadline - 1.0:
        # Implement a simple 2-flip improvement for the best solution
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Iterate over pairs
            # To avoid O(n^3), we can vectorize the gain calculation for all pairs at once?
            # Gain(i, j) = Gain(i) + Gain(j) + 2 * adj[i, j] * (something?)
            # Let's just do a few passes of random pair swaps or systematic if fast enough.
            # For n=100, systematic O(n^2) pairs, O(1) gain check if precomputed?
            
            # Precompute row sums relative to current labels
            # This is complex. Given n=100, a simple nested loop with early exit might suffice for a few iterations.
            for i in range(n):
                for j in range(i + 1, n):
                    if best_labels[i] == best_labels[j]:
                        continue
                    # Calculate gain of flipping i and j
                    # Temporarily flip
                    old_l_i = best_labels[i]
                    old_l_j = best_labels[j]
                    best_labels[i] = 1 - old_l_i
                    best_labels[j] = 1 - old_l_j
                    
                    new_cut = _cut_size(best_labels)
                    if new_cut > best_cut:
                        best_cut = new_cut
                        improved = True
                    else:
                        # Revert
                        best_labels[i] = old_l_i
                        best_labels[j] = old_l_j
            
            # If we did a full pass and didn't improve, stop
            # Note: The loop above reverts if not improved, so best_labels remains at best_cut state
            # If improved, we kept the change.
            # This is O(n^2 * n) per pass due to _cut_size being O(n^2).
            # 100^2 * 100^2 = 10^8 ops per pass. Too slow for Python in 1s.
            # So this 2-flip polish is likely too slow to complete many passes.
            # We will skip the full 2-flip polish if it's too slow, relying on 1-flip restarts.
            # However, a single pass might be doable if optimized?
            # _cut_size is vectorized O(n^2). 10^4 ops.
            # 5000 pairs * 10^4 ops = 5 * 10^7 ops.
            # In Python, 5*10^7 simple ops might take ~1-5 seconds.
            # So we can probably afford 1-2 passes of 2-flip at the end.
            
            if time.time() > deadline - 0.5:
                break

    return best_labels

# EVOLVE-BLOCK-END
