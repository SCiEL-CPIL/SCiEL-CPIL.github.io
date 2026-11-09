"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    # Safety margin: The limit is 150s, we stop at 145s to ensure return.
    deadline = start_time + 145.0
    
    # Convert to numpy array for vectorized operations
    # bool matrix -> int8 for fast summing
    adj = np.array(adjacency, dtype=np.int8)
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        # Cut size = sum of edges where labels differ
        # labels is 0/1 array.
        # Edge (i,j) contributes if adj[i,j]==1 and labels[i] != labels[j]
        # This is equivalent to 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
        # But since we want integer count of undirected edges, we can do:
        # sum over i<j where adj[i,j]=1 and labels[i]!=labels[j]
        # Vectorized: 
        # diff = labels[:, None] != labels[None, :]
        # cut = np.sum(adj * diff) // 2  (since symmetric)
        # However, creating n x n diff matrix is O(n^2) memory and time.
        # For n=100, 100x100 is 10k elements, very fast.
        diff = labels[:, None] != labels[None, :]
        return np.sum(adj * diff) // 2

    def greedy_improve_2swap(labels):
        # Local search using pairwise swaps (2-flips)
        # This allows escaping local minima of 1-flip search.
        # We iterate until no improvement is found in a full pass.
        improved = True
        # Shuffle order to avoid bias
        order = list(range(n))
        
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(order)
            for i in order:
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    # Try swapping labels[i] and labels[j]
                    # Only useful if labels[i] != labels[j]
                    if labels[i] == labels[j]:
                        continue
                    
                    # Calculate gain of swapping i and j
                    # Current cut contribution involving i or j
                    # New cut contribution involving i or j
                    # Let A be adjacency.
                    # Let L be labels.
                    # Swap L[i], L[j].
                    # Change in cut = 
                    #   - (edges from i to same-side + edges from j to same-side) 
                    #   + (edges from i to other-side + edges from j to other-side)
                    #   - 2 * edge(i,j) if it was crossing? No.
                    
                    # Let's just compute the cut before and after for the affected nodes?
                    # Or use the formula:
                    # gain = 2 * (adj[i,j] * (1 if labels[i]!=labels[j] else -1) ... )
                    # Actually, simpler to just compute the cut size for the whole graph
                    # if n is small (100), computing full cut is O(n^2) = 10k ops.
                    # Doing this for every pair (100*100/2 = 5000 pairs) in a full pass
                    # means 5000 * 10k = 50M ops per pass.
                    # In Python/NumPy, this might be slow if done in pure Python loops.
                    # We need a faster gain calculation.
                    
                    # Gain calculation for swapping i and j:
                    # Let S_i = set of neighbors of i with same label as i.
                    # Let S_j = set of neighbors of j with same label as j.
                    # Let D_i = set of neighbors of i with diff label from i.
                    # Let D_j = set of neighbors of j with diff label from j.
                    
                    # Before swap:
                    # i is connected to S_i (not cut) and D_i (cut).
                    # j is connected to S_j (not cut) and D_j (cut).
                    # Edge (i,j): if labels diff, it is cut.
                    
                    # After swap:
                    # i takes label of j, j takes label of i.
                    # i is now connected to:
                    #   - Neighbors with label j: these were in D_i (if label j != label i) or S_i?
                    #     Wait. labels[i] != labels[j].
                    #     Neighbors of i with label == labels[j] are currently in D_i (diff from i).
                    #     After swap, i has label j. So these neighbors become "same" as i.
                    #     So edges i->(neighbors with label j) go from Cut to Not-Cut.
                    #     Count: |D_i ∩ {neighbors with label j}| = sum(adj[i, :][labels == labels[j]])
                    #   - Neighbors of i with label == labels[i]: currently in S_i.
                    #     After swap, i has label j. These become "diff" from i.
                    #     So edges i->(neighbors with label i) go from Not-Cut to Cut.
                    #     Count: |S_i| = sum(adj[i, :][labels == labels[i]])
                    
                    # Similarly for j.
                    
                    # Let c_i_j = 1 if adj[i,j] else 0.
                    # Let n_i_j = sum(adj[i, :][labels == labels[j]])  (neighbors of i with label j)
                    # Let n_i_i = sum(adj[i, :][labels == labels[i]])  (neighbors of i with label i)
                    # Let n_j_i = sum(adj[j, :][labels == labels[i]])  (neighbors of j with label i)
                    # Let n_j_j = sum(adj[j, :][labels == labels[j]])  (neighbors of j with label j)
                    
                    # Change in cut for edges incident to i:
                    # - n_i_j (were cut, now not)
                    # + n_i_i (were not cut, now cut)
                    # Note: edge (i,j) is included in n_i_j if adj[i,j]=1 and labels[i]!=labels[j].
                    # Since labels[i]!=labels[j], if adj[i,j]=1, it is in n_i_j.
                    
                    # Change in cut for edges incident to j:
                    # - n_j_i (were cut, now not)
                    # + n_j_j (were not cut, now cut)
                    # Edge (i,j) is included in n_j_i if adj[i,j]=1.
                    
                    # Total gain = (n_i_i - n_i_j) + (n_j_j - n_j_i)
                    # Wait, let's verify signs.
                    # Cut increases if we move from Not-Cut to Cut.
                    # i: moves from label i to label j.
                    #   Edges to label i neighbors: Not-Cut -> Cut. Gain: + n_i_i
                    #   Edges to label j neighbors: Cut -> Not-Cut. Gain: - n_i_j
                    # j: moves from label j to label i.
                    #   Edges to label j neighbors: Not-Cut -> Cut. Gain: + n_j_j
                    #   Edges to label i neighbors: Cut -> Not-Cut. Gain: - n_j_i
                    
                    # Total Gain = n_i_i + n_j_j - n_i_j - n_j_i
                    
                    # Note: If adj[i,j]=1, it is counted in n_i_j and n_j_i.
                    # So we subtract it twice.
                    # Before: Edge (i,j) was Cut (since labels diff).
                    # After: Edge (i,j) is Not-Cut (since labels swapped, still diff? No.
                    #   i becomes label j, j becomes label i.
                    #   labels[i]_new = labels[j]_old. labels[j]_new = labels[i]_old.
                    #   They are still different! So edge (i,j) is STILL Cut.
                    #   Wait. If labels are 0 and 1.
                    #   i=0, j=1. Swap -> i=1, j=0.
                    #   i != j still. So edge (i,j) remains cut.
                    #   So the status of edge (i,j) does NOT change.
                    #   But my formula:
                    #   n_i_j includes adj[i,j] if labels[j] is the "other" label. Yes.
                    #   n_j_i includes adj[i,j] if labels[i] is the "other" label. Yes.
                    #   So we subtract adj[i,j] twice.
                    #   But the actual change for edge (i,j) is 0.
                    #   So we should add 2*adj[i,j] back?
                    #   Let's re-evaluate.
                    
                    # Let's use the explicit change:
                    # Delta = (New Cut) - (Old Cut)
                    # Edges not incident to i or j: No change.
                    # Edge (i,j):
                    #   Old: Cut if labels[i]!=labels[j].
                    #   New: Cut if labels[j]!=labels[i].
                    #   Same. Delta = 0.
                    # Edges (i,k) where k!=j:
                    #   Old: Cut if labels[i]!=labels[k].
                    #   New: Cut if labels[j]!=labels[k].
                    #   Change: 
                    #     If labels[k]==labels[i]: Old Not-Cut, New Cut (if labels[j]!=labels[i]). +1
                    #     If labels[k]==labels[j]: Old Cut, New Not-Cut. -1
                    #     If labels[k]==other (none, binary): 
                    #   So for each k:
                    #     If labels[k]==labels[i]: +1 (since labels[j]!=labels[i])
                    #     If labels[k]==labels[j]: -1
                    #   Sum over k!=j:
                    #     (Count of neighbors of i with label i, excluding j) - (Count of neighbors of i with label j, excluding j)
                    #     = (n_i_i - adj[i,j]) - (n_i_j - adj[i,j])  -- wait, if adj[i,j]=1, j is a neighbor.
                    #     If labels[j] != labels[i], then j is in the "label j" set for i?
                    #     Yes. So n_i_j includes j if adj[i,j]=1.
                    #     n_i_i does not include j if adj[i,j]=1 and labels[j]!=labels[i].
                    #     So:
                    #     Sum_{k!=j} delta(i,k) = (n_i_i) - (n_i_j - adj[i,j])?
                    #     Let's check:
                    #     n_i_i = sum_{k: labels[k]=labels[i]} adj[i,k]. j is not in this set.
                    #     n_i_j = sum_{k: labels[k]=labels[j]} adj[i,k]. j IS in this set if adj[i,j]=1.
                    #     So count of neighbors with label i = n_i_i.
                    #     Count of neighbors with label j = n_i_j.
                    #     We want to exclude k=j.
                    #     j has label j. So it is in the "label j" group.
                    #     So neighbors with label i (excluding j) = n_i_i.
                    #     Neighbors with label j (excluding j) = n_i_j - adj[i,j].
                    #     Delta_i = n_i_i - (n_i_j - adj[i,j])
                    
                    # Similarly for j:
                    #     Delta_j = n_j_j - (n_j_i - adj[i,j])
                    
                    # Total Delta = Delta_i + Delta_j
                    #            = n_i_i - n_i_j + adj[i,j] + n_j_j - n_j_i + adj[i,j]
                    #            = n_i_i + n_j_j - n_i_j - n_j_i + 2*adj[i,j]
                    
                    # Let's test this logic.
                    # If we just compute the cut before and after for the subgraph induced by i,j and their neighbors?
                    # Given n=100, computing full cut is 10k ops.
                    # 5000 pairs * 10k ops = 50M ops per pass.
                    # NumPy vectorized full cut is very fast.
                    # Let's just do:
                    # old_cut = compute_cut(labels)
                    # labels[i], labels[j] = labels[j], labels[i]
                    # new_cut = compute_cut(labels)
                    # if new_cut > old_cut: accept
                    # else: revert
                    
                    # This is O(n^2) per pair. 5000 * 10000 = 50,000,000 operations.
                    # In Python, this loop is the bottleneck.
                    # We can optimize by precomputing row sums?
                    # Or just accept the cost if we limit passes.
                    # 145 seconds / (50M ops / 100M ops/s approx) = ~145 seconds for 1 pass?
                    # NumPy is C-speed. 50M simple int ops is < 0.1s.
                    # The Python loop overhead is 5000 iterations.
                    # 5000 * (2 compute_cut calls) = 10000 calls to compute_cut.
                    # compute_cut is ~10us? 10000 * 10us = 0.1s.
                    # So one full pass of 2-swap is ~0.1-0.5s.
                    # We can afford ~200-1000 passes.
                    
                    old_cut = compute_cut(labels)
                    labels[i], labels[j] = labels[j], labels[i]
                    new_cut = compute_cut(labels)
                    
                    if new_cut > old_cut:
                        improved = True
                    else:
                        labels[i], labels[j] = labels[j], labels[i]
                    
        return labels

    # Strategy:
    # 1. "Go-with-the-flow" initialization: 
    #    Start with random, then do 1-flip greedy. This is fast and gets to a local min.
    # 2. Use this as start for 2-swap local search.
    # 3. Also try pure random starts for 2-swap.
    
    # Pre-compute degrees? Not needed for compute_cut.
    
    # Restart 0: Go-with-the-flow
    if time.time() < deadline:
        labels = np.arange(n) % 2
        # Quick 1-flip greedy
        for _ in range(10): # Max 10 passes
            if time.time() > deadline: break
            changed = False
            for v in range(n):
                if time.time() > deadline: break
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed: break
        
        # Now 2-swap improve
        labels = greedy_improve_2swap(labels)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Restarts 1-19: Random init + 2-swap
    for restart in range(1, 20):
        if time.time() > deadline:
            break
        
        labels = np.random.randint(0, 2, size=n)
        
        # Optional: Quick 1-flip to get close to local min before 2-swap
        # This can speed up 2-swap convergence
        for _ in range(5): # Fewer passes for random starts
            if time.time() > deadline: break
            changed = False
            for v in range(n):
                if time.time() > deadline: break
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed: break
            
        # 2-swap improve
        labels = greedy_improve_2swap(labels)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
