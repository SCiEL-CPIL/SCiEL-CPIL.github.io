"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Edge (i,j) crosses if labels[i] != labels[j]
    # For undirected graph, each edge counted once
    # cut = sum over all i<j where labels differ of adj[i,j]
    # Vectorized: for each i, count neighbors j where labels[j] != labels[i]
    # Then divide by 2 since each edge counted twice
    n = len(labels)
    # For each vertex i, count neighbors with different label
    # labels is 0/1, so different label = 1 - labels[i]
    # diff[i,j] = 1 if labels[i] != labels[j]
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    cut = (adj * diff).sum() // 2
    return cut


def _spectral_init(adj, n):
    """Initialize using Fiedler vector of the Laplacian."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the second smallest eigenvalue's eigenvector
        # eigvals are sorted in ascending order
        fiedler = eigvecs[:, 1]
        # Partition based on sign
        labels = (fiedler >= 0).astype(np.int8)
        # Ensure both labels are present
        if labels.min() == labels.max():
            labels[0] = 1 - labels[0]
        return labels
    except Exception:
        # Fallback to random
        return np.random.randint(0, 2, n)


def _local_search_1flip(adj, labels, deadline):
    """Greedy 1-flip local search with vectorized gain computation."""
    n = len(labels)
    labels = labels.copy()
    
    while True:
        if time.time() >= deadline:
            break
        
        # Compute gain for flipping each vertex
        # If vertex v is in group g, flipping to 1-g changes cut by:
        # gain = (edges to other group) - (edges to same group)
        # = 2 * (edges to other group) - degree(v)
        
        # Vectorized computation
        # same_count[v] = number of neighbors of v with same label
        # other_count[v] = number of neighbors of v with different label
        # gain[v] = other_count[v] - same_count[v]
        
        # For vertex v with label l:
        # same_count[v] = sum_j adj[v,j] * (labels[j] == l)
        # other_count[v] = sum_j adj[v,j] * (labels[j] != l)
        
        # Using matrix operations:
        # Let's compute for all vertices at once
        # same_mask[v,j] = 1 if labels[v] == labels[j]
        same_mask = (labels[:, None] == labels[None, :]).astype(np.int8)
        diff_mask = 1 - same_mask
        
        same_count = (adj * same_mask).sum(axis=1)
        other_count = (adj * diff_mask).sum(axis=1)
        
        gain = other_count - same_count
        
        # Find the vertex with maximum positive gain
        max_gain = gain.max()
        if max_gain <= 0:
            break
        
        # Flip the vertex with max gain (if tie, pick first)
        v = gain.argmax()
        labels[v] = 1 - labels[v]
    
    return labels


def _local_search_2flip(adj, labels, deadline):
    """2-flip local search to escape 1-flip local optima."""
    n = len(labels)
    labels = labels.copy()
    
    # First do 1-flip to converge
    labels = _local_search_1flip(adj, labels, deadline)
    
    # Now try 2-flip moves
    # For a pair (i, j), the gain of flipping both simultaneously
    # We need to compute the change in cut size
    
    improved = True
    while improved:
        if time.time() >= deadline:
            break
        
        improved = False
        
        # Compute current cut size
        cur_cut = _cut_size(adj, labels)
        
        # Try all pairs? That's O(n^2) pairs, each requiring O(n) work = O(n^3)
        # For n=100, that's 10^6 * 100 = 10^8, might be too slow
        # Let's be smarter: only try pairs that are in different groups
        # or use a heuristic subset
        
        # Actually, let's just try a limited number of random pairs
        # or use a smarter approach
        
        # Compute gain for flipping vertex i alone
        same_mask = (labels[:, None] == labels[None, :]).astype(np.int8)
        diff_mask = 1 - same_mask
        same_count = (adj * same_mask).sum(axis=1)
        other_count = (adj * diff_mask).sum(axis=1)
        gain_1flip = other_count - same_count
        
        # For 2-flip of (i, j):
        # The interaction term matters. Let's compute it properly.
        # When we flip both i and j:
        # - Edges from i to others: gain changes based on whether others are flipped
        # - Edges from j to others: similar
        # - Edge (i,j): if it exists, its contribution changes
        
        # Let's compute the 2-flip gain analytically
        # Let s_i = labels[i], s_j = labels[j]
        # After flip: s_i' = 1-s_i, s_j' = 1-s_j
        
        # For edge (i, k) where k != j:
        # Before: crosses if s_i != s_k
        # After: crosses if (1-s_i) != s_k = s_i != s_k (same!)
        # Wait, that's not right. Let me think again.
        
        # Actually, flipping i changes the cut status of all edges incident to i.
        # Flipping j changes the cut status of all edges incident to j.
        # The edge (i,j) is affected by both flips.
        
        # Gain from flipping i alone (assuming j stays):
        # = other_count[i] - same_count[i]
        
        # Gain from flipping j alone (assuming i stays):
        # = other_count[j] - same_count[j]
        
        # But when we flip both, the edge (i,j) is counted in both.
        # Let's compute the combined gain:
        
        # For all edges except (i,j):
        # Flipping i changes cut status of edges (i,k) for k != j
        # Flipping j changes cut status of edges (j,k) for k != i
        # These are disjoint sets of edges (except possibly if there's a triangle, but edges are distinct)
        
        # For edge (i,j):
        # Before flip: crosses if s_i != s_j
        # After flip: crosses if (1-s_i) != (1-s_j) = s_i != s_j (same!)
        # So edge (i,j) contribution doesn't change!
        
        # Therefore, 2-flip gain = gain_1flip[i] + gain_1flip[j]
        # Wait, that can't be right because then 2-flip would never help if 1-flip is optimal.
        
        # Let me reconsider. The issue is that when we flip i, the "same" and "other" 
        # counts for j change if i and j are connected.
        
        # Let me just compute it directly for candidate pairs.
        
        # For efficiency, only consider pairs where at least one has positive 1-flip gain
        # or where they are in the same group (flipping both keeps group sizes balanced)
        
        # Actually, let's just try all pairs in the same group first
        # since flipping two vertices in the same group to the other group
        # is a natural 2-flip move
        
        group0 = np.where(labels == 0)[0]
        group1 = np.where(labels == 1)[0]
        
        # Try pairs within group0
        if len(group0) >= 2:
            for idx_i in range(len(group0)):
                if time.time() >= deadline:
                    break
                i = group0[idx_i]
                for idx_j in range(idx_i + 1, len(group0)):
                    j = group0[idx_j]
                    
                    # Compute 2-flip gain for (i, j)
                    # Flip both i and j from 0 to 1
                    # New labels: labels[i]=1, labels[j]=1
                    # 
                    # Change in cut:
                    # For each neighbor k of i:
                    #   Before: edge (i,k) crosses if labels[k] == 1
                    #   After: edge (i,k) crosses if labels[k] == 0
                    #   Change: +1 if labels[k]==0, -1 if labels[k]==1, 0 if no edge
                    #   But if k==j, we need to be careful
                    #
                    # Let's compute the delta directly
                    # 
                    # delta = 0
                    # For each k in neighbors of i:
                    #   if k == j: 
                    #     Before: edge (i,j) crosses if labels[i] != labels[j] = 0 != 0 = False
                    #     After: edge (i,j) crosses if 1 != 1 = False
                    #     Change: 0
                    #   else:
                    #     Before: crosses if labels[k] == 1
                    #     After: crosses if labels[k] == 0
                    #     Change: +1 if labels[k]==0 and adj[i,k]==1, -1 if labels[k]==1 and adj[i,k]==1
                    #
                    # Similarly for neighbors of j (excluding i)
                    
                    # Vectorized:
                    # delta_i = sum over k != j of adj[i,k] * (1 if labels[k]==0 else -1)
                    # delta_j = sum over k != i of adj[j,k] * (1 if labels[k]==0 else -1)
                    # delta = delta_i + delta_j
                    
                    # Note: for k in group0 (label 0), contribution is +1
                    # For k in group1 (label 1), contribution is -1
                    # Exclude each other
                    
                    delta_i = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[i, k]:
                            if labels[k] == 0:
                                delta_i += 1
                            else:
                                delta_i -= 1
                    
                    delta_j = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[j, k]:
                            if labels[k] == 0:
                                delta_j += 1
                            else:
                                delta_j -= 1
                    
                    delta = delta_i + delta_j
                    
                    if delta > 0:
                        labels[i] = 1
                        labels[j] = 1
                        improved = True
                        break
                if improved:
                    break
        
        if not improved and len(group1) >= 2:
            for idx_i in range(len(group1)):
                if time.time() >= deadline:
                    break
                i = group1[idx_i]
                for idx_j in range(idx_i + 1, len(group1)):
                    j = group1[idx_j]
                    
                    delta_i = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[i, k]:
                            if labels[k] == 0:
                                delta_i += 1
                            else:
                                delta_i -= 1
                    
                    delta_j = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[j, k]:
                            if labels[k] == 0:
                                delta_j += 1
                            else:
                                delta_j -= 1
                    
                    delta = delta_i + delta_j
                    
                    if delta > 0:
                        labels[i] = 0
                        labels[j] = 0
                        improved = True
                        break
                if improved:
                    break
    
    return labels


def partition(adjacency):
    """Give each vertex a 0/1 label to maximize the cut size."""
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline: 145 seconds to leave safety margin
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    n_restarts = 20  # Will be limited by time
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            # Spectral initialization
            labels = _spectral_init(adj, n)
        elif restart == 1:
            # Alternating initialization
            labels = np.arange(n) % 2
        else:
            # Random initialization
            labels = np.random.randint(0, 2, n)
        
        # 1-flip local search
        labels = _local_search_1flip(adj, labels, deadline)
        
        # 2-flip local search to escape local optima
        labels = _local_search_2flip(adj, labels, deadline)
        
        # Compute cut size
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we've found a very good solution, we can stop early
        # The maximum possible cut is sum of all edges
        max_possible = adj.sum() // 2
        if best_cut >= 0.9 * max_possible:
            break
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
