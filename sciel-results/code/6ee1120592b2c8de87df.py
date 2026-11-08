"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, adj_int, max_time=None):
    """Iterated local search via single-flip improvement with vectorized gain computation."""
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees and initial gains
    # gain[v] = number of neighbors in same partition - number in opposite partition
    # If gain[v] > 0, flipping v increases the cut by gain[v]
    
    # Compute initial cut size
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    current_cut = int((adjacency & diff).sum() // 2)
    
    # Compute gains: for each vertex v, 
    # same_count = sum of adj[v, u] where labels[u] == labels[v]
    # other_count = sum of adj[v, u] where labels[u] != labels[v]
    # gain[v] = same_count - other_count = 2*same_count - degree[v]
    
    degree = adj_int.sum(axis=1).astype(np.int32)
    
    # For each vertex, count neighbors with same label
    # labels is 0/1, adj_int is 0/1
    # same_count[v] = sum_u adj_int[v,u] * (labels[u] == labels[v])
    # = sum_u adj_int[v,u] * (2*labels[u] - 1) * (2*labels[v] - 1) / 2 + degree[v]/2
    # Actually simpler: 
    # same_count[v] = (adj_int[v] @ (labels == labels[v]))
    # But we can vectorize:
    # Let s = 2*labels - 1, so s is +1/-1
    # Then for vertex v with label s_v, neighbors with same label have s_u = s_v
    # same_count[v] = sum_u adj_int[v,u] * (1 + s_v * s_u) / 2
    # = (1/2) * (degree[v] + s_v * (adj_int @ s)[v])
    
    s = (2 * labels - 1).astype(np.int32)
    adj_s = adj_int @ s  # shape (n,)
    same_count = (degree + s * adj_s) // 2
    other_count = degree - same_count
    gains = same_count - other_count  # = 2*same_count - degree
    
    start_time = time.time()
    deadline = start_time + (max_time if max_time else 100)
    
    improved = True
    while improved:
        improved = False
        # Find all vertices with positive gain
        positive_mask = gains > 0
        if not positive_mask.any():
            break
        
        # Sort by gain descending for better convergence
        positive_indices = np.where(positive_mask)[0]
        positive_gains = gains[positive_indices]
        order = np.argsort(-positive_gains)
        
        for idx in order:
            if time.time() > deadline:
                return labels, current_cut
            
            v = positive_indices[idx]
            g = gains[v]
            
            if g > 0:
                # Flip vertex v
                labels[v] = 1 - labels[v]
                current_cut += g
                
                # Update gains for v and its neighbors
                # When v flips:
                # - v's gain becomes -g (since same and other swap)
                # - For each neighbor u of v: 
                #   if u was same as v before, now u is other -> u's same_count decreases by 1, other_count increases by 1 -> gain decreases by 2
                #   if u was other than v before, now u is same -> u's same_count increases by 1, other_count decreases by 1 -> gain increases by 2
                # In terms of s: s[v] flips sign
                # adj_s[u] changes by -2*adj_int[u,v]*s_old[v]... 
                # Actually: adj_s[u] = sum_w adj_int[u,w]*s[w]
                # After flip, s[v] -> -s[v], so adj_s[u] changes by -2*adj_int[u,v]*s[v]_old
                # But s[v]_old is the value before flip.
                
                old_s_v = s[v]
                s[v] = -s[v]
                
                # Update adj_s for neighbors of v
                # adj_s[u] += -2 * adj_int[u,v] * old_s_v for all u
                # adj_int[:, v] is the column (since symmetric, same as row)
                adj_s += -2 * adj_int[:, v] * old_s_v
                
                # Update gains for all vertices
                # gains[u] = 2*same_count[u] - degree[u] = 2*((degree[u] + s[u]*adj_s[u])//2) - degree[u]
                # = s[u]*adj_s[u] (if we ignore the //2 rounding, but let's recompute properly)
                # Actually: same_count = (degree + s*adj_s) // 2
                # gains = 2*same_count - degree = 2*((degree + s*adj_s)//2) - degree
                # If (degree + s*adj_s) is even: gains = s*adj_s
                # If odd: gains = s*adj_s - 1 or s*adj_s + 1?
                # Let's just recompute gains from s and adj_s
                # But that's O(n) per flip. Instead, let's track gains incrementally.
                
                # For v: gains[v] was g, now becomes -g
                gains[v] = -g
                
                # For neighbors u of v:
                # If u was in same partition as v before flip:
                #   u's same_count decreases by 1, other_count increases by 1
                #   gain[u] changes by -2
                # If u was in other partition than v before flip:
                #   u's same_count increases by 1, other_count decreases by 1
                #   gain[u] changes by +2
                # 
                # Before flip: s[u] * s[v]_old = +1 if same, -1 if other
                # So change in gain[u] = -2 * (s[u] * s[v]_old) ... wait
                # If same: s[u]*s[v]_old = 1, gain[u] should decrease by 2 -> change = -2
                # If other: s[u]*s[v]_old = -1, gain[u] should increase by 2 -> change = +2
                # So change = -2 * s[u] * s[v]_old
                # But s[v]_old = old_s_v, and s[u] is unchanged
                # So gains[u] += -2 * s[u] * old_s_v for neighbors u of v
                
                neighbors = np.where(adj_int[v] > 0)[0]
                if len(neighbors) > 0:
                    gains[neighbors] += -2 * s[neighbors] * old_s_v
        
        # After a full pass, check if any more improvements
        # The while loop will re-check positive_mask
    
    return labels, current_cut


def partition(adjacency):
    """Find a good partition using multiple restarts and local search."""
    n = len(adjacency)
    start_time = time.time()
    total_deadline = start_time + 115  # Safety margin for 120s limit
    
    # Convert to int array for speed
    adj_int = adjacency.astype(np.int8)
    
    best_labels = None
    best_cut = -1
    
    # Use a mix of random and structured initializations
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > total_deadline - 5:
            break
        
        # Choose initialization strategy
        if restart % 3 == 0:
            # Alternating pattern
            labels = np.arange(n) % 2
        elif restart % 3 == 1:
            # Random
            labels = np.random.randint(0, 2, size=n)
        else:
            # Random with bias toward one side (helps escape some local optima)
            labels = (np.random.rand(n) < 0.5).astype(np.int8)
        
        # Run local search
        remaining = total_deadline - time.time() - 2
        if remaining < 1:
            break
            
        labels, cut = _local_search(adjacency, labels, adj_int, max_time=remaining)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we've used up most of our time, stop
        if time.time() > total_deadline - 10:
            break
    
    # If no best found, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
