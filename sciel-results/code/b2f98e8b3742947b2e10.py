"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    total_edges = adj.sum() // 2
    
    def cut_size(labels):
        s = 2 * labels - 1
        val = s @ (adj @ s)
        return total_edges - val // 2
    
    def compute_gains(labels, adj, n):
        """Vectorized computation of flip gains for all vertices."""
        s = 2 * labels - 1
        # For vertex v: gain = 2 * (edges to same side) - degree
        # edges to same side = sum of adj[v, u] where s[u] == s[v]
        # = (sum of adj[v,:] * (s==s[v])) 
        # More efficiently: gain[v] = deg[v] - 2*edges_to_opp[v]
        # edges_to_opp[v] = sum of adj[v,u] where s[u] != s[v]
        # = sum of adj[v,u] where s[u] = -s[v]
        # = s[v] * sum(adj[v,:] * s) ... let's think differently
        # 
        # Actually: gain[v] = 2 * same[v] - deg[v]
        # same[v] = sum of adj[v,u] where labels[u] == labels[v]
        # 
        # Let's use: for each v, same_side_sum = sum(adj[v] * (labels == labels[v]))
        # This is O(n^2) but vectorized per vertex... still slow for n=100
        # 
        # Better: same[v] = (adj[v] @ (labels == labels[v]).astype(int))
        # But we can do it all at once:
        # same_matrix = adj * (labels[:, None] == labels[None, :])
        # same[v] = same_matrix[v].sum()
        
        same_mask = (labels[:, None] == labels[None, :]).astype(np.int8)
        same_counts = (adj * same_mask).sum(axis=1)
        degs = adj.sum(axis=1)
        gains = 2 * same_counts - degs
        return gains
    
    def local_search(labels, adj, n, time_limit=None, max_iters=20000):
        """Fast local search with vectorized gain updates."""
        labels = labels.copy()
        iters = 0
        
        while iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            gains = compute_gains(labels, adj, n)
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
            
            if best_gain <= 0:
                break
            
            labels[best_v] = 1 - labels[best_v]
        
        return labels
    
    def sa_from_start(labels, adj, n, t_start, t_end, iters, time_limit=None):
        """Simulated annealing."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(iters, 1))
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            
            v = random.randint(0, n - 1)
            s = 2 * labels - 1
            # gain for flipping v
            same_side = (adj[v] * (s == s[v])).sum()
            deg = adj[v].sum()
            gain = 2 * same_side - deg
            
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    def greedy_init(adj, n):
        """Greedy initialization: add vertices one by one to the side that maximizes cut."""
        labels = np.zeros(n, dtype=int)
        order = np.arange(n)
        np.random.shuffle(order)
        
        for v in order:
            # Count edges to each side
            left_count = (adj[v] * (labels == 0)).sum()
            right_count = (adj[v] * (labels == 1)).sum()
            if right_count > left_count:
                labels[v] = 0
            else:
                labels[v] = 1
        return labels
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Multiple diverse starts with local search
    starts = []
    
    # Start 1: alternating
    starts.append(np.arange(n) % 2)
    
    # Start 2: greedy
    starts.append(greedy_init(adj, n))
    
    # Starts 3-10: random
    for _ in range(8):
        starts.append(np.random.randint(0, 2, size=n))
    
    for i, init_labels in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Local search from this start
        ls_labels = local_search(init_labels, adj, n, time_limit=deadline)
        c = cut_size(ls_labels)
        
        if c > best_cut:
            best_cut = c
            best_labels = ls_labels.copy()
    
    # Phase 2: Simulated annealing from best solution
    if time.time() < deadline - 15:
        sa_time_limit = deadline - 15
        sa_labels, sa_cut = sa_from_start(
            best_labels, adj, n, 
            t_start=3.0, t_end=0.01, 
            iters=200000, 
            time_limit=sa_time_limit
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 3: Final local search polish
    if time.time() < deadline:
        final_labels = local_search(best_labels, adj, n, time_limit=deadline)
        final_cut = cut_size(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    # Phase 4: One more SA pass if time allows
    if time.time() < deadline - 5:
        sa2_labels, sa2_cut = sa_from_start(
            best_labels, adj, n,
            t_start=1.0, t_end=0.005,
            iters=100000,
            time_limit=deadline - 5
        )
        if sa2_cut > best_cut:
            best_cut = sa2_cut
            best_labels = sa2_labels.copy()
        
        # Final polish
        if time.time() < deadline:
            final2 = local_search(best_labels, adj, n, time_limit=deadline)
            final2_cut = cut_size(final2)
            if final2_cut > best_cut:
                best_labels = final2
    
    return best_labels
# EVOLVE-BLOCK-END
