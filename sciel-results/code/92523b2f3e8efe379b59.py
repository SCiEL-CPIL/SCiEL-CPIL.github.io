"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline: 150 seconds total, leave 15s safety margin
    start_time = time.time()
    deadline = start_time + 135.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        s = 2 * labels - 1  # ±1 vector
        val = s @ (adj @ s)
        total_edges = adj.sum() // 2
        return total_edges - val // 2
    
    def vectorized_gains(labels):
        """Compute flip gains for all vertices at once."""
        # For each vertex v, gain = 2 * (edges to same side) - degree
        # edges to same side = sum of adj[v, u] where labels[u] == labels[v]
        # We can compute this as: for each v, sum adj[v] * (labels == labels[v])
        
        # Create a matrix where M[v,u] = 1 if labels[v] == labels[u], else 0
        # Then same_side_counts[v] = M[v] @ adj[v] = row-wise dot product
        # Actually: same_side_counts[v] = sum_u adj[v,u] * (labels[u] == labels[v])
        
        # Efficient approach:
        # Let s = 2*labels - 1 (±1 vector)
        # For vertex v: same_side = sum_u adj[v,u] * (1 + s[v]*s[u])/2
        #             = (deg[v] + s[v] * sum_u adj[v,u]*s[u]) / 2
        #             = (deg[v] + s[v] * (adj @ s)[v]) / 2
        # gain[v] = 2 * same_side - deg[v]
        #         = 2 * (deg[v] + s[v] * (adj @ s)[v]) / 2 - deg[v]
        #         = deg[v] + s[v] * (adj @ s)[v] - deg[v]
        #         = s[v] * (adj @ s)[v]
        
        s = 2 * labels - 1
        adj_s = adj @ s
        gains = s * adj_s
        return gains
    
    def flip_gain_single(v, labels):
        """Gain in cut size if we flip vertex v."""
        s = 2 * labels - 1
        return s[v] * (adj @ s)[v]
    
    def local_search(labels, adj, time_limit=None, max_iters=100000):
        """Local search using vectorized gain computation."""
        labels = labels.copy()
        
        iters = 0
        improved = True
        while improved and iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            # Compute all gains at once
            gains = vectorized_gains(labels)
            
            # Find the best positive gain
            max_gain = gains.max()
            if max_gain <= 0:
                break
            
            # Flip all vertices with positive gain (batch flip)
            flip_mask = gains > 0
            labels[flip_mask] = 1 - labels[flip_mask]
        
        return labels
    
    def simulated_annealing(labels, adj, iters=200000, t_start=3.0, t_end=0.01, time_limit=None):
        """Simulated annealing for max cut with vectorized gain for single flip."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / iters)
        
        # Precompute degrees
        degrees = adj.sum(axis=1)
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            
            # Pick random vertex
            v = random.randint(0, n - 1)
            
            # Compute gain for flipping v
            s = 2 * labels - 1
            gain = s[v] * (adj @ s)[v]
            
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    def multi_start_search(adj, deadline):
        """Multiple starts with different initializations."""
        best_labels = None
        best_cut = -1
        
        # Start 1: alternating
        labels = np.arange(n) % 2
        labels = local_search(labels, adj, time_limit=deadline)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        
        # Start 2-10: random starts
        for start_iter in range(9):
            if time.time() > deadline:
                break
            labels = np.random.randint(0, 2, size=n)
            labels = local_search(labels, adj, time_limit=deadline)
            c = cut_size(labels)
            if c > best_cut:
                best_cut = c
                best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Phase 1: Multi-start local search
    best_labels, best_cut = multi_start_search(adj, deadline)
    
    # Phase 2: Simulated annealing from best solution
    if time.time() < deadline - 15:
        sa_labels, sa_cut = simulated_annealing(
            best_labels, adj, iters=200000, t_start=3.0, t_end=0.01,
            time_limit=deadline - 15
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 3: Final local search polish
    if time.time() < deadline:
        best_labels = local_search(best_labels, adj, time_limit=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
