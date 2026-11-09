"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    # Max-Cut via multi-restart simulated annealing with greedy polish
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def cut_size(labels):
        # Efficient cut size: sum over edges where endpoints differ
        # cut = 0.5 * sum_{i,j} adj[i,j] * (1 - 2*(labels[i]==labels[j]))
        # More simply: for each node, count neighbors in opposite set
        # Using matrix ops: cut = sum(adj * (labels[:,None] != labels[None,:]))
        # But that's O(n^2) per call. Let's use a faster incremental approach in SA.
        # For now, direct computation:
        diff = (labels[:, None] != labels[None, :])
        return int(np.sum(adj * diff))
    
    def greedy_polish(labels):
        # Greedy local improvement: flip node if it increases cut
        labels = labels.copy()
        improved = True
        while improved:
            improved = False
            for v in range(n):
                same = int(adj[v, labels == labels[v]].sum())
                other = int(adj[v, labels != labels[v]].sum())
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def simulated_annealing(labels_init, max_iters=5000):
        labels = labels_init.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        T_start = 10.0
        T_end = 0.01
        iters = max_iters
        
        for i in range(iters):
            if time.time() > deadline - 5.0:
                break
            T = T_start * (T_end / T_start) ** (i / iters)
            
            # Pick random node to flip
            v = random.randint(0, n - 1)
            
            # Compute delta in cut size if we flip v
            same = int(adj[v, labels == labels[v]].sum())
            other = int(adj[v, labels != labels[v]].sum())
            delta = other - same  # new_cut - current_cut = (other - same) - (same - other)... 
            # Actually: if we flip v, edges that were crossing become non-crossing and vice versa.
            # Current contribution of v to cut: 'other' (neighbors in other set)
            # After flip: neighbors in 'same' set become 'other', so new contribution: 'same'
            # delta = same - other
            delta = same - other
            
            if delta > 0:
                # Accept
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Accept with probability exp(delta / T)
                if T > 0:
                    prob = np.exp(delta / T)
                    if random.random() < prob:
                        labels[v] = 1 - labels[v]
                        current_cut += delta
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Multi-restart strategy
    best_labels = None
    best_cut = -1
    
    # Initial restarts with different random seeds
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initialization strategies
        if r == 0:
            # Alternating
            labels_init = np.arange(n) % 2
        elif r == 1:
            # Random
            labels_init = np.random.randint(0, 2, n)
        elif r == 2:
            # Majority of neighbors' labels (one step of message passing)
            labels_init = np.random.randint(0, 2, n)
            # One step: each node takes majority of neighbors
            for v in range(n):
                if adj[v].sum() > 0:
                    nb_labels = labels_init[adj[v] == 1]
                    if len(nb_labels) > 0:
                        labels_init[v] = 1 if nb_labels.sum() > len(nb_labels) / 2 else 0
        else:
            # Random with bias
            p = 0.3 + 0.4 * (r % 5) / 5.0
            labels_init = (np.random.rand(n) < p).astype(int)
        
        # Run SA
        sa_labels, sa_cut = simulated_annealing(labels_init, max_iters=3000)
        
        # Greedy polish
        polished_labels = greedy_polish(sa_labels)
        polished_cut = cut_size(polished_labels)
        
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished_labels
        
        # Also try polishing the SA result directly
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final greedy polish on the best found
    if best_labels is not None:
        final_labels = greedy_polish(best_labels)
        final_cut = cut_size(final_labels)
        if final_cut > best_cut:
            best_labels = final_labels
            best_cut = final_cut
    
    # Safety: if we haven't found anything, return a simple heuristic
    if best_labels is None:
        labels = np.arange(n) % 2
        labels = greedy_polish(labels)
        return labels
    
    return best_labels
# EVOLVE-BLOCK-END
