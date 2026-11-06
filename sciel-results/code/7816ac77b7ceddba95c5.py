"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline: 120 seconds total, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        # Edge (i,j) crosses if labels[i] != labels[j]
        # For each i, count neighbors j with different label
        # cut = sum over i of (adj[i] @ (labels != labels[i])) / 2
        # But easier: cut = sum_{i<j, labels[i]!=labels[j]} adj[i,j]
        # Using matrix: cut = 0.5 * sum_i sum_j adj[i,j] * (1 - (2*labels[i]-1)*(2*labels[j]-1))
        # = 0.5 * (total_edges - sum_i sum_j adj[i,j] * s[i]*s[j]) where s = 2*labels-1
        s = 2 * labels - 1  # ±1 vector
        # sum_i sum_j adj[i,j] s[i] s[j] = s^T A s
        val = s @ (adj @ s)
        total_edges = adj.sum() // 2
        return total_edges - val // 2
    
    def flip_gain(v, labels, adj):
        """Gain in cut size if we flip vertex v."""
        # Current contribution of v to cut: neighbors with different label
        # After flip: neighbors with same label become different
        # Gain = (neighbors with same label) - (neighbors with different label)
        # = 2 * (neighbors with same label) - degree(v)
        same = adj[v, labels == labels[v]].sum()
        deg = adj[v].sum()
        return 2 * same - deg
    
    def local_search(labels, adj, max_iters=10000, time_limit=None):
        """Greedy local search: flip vertices that improve cut size."""
        labels = labels.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time_limit is not None and time.time() > time_limit:
                break
            for v in range(n):
                gain = flip_gain(v, labels, adj)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def simulated_annealing(labels, adj, iters=50000, t_start=1.0, t_end=0.01, time_limit=None):
        """Simulated annealing for max cut."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / iters)
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            # Pick random vertex
            v = random.randint(0, n - 1)
            gain = flip_gain(v, labels, adj)
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Multi-start with different initializations
    best_labels = None
    best_cut = -1
    
    # Start 1: alternating
    labels = np.arange(n) % 2
    labels = local_search(labels, adj, max_iters=5000, time_limit=deadline)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Start 2: random
    for start_iter in range(5):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = local_search(labels, adj, max_iters=5000, time_limit=deadline)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # Simulated annealing from best solution found so far
    if time.time() < deadline - 5:
        sa_labels, sa_cut = simulated_annealing(
            best_labels, adj, iters=100000, t_start=1.0, t_end=0.01,
            time_limit=deadline - 5
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Final local search polish
    if time.time() < deadline:
        best_labels = local_search(best_labels, adj, max_iters=10000, time_limit=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
