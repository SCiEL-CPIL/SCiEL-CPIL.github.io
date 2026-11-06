"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    return int(adj[labels[:, None] != labels[None, :]].sum() // 2)


def _greedy_improve(adj, labels):
    """Iteratively flip vertices that improve the cut until local optimum."""
    n = len(adj)
    adj_int = adj.astype(int)
    labels = labels.copy()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            same = adj_int[v][labels == labels[v]].sum()
            other = adj_int[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=bool)
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple random restarts with greedy local search
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Random initial labeling
        labels = np.random.randint(0, 2, n)
        
        # Greedy local improvement
        improved_labels = _greedy_improve(adj, labels)
        cut = _cut_size(adj, improved_labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = improved_labels.copy()
    
    # If no restarts completed (shouldn't happen), use alternating start
    if best_labels is None:
        labels = np.arange(n) % 2
        best_labels = _greedy_improve(adj, labels)
    
    return best_labels
# EVOLVE-BLOCK-END
