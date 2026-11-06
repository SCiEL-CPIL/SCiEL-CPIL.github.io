"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # For each edge (i,j), it crosses if labels[i] != labels[j]
    # adj is symmetric, so each crossing edge is counted twice in the full sum
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)


def _greedy_improve(adj, labels):
    """Iteratively flip vertices that improve the cut until local optimum."""
    n = len(adj)
    adj_int = adj.astype(np.int8)
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


def _greedy_improve_fast(adj, labels, deadline):
    """Fast greedy improvement with deadline guard."""
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    # Precompute degree for each vertex
    deg = adj_int.sum(axis=1)
    # gain[v] = number of edges from v to opposite side - number to same side
    # gain[v] = (deg[v] - 2 * same_count[v])
    # We maintain same_count implicitly
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for v in range(n):
            if time.time() > deadline:
                return labels
            same = adj_int[v][labels == labels[v]].sum()
            other = deg[v] - same
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _random_restart_greedy(adj, n_restarts, deadline):
    """Try multiple random initializations, each followed by greedy improvement."""
    n = len(adj)
    best_labels = None
    best_cut = -1
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        # Greedy improve
        labels = _greedy_improve_fast(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback to alternating
        labels = np.arange(n) % 2
        labels = _greedy_improve_fast(adj, labels, deadline)
        best_labels = labels
    
    return best_labels


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    # Use a safety margin: stop searching 5 seconds before the 120s limit
    deadline = start_time + 115.0
    
    # Convert to numpy array once
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Strategy: multiple random restarts with greedy local search
    # For n=100, greedy is fast, so we can do many restarts
    # Start with a few restarts, then use remaining time for more
    
    # First, try a structured initialization (alternating)
    labels_alt = np.arange(n) % 2
    labels_alt = _greedy_improve_fast(adj, labels_alt, deadline)
    best_labels = labels_alt.copy()
    best_cut = _cut_size(adj, best_labels)
    
    elapsed = time.time() - start_time
    # Estimate time per restart
    time_per_restart = elapsed / 1.0  # one restart so far
    
    # Do additional random restarts
    max_restarts = 50
    for r in range(1, max_restarts + 1):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = _greedy_improve_fast(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
