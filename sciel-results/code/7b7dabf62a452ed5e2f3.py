"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: return well before the 150s wall-clock kill.
DEADLINE_MARGIN = 10.0


def _cut_size(labels, adj):
    """Compute the number of edges crossing the partition."""
    diff = (labels[None, :] != labels[:, None])
    return int(np.sum(adj * diff))


def _flip_gain(v, labels, adj, row_sums):
    """
    Gain in cut size if vertex v is flipped.
    gain = 2 * other_side_neighbors - deg(v)
    where other_side_neighbors = deg(v) - same_side_neighbors
    """
    same = int(adj[v, labels == labels[v]].sum())
    deg = int(row_sums[v])
    other = deg - same
    return 2 * other - deg


def _local_search(labels, adj, row_sums, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain
    until no positive gain exists or deadline is reached.
    """
    n = len(labels)
    labels = labels.copy()
    
    while True:
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_v = -1
        for v in range(n):
            g = _flip_gain(v, labels, adj, row_sums)
            if g > best_gain:
                best_gain = g
                best_v = v
        
        if best_v == -1:
            break
        
        labels[best_v] = 1 - labels[best_v]
    
    return labels


def _random_restarts(adj, num_restarts, deadline):
    """
    Perform multiple random restarts with local search, returning the best partition.
    """
    n = len(adj)
    row_sums = adj.sum(axis=1)
    best_labels = None
    best_cut = -1
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial labeling
        labels = np.random.randint(0, 2, size=n)
        
        # Local search
        labels = _local_search(labels, adj, row_sums, deadline)
        
        cut = _cut_size(labels, adj)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels


def partition(adjacency):
    """
    Find a good 2-partition maximizing the number of crossing edges.
    Uses random restarts with greedy local search.
    """
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Convert to numpy array if needed
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=bool)
    
    n = len(adjacency)
    
    # Start with a deterministic alternating initialization as a baseline
    labels_alt = np.arange(n) % 2
    labels_alt = _local_search(labels_alt, adjacency, adjacency.sum(axis=1), deadline)
    best_labels = labels_alt.copy()
    best_cut = _cut_size(best_labels, adjacency)
    
    # Use remaining time for random restarts
    elapsed = time.time() - start_time
    time_remaining = max(0, (start_time + 150.0 - DEADLINE_MARGIN) - time.time())
    
    # Estimate how many restarts we can afford; each local search on 100 nodes
    # should be fast (milliseconds), so we can do many restarts.
    # Let's do restarts until we run out of time.
    num_restarts_done = 0
    while time.time() < deadline:
        labels = _random_restarts(adjacency, 1, deadline)
        if labels is not None:
            cut = _cut_size(labels, adjacency)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        num_restarts_done += 1
        if time.time() > deadline:
            break
    
    return best_labels
# EVOLVE-BLOCK-END
