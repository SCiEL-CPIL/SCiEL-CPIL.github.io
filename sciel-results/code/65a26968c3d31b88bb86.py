# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _all_flip_gains(adjacency, labels):
    """
    Compute flip gains for all vertices at once using vectorized operations.
    Gain[v] = (neighbors with same label) - (neighbors with opposite label)
    If Gain[v] > 0, flipping v increases the cut size.
    """
    n = len(labels)
    # Convert to float for matrix multiplication efficiency if needed, 
    # but boolean logic is fast too.
    # Let's use the property: 
    # same_count[v] = sum(adj[v, u] where labels[u] == labels[v])
    # opp_count[v] = sum(adj[v, u] where labels[u] != labels[v])
    
    # We can compute the sum of edges from v to label 0 and label 1.
    adj_float = adjacency.astype(np.float32)
    labels_float = labels.astype(np.float32)
    
    # Sum of edges from each vertex to vertices labeled 1
    edges_to_1 = adj_float @ labels_float
    # Sum of edges from each vertex to vertices labeled 0
    # Total degree - edges_to_1
    degrees = adj_float.sum(axis=1)
    edges_to_0 = degrees - edges_to_1
    
    # For a vertex v with label 0:
    # same_count = edges_to_0[v]
    # opp_count = edges_to_1[v]
    # gain = edges_to_0[v] - edges_to_1[v]
    
    # For a vertex v with label 1:
    # same_count = edges_to_1[v]
    # opp_count = edges_to_0[v]
    # gain = edges_to_1[v] - edges_to_0[v]
    
    gains = np.where(labels == 0, edges_to_0 - edges_to_1, edges_to_1 - edges_to_0)
    return gains.astype(np.int32)


def _local_search(adjacency, labels, deadline):
    """
    Iterative local search: flip vertices if it improves cut size.
    Uses a parallel update strategy: find all positive gains, flip them all.
    This is very fast for small N.
    """
    n = len(labels)
    labels = labels.copy()
    
    while time.time() < deadline:
        gains = _all_flip_gains(adjacency, labels)
        positive = np.where(gains > 0)[0]
        if len(positive) == 0:
            break
        # Flip all vertices with positive gain simultaneously
        labels[positive] = 1 - labels[positive]
        
    return labels


def _spectral_partition(adjacency, n):
    """Use Fiedler vector for initial partition."""
    A = adjacency.astype(np.float32)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        # For N=100, eigh is very fast
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_partition(adjacency, n, deadline):
    """Greedy construction: add vertices one by one, choosing label that maximizes cut."""
    labels = np.zeros(n, dtype=np.int8)
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        # Count edges to 0 and 1
        cut_if_0 = adjacency[v][labels == 1].sum()
        cut_if_1 = adjacency[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(np.int8)


def _perturb(labels, n, num_flips, rng):
    """Perturb the labels by flipping 'num_flips' random vertices."""
    perturbed = labels.copy()
    if num_flips > 0:
        indices = rng.choice(n, size=min(num_flips, n), replace=False)
        perturbed[indices] = 1 - perturbed[indices]
    return perturbed


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Multiple diverse initializations and local search
    initializations = []
    
    # 1. Spectral partition (High quality start)
    initializations.append(_spectral_partition(adjacency, n))
    
    # 2. Greedy partition
    initializations.append(_greedy_partition(adjacency, n, deadline))
    
    # 3. Alternating pattern
    initializations.append((np.arange(n) % 2).astype(np.int8))
    
    # 4-13. Random initializations
    for seed in range(10):
        initializations.append(_random_labels(n, 1000 + seed))
    
    # Run local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline - 5.0: # Leave 5s for ILS
            break
        labels = _local_search(adjacency, init_labels, deadline - 5.0)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 2: Iterated Local Search (ILS)
    # Strategy: Perturb the current best solution and run local search again.
    # This allows escaping local optima found by the parallel LS.
    rng = np.random.RandomState(42)
    il_deadline = deadline - 2.0 # Leave 2s for final safety check/return
    
    # We can vary the perturbation size.
    # Start with small perturbations (1-3 flips) to explore nearby landscape,
    # then increase if stuck.
    
    current_iter = 0
    stall_counter = 0
    last_best_cut = best_cut
    
    while time.time() < il_deadline:
        # Determine perturbation size
        # If we haven't improved in a while, perturb more aggressively
        if stall_counter == 0:
            num_flips = rng.randint(1, 4)
        elif stall_counter < 10:
            num_flips = rng.randint(3, 8)
        else:
            num_flips = rng.randint(5, 15)
            
        perturbed_labels = _perturb(best_labels, n, num_flips, rng)
        
        # Run local search from perturbed state
        new_labels = _local_search(adjacency, perturbed_labels, il_deadline)
        new_cut = _cut_size(adjacency, new_labels)
        
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels.copy()
            stall_counter = 0
        else:
            stall_counter += 1
            
        current_iter += 1
        
        # Optional: If we have a lot of time left and are stuck, try a "random restart" 
        # of the ILS process from a completely new random start?
        # The initial phase already did 13 starts. ILS is usually sufficient.
        
    # Final check
    if time.time() < deadline - 1.0:
        # One last local search pass on the absolute best
        final_labels = _local_search(adjacency, best_labels, deadline - 1.0)
        final_cut = _cut_size(adjacency, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
