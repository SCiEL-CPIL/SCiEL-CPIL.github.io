"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _compute_gains(A, labels):
    """
    Compute the gain (change in cut size) for flipping each vertex.
    Returns an array where gains[i] = change in cut if vertex i is flipped.
    """
    n = len(labels)
    # For vertex v with label lv:
    # Current contribution to cut: edges from v to vertices with label != lv
    # After flip: edges from v to vertices with label == lv (old)
    # Gain = (new_cut - old_cut) = (edges to same) - (edges to opp)
    
    # same_mask[v] = True if labels[i] == labels[v] for all i
    same = (labels[:, None] == labels[None, :])
    # For each vertex v, count neighbors in same partition
    same_counts = (A.astype(np.int32) * same).sum(axis=1)
    # Total degree
    degrees = A.sum(axis=1).astype(np.int32)
    # Opp counts = degree - same counts
    opp_counts = degrees - same_counts
    # Gain if flipped = same_counts - opp_counts
    gains = same_counts - opp_counts
    return gains


def _vectorized_local_search(A, labels, deadline):
    """
    Vectorized local search: repeatedly flip all improving vertices simultaneously.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        gains = _compute_gains(A, labels)
        # Flip vertices with positive gain
        flip_mask = gains > 0
        if not flip_mask.any():
            break
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels


def _adaptive_sa(A, labels, deadline, max_iter=50000):
    """
    Multi-start style simulated annealing with adaptive cooling.
    Temperature drops when stuck (no improvement for a while),
    but can be boosted if we're exploring well.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    
    current_cut = _cut_size(A, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    T = 5.0
    T_min = 0.001
    T_max = 20.0
    cool_rate = 0.999
    stuck_count = 0
    improvement_threshold = 500  # iterations without improvement before cooling
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Adaptive cooling: cool faster when stuck, slower when improving
        if stuck_count > improvement_threshold:
            T *= 0.95  # aggressive cooling
        else:
            T *= cool_rate
        
        T = max(T, T_min)
        
        # Pick random vertex to flip
        v = np.random.randint(n)
        old_label = labels[v]
        
        # Compute gain for this specific flip
        # Gain = (neighbors in same) - (neighbors in opp)
        same_mask = (labels == old_label)
        same_count = A[v, same_mask].sum()
        opp_count = A[v, ~same_mask].sum()
        delta = same_count - opp_count
        
        if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
            labels[v] = 1 - old_label
            current_cut += delta
            stuck_count = 0
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            stuck_count += 1
    
    return best_labels, best_cut


def _spectral_init(adjacency):
    """Spectral partition for diverse initial solution."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler >= 0).astype(np.int32)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, size=n)
        return labels
    except:
        return np.random.randint(0, 2, size=n)


def partition(adjacency):
    """
    Multi-start simulated annealing with vectorized local search polish.
    Uses adaptive cooling to escape local optima.
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    A = adjacency.astype(np.int32)
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # Spectral
    initial_solutions.append(_spectral_init(adjacency))
    
    # Random
    for _ in range(4):
        initial_solutions.append(np.random.randint(0, 2, size=n))
    
    # Alternating
    initial_solutions.append(np.arange(n) % 2)
    
    # Perturbed spectral
    spec = _spectral_init(adjacency)
    perturb_mask = np.random.random(n) < 0.3
    perturbed = spec.copy()
    perturbed[perturb_mask] = 1 - perturbed[perturb_mask]
    initial_solutions.append(perturbed)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Local search on each initial solution to get good starting points
    for init in initial_solutions:
        if time.time() > deadline - 20:
            break
        ls_result = _vectorized_local_search(A, init, deadline - 20)
        cut = _cut_size(adjacency, ls_result)
        if cut > best_cut:
            best_cut = cut
            best_labels = ls_result
    
    # Phase 2: Multi-start SA from best local search results
    # Run SA from multiple diverse starting points
    sa_starts = [best_labels]
    
    # Add perturbed versions of best
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        pert = best_labels.copy()
        mask = np.random.random(n) < 0.2
        pert[mask] = 1 - pert[mask]
        sa_starts.append(pert)
    
    # Run SA from each start, keeping the best
    for sa_init in sa_starts:
        if time.time() > deadline - 5:
            break
        sa_result, sa_cut = _adaptive_sa(A, sa_init, deadline - 5)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_result
    
    # Phase 3: Final polish with vectorized local search
    if time.time() < deadline - 2:
        final = _vectorized_local_search(A, best_labels, deadline - 1)
        final_cut = _cut_size(adjacency, final)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
