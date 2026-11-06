"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)

def _spectral_init(adj, n):
    """
    Initialize using the second eigenvector of the graph Laplacian.
    Returns a 0/1 label array.
    """
    # Build Laplacian L = D - A
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        # Use numpy's eigvalsh/eigh for symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvector (index 1) is the Fiedler vector
        fiedler = eigenvectors[:, 1]
        # Threshold at 0 to create partition
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        # Fallback to random
        labels = np.random.randint(0, 2, size=n).astype(np.int8)

    return labels

def _random_init(n):
    """Random 0/1 initialization."""
    return np.random.randint(0, 2, size=n).astype(np.int8)

def _greedy_improve(adj, labels, max_passes=100):
    """
    Greedy local search: repeatedly flip vertices that improve the cut.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()

    for _ in range(max_passes):
        improved = False
        for v in range(n):
            same = adj_int[v][labels == labels[v]].sum()
            other = adj_int[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break

    return labels

def _fast_simulated_annealing(adj, labels, start_temp=5.0, cool_rate=0.995, max_steps=10000):
    """
    Fast Simulated Annealing using vectorized operations for delta calculation.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    
    # Precompute degrees
    degrees = np.sum(adj_int, axis=1)
    
    # Helper to compute cut size quickly
    def calc_cut(lab):
        # cross_mask is (n, n) bool
        # Optimization: use matrix multiplication if possible, but for n=100,
        # (adj * (lab[:, None] != lab[None, :])).sum() is fine.
        # Even better:
        # Let s_i = 1 if lab_i == 0 else -1
        # Cut = 0.5 * (sum_i sum_j A_ij (1 - s_i s_j) / 2) ? No.
        # Let's stick to the efficient vectorized version:
        # cross = (lab[:, None] != lab[None, :])
        # return (adj_int * cross).sum() // 2
        
        # Alternative fast cut:
        # sum of edges within same partition
        # same_mask = (lab[:, None] == lab[None, :])
        # same_edges = (adj_int * same_mask).sum() // 2
        # total_edges = np.sum(adj_int) // 2
        # return total_edges - same_edges
        
        # Let's use the direct one, it's fast enough for n=100
        cross = (lab[:, None] != lab[None, :])
        return (adj_int * cross).sum() // 2

    current_cut = calc_cut(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp

    # Precompute neighbor lists or just use matrix rows
    # For n=100, iterating over neighbors is fine, but vectorizing delta is key.
    
    for step in range(max_steps):
        # Check time occasionally
        if step % 100 == 0 and time.time() > deadline:
            break
            
        v = np.random.randint(0, n)
        old_label = labels[v]
        new_label = 1 - old_label

        # Compute delta
        # delta = (edges from v to same side) - (edges from v to other side)
        # If we flip v, edges to same become crossing (+1), edges to other become same (-1)
        # Wait.
        # Current state: v is on side S.
        # Edges to S (same) are NOT crossing.
        # Edges to 1-S (other) ARE crossing.
        # New state: v is on side 1-S.
        # Edges to S (now other) ARE crossing.
        # Edges to 1-S (now same) are NOT crossing.
        
        # Change in cut:
        # + (edges from v to S)  [these were same, now cross]
        # - (edges from v to 1-S) [these were cross, now same]
        
        # So delta = count_same - count_other
        
        # Vectorized calculation:
        # same_mask = (labels == old_label)
        # other_mask = (labels != old_label)
        # count_same = adj_int[v, same_mask].sum()
        # count_other = adj_int[v, other_mask].sum()
        
        # Even faster:
        # Let s be a vector where s_i = 1 if labels[i] == 0 else -1
        # This might be overkill for n=100.
        
        # Simple loop over neighbors is actually quite fast for sparse-ish graphs,
        # but adjacency matrix is dense bool.
        # adj_int[v] is a vector of 0s and 1s.
        # labels == old_label is a vector of 0s and 1s (bool).
        
        same_mask = (labels == old_label)
        count_same = np.dot(adj_int[v], same_mask)
        count_other = np.dot(adj_int[v], 1 - same_mask) # or adj_int[v].sum() - count_same
        
        delta = count_same - count_other
        
        new_cut = current_cut + delta
        
        if new_cut > current_cut:
            accept = True
        else:
            if temp < 1e-10:
                accept = False
            else:
                # Avoid overflow/underflow issues
                if delta < -100 * temp:
                    accept = False
                else:
                    accept = np.random.random() < np.exp(delta / temp)

        if accept:
            labels[v] = new_label
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        temp *= cool_rate

    return best_labels

def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Uses spectral initialization + greedy local search + fast simulated annealing.
    """
    global deadline
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)
    
    # Ensure symmetry
    adj = (adj | adj.T)
    np.fill_diagonal(adj, 0)

    # Generate multiple initializations
    inits = []

    # Spectral initialization
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)

    # Alternating initialization
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)

    # Several random initializations
    for _ in range(5):
        inits.append(_random_init(n))

    best_labels = None
    best_cut = -1

    for init_labels in inits:
        if time.time() > deadline:
            break

        # Greedy local search
        greedy_labels = _greedy_improve(adj, init_labels)
        greedy_cut = _cut_size(adj, greedy_labels)

        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()

        # Fast Simulated Annealing from the greedy result
        # Allocate remaining time proportionally or fixed budget
        if time.time() < deadline - 5:
            # Adjust steps based on remaining time? 
            # Let's just run a fixed number of steps, it's fast.
            sa_labels = _fast_simulated_annealing(adj, greedy_labels, start_temp=5.0, cool_rate=0.995, max_steps=5000)
            sa_cut = _cut_size(adj, sa_labels)

            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()

    # Final greedy pass on the best found
    if best_labels is not None and time.time() < deadline:
        final_labels = _greedy_improve(adj, best_labels)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()

    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
