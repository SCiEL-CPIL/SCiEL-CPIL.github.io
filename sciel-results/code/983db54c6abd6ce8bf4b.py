"""Rewritten program for Maximum Cut task using spectral initialization + greedy + simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the cut size (number of edges crossing the partition)."""
    n = len(labels)
    # For each vertex, count neighbors on the opposite side
    # Use matrix multiplication for speed
    # labels_as_int: 0 or 1
    # For edge (u,v), it crosses the cut if labels[u] != labels[v]
    # cut = 0.5 * sum_{u,v} adjacency[u,v] * (labels[u] != labels[v])
    # (labels[u] != labels[v]) = 1 - (labels[u] == labels[v])
    # For binary labels: (labels[u] != labels[v]) = labels[u] XOR labels[v]
    # = labels[u] + labels[v] - 2*labels[u]*labels[v]
    # So cut = 0.5 * sum_{u,v} adj[u,v] * (labels[u] + labels[v] - 2*labels[u]*labels[v])
    # = 0.5 * (sum_u labels[u] * deg[u] + sum_v labels[v] * deg[v] - 2 * labels^T A labels)
    # = sum_v labels[v] * deg[v] - labels^T A labels
    labels_int = labels.astype(np.float64)
    degrees = adjacency.sum(axis=1)
    cut = np.dot(labels_int, degrees) - labels_int @ adjacency @ labels_int
    return cut


def _greedy_improve(adjacency, labels, deadline):
    """Perform greedy local search: flip vertices that improve the cut."""
    n = len(labels)
    degrees = adjacency.sum(axis=1)
    labels_int = labels.astype(np.float64)
    
    # Precompute for each vertex the number of neighbors on each side
    # same_side[v] = number of neighbors of v with same label as v
    # diff_side[v] = number of neighbors of v with different label as v
    # If we flip v, the change in cut is: diff_side[v] - same_side[v]
    # (edges that were crossing become non-crossing, and vice versa)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order to break symmetry
        order = list(range(n))
        random.shuffle(order)
        for v in order:
            if time.time() >= deadline:
                return labels
            # Count neighbors with same and different labels
            same = np.sum(adjacency[v] & (labels == labels[v]))
            diff = np.sum(adjacency[v] & (labels != labels[v]))
            if diff < same:
                # Flipping v would improve the cut
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels


def _simulated_annealing(adjacency, labels, deadline, start_temp=10.0, cooling=0.999):
    """Perform simulated annealing to escape local optima."""
    n = len(labels)
    temp = start_temp
    best_labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    
    # Precompute degrees for quick delta computation
    degrees = adjacency.sum(axis=1)
    
    while time.time() < deadline and temp > 0.01:
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        
        # Compute delta in cut size if we flip v
        # Before flip: v is on side labels[v]
        # same = neighbors on same side, diff = neighbors on different side
        # After flip: v is on opposite side
        # Change in cut: edges from v to same side become crossing (+same),
        #                edges from v to diff side become non-crossing (-diff)
        # But we need to be careful: the cut size change is (same - diff)
        # because previously 'diff' edges were crossing, now 'same' edges are crossing
        # Net change = same - diff
        
        same = np.sum(adjacency[v] & (labels == labels[v]))
        diff = np.sum(adjacency[v] & (labels != labels[v]))
        delta = same - diff  # change in cut size if we flip v
        
        if delta > 0:
            # Accept (improvement)
            labels[v] = 1 - labels[v]
            new_cut = _cut_size(adjacency, labels)
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = labels.copy()
        else:
            # Accept with probability exp(delta / temp)
            if random.random() < np.exp(delta / temp):
                labels[v] = 1 - labels[v]
        
        temp *= cooling
    
    return best_labels


def _spectral_init(adjacency):
    """Generate initial partition using the Fiedler vector (2nd eigenvector of Laplacian)."""
    n = len(adjacency)
    adj = adjacency.astype(np.float64)
    
    # Compute Laplacian: L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    # Find the second smallest eigenvector (Fiedler vector)
    # For 100 nodes, we can use full eigendecomposition or sparse methods
    # Using numpy's eigh for symmetric matrix
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue (eigh already returns sorted)
        # The Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        fiedler = eigvecs[:, 1]
    except:
        # Fallback to random
        fiedler = np.random.randn(n)
    
    # Use the sign of the Fiedler vector to create initial partition
    # Vertices with positive Fiedler values go to one side, negative to the other
    # Use the mean as threshold for better balance
    threshold = np.mean(fiedler)
    labels = (fiedler > threshold).astype(int)
    
    # Ensure both sides are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        # Fallback: use median split
        threshold = np.median(fiedler)
        labels = (fiedler > threshold).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, size=n)
    
    return labels


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization, greedy local search,
    and simulated annealing with multiple restarts.
    
    Returns a 0/1 label array for each vertex.
    """
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 115s safety margin out of 120s limit
    
    n = len(adjacency)
    adjacency = np.array(adjacency, dtype=bool)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + greedy refinement
    spectral_labels = _spectral_init(adjacency)
    refined = _greedy_improve(adjacency, spectral_labels.copy(), DEADLINE)
    cut = _cut_size(adjacency, refined)
    best_labels = refined.copy()
    best_cut = cut
    
    # Phase 2: Simulated annealing from the greedy-refined spectral solution
    sa_labels = _simulated_annealing(adjacency, refined.copy(), DEADLINE, start_temp=5.0, cooling=0.9995)
    sa_cut = _cut_size(adjacency, sa_labels)
    if sa_cut > best_cut:
        best_cut = sa_cut
        best_labels = sa_labels.copy()
    
    # Phase 3: Multiple random restarts with greedy + short SA
    num_restarts = 0
    while time.time() < DEADLINE:
        num_restarts += 1
        
        # Random initial assignment
        labels = np.random.randint(0, 2, size=n)
        
        # Greedy refinement
        refined = _greedy_improve(adjacency, labels.copy(), DEADLINE)
        cut = _cut_size(adjacency, refined)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined.copy()
        
        # Short simulated annealing if time permits
        remaining = DEADLINE - time.time()
        if remaining > 5.0:
            sa_labels = _simulated_annealing(adjacency, refined.copy(), DEADLINE - 2.0, 
                                             start_temp=3.0, cooling=0.999)
            sa_cut = _cut_size(adjacency, sa_labels)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        # Limit restarts to avoid excessive computation
        if num_restarts >= 30:
            break
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
