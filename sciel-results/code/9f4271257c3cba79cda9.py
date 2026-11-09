"""Combined Max-Cut solver: spectral init, WalkSAT with gain caching, and simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(labels, adj):
    """Compute the cut size (number of edges crossing the partition)."""
    x = np.where(labels == 0, 1.0, -1.0)
    xAx = x @ adj @ x
    E = adj.sum() / 2.0
    return E - 0.5 * xAx


def _compute_gains(labels, adj, degrees):
    """
    Compute the gain for flipping each vertex.
    gain[v] = (neighbors of v on same side) - (neighbors of v on opposite side)
    If gain > 0, flipping v increases the cut size.
    """
    # For each vertex v, count neighbors on the same side
    # same_counts[v] = sum_u adj[v,u] * (labels[v] == labels[u])
    same_eq = (labels[:, None] == labels[None, :])
    same_counts = (adj * same_eq).sum(axis=1)
    other_counts = degrees - same_counts
    gains = same_counts - other_counts
    return gains


def _walksat(adj, labels, deadline, max_iters=50000):
    """
    WalkSAT-style local search for Max-Cut.
    - Greedy phase: flip the vertex with the best positive gain.
    - Escape phase: when stuck, flip a random vertex involved in an uncut edge.
    """
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(labels, adj)
    degrees = adj.sum(axis=1)
    
    for it in range(max_iters):
        if time.time() > deadline - 1.0:
            break
        
        gains = _compute_gains(labels, adj, degrees)
        max_gain = np.max(gains)
        
        if max_gain > 0:
            # Greedy: flip the vertex with the best gain
            v = np.argmax(gains)
            labels[v] = 1 - labels[v]
            cut += max_gain
        else:
            # Stuck: find an uncut edge and flip one of its endpoints
            # Uncut edge: adj[i,j] == 1 and labels[i] == labels[j]
            same_eq = (labels[:, None] == labels[None, :])
            uncut_mask = adj & same_eq
            uncut_edges = np.argwhere(uncut_mask)
            
            if len(uncut_edges) == 0:
                break  # Perfect cut found
            
            # Pick a random uncut edge
            edge_idx = np.random.randint(0, len(uncut_edges))
            u, v = uncut_edges[edge_idx]
            
            # Flip one of the endpoints randomly
            flip_v = u if np.random.rand() < 0.5 else v
            labels[flip_v] = 1 - labels[flip_v]
            cut += gains[flip_v]
    
    return labels, cut


def _simulated_annealing(adj, labels, deadline, max_iters=100000):
    """
    Simulated annealing to escape local optima.
    Accepts worse moves with probability exp(gain / temperature).
    """
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(labels, adj)
    best_labels = labels.copy()
    best_cut = cut
    degrees = adj.sum(axis=1)
    
    current_temp = 5.0
    cooling_rate = 0.9998
    
    for it in range(max_iters):
        if time.time() > deadline - 1.0:
            break
        
        gains = _compute_gains(labels, adj, degrees)
        
        # Choose a random vertex to flip
        v = np.random.randint(n)
        gain = gains[v]
        
        # Accept if gain > 0 or with probability exp(gain / temp)
        if gain > 0 or np.random.rand() < np.exp(gain / max(current_temp, 1e-8)):
            labels[v] = 1 - labels[v]
            cut += gain
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        current_temp *= cooling_rate
    
    return best_labels, best_cut


def _spectral_init(adj, n):
    """
    Spectral initialization using the Fiedler vector of the Laplacian.
    """
    A = adj.astype(np.float64)
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    
    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
        labels = (fiedler < 0).astype(np.int8)
        
        # Ensure both sides are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
    except Exception:
        labels = np.arange(n) % 2
    
    return labels


def partition(adjacency):
    """
    Find a good 2-partition (Max-Cut) of the graph.
    Uses spectral initialization + WalkSAT + Simulated Annealing with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Ensure symmetry
    adj = (adj + adj.T) // 2
    
    best_labels = None
    best_cut = -1
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Strategy: Multiple restarts with different initializations
    # 1. Spectral init
    # 2. Alternating start
    # 3. Random starts
    # 4. Perturbed spectral starts
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Choose initialization strategy
        if r == 0:
            # Spectral initialization
            labels = _spectral_init(adj, n)
        elif r == 1:
            # Alternating start
            labels = np.arange(n) % 2
        elif r % 4 == 2:
            # Spectral with random perturbation
            labels = _spectral_init(adj, n)
            # Perturb ~10% of vertices
            num_perturb = max(1, n // 10)
            perturb_idx = np.random.choice(n, size=num_perturb, replace=False)
            labels[perturb_idx] = 1 - labels[perturb_idx]
        else:
            # Random start
            labels = np.random.randint(0, 2, size=n).astype(np.int8)
            # Ensure both sides are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels[0] = 1 - labels[0]
        
        # Phase 1: WalkSAT (greedy + random escape)
        labels, cut = _walksat(adj, labels, deadline)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Phase 2: Simulated Annealing on the current solution
        if time.time() < deadline - 3.0:
            sa_labels, sa_cut = _simulated_annealing(adj, labels, deadline)
            
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        # Phase 3: Another WalkSAT pass on the SA result (polish)
        if time.time() < deadline - 2.0:
            ws_labels, ws_cut = _walksat(adj, sa_labels, deadline)
            if ws_cut > best_cut:
                best_cut = ws_cut
                best_labels = ws_labels.copy()
    
    # Final polish: run WalkSAT once more on the best solution
    if best_labels is not None and time.time() < deadline - 1.0:
        final_labels, final_cut = _walksat(adj, best_labels, deadline)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
