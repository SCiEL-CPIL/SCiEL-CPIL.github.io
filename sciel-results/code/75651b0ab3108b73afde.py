"""Rewritten program for Maximum Cut using spectral initialization + greedy + simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    cut = (adj & diff).sum() // 2
    return int(cut)


def _greedy_improve(adj, labels, max_passes=50):
    """Greedy local search: flip vertices that improve the cut."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            same_mask = (labels == labels[v])
            diff_mask = (labels != labels[v])
            same_count = adj[v, same_mask].sum()
            diff_count = adj[v, diff_mask].sum()
            
            delta = same_count - diff_count
            if delta > 0:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels


def _spectral_init(adj, n, rng):
    """Spectral initialization using the Fiedler vector (second eigenvector of Laplacian)."""
    # Compute degree matrix
    degrees = adj.sum(axis=1).astype(float)
    # Laplacian L = D - A
    L = np.diag(degrees) - adj.astype(float)
    
    # Compute eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigvecs columns are sorted by eigenvalue in ascending order
        fiedler = eigvecs[:, 1]
        
        # Use the sign of the Fiedler vector to initialize labels
        # Positive values -> 1, negative values -> 0
        labels = (fiedler >= 0).astype(int)
        
        # Check if this gives a balanced partition; if not, try thresholding at mean
        if labels.sum() < n * 0.2 or labels.sum() > n * 0.8:
            threshold = np.mean(fiedler)
            labels = (fiedler >= threshold).astype(int)
        
        return labels
    except Exception:
        # Fallback to random
        return rng.integers(0, 2, size=n)


def _simulated_annealing(adj, labels, rng, time_limit=5.0):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Initial and final temperatures
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    start_time = time.time()
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        
        # Compute delta in cut size if we flip v
        same_mask = (labels == labels[v])
        diff_mask = (labels != labels[v])
        same_count = adj[v, same_mask].sum()
        diff_count = adj[v, diff_mask].sum()
        
        # Delta in the sum S (before dividing by 2)
        delta_S = same_count - diff_count
        # Delta in actual cut size
        delta_cut = delta_S // 2
        
        if delta_cut >= 0 or rng.random() < np.exp(delta_cut / T):
            labels[v] = 1 - labels[v]
            current_cut += delta_cut
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        T *= cooling_rate
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization, greedy local search,
    and simulated annealing with multiple restarts.
    """
    start_time = time.time()
    DEADLINE = 145.0  # seconds, leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy bool matrix for efficient operations
    adj = np.array(adjacency, dtype=bool)
    
    # Ensure symmetry (in case input isn't perfectly symmetric)
    adj = adj | adj.T
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral initialization + greedy
    labels_spec = _spectral_init(adj, n, rng)
    labels_spec = _greedy_improve(adj, labels_spec, max_passes=50)
    cut_spec = _cut_size(adj, labels_spec)
    if cut_spec > best_cut:
        best_cut = cut_spec
        best_labels = labels_spec.copy()
    
    # Phase 2: Multiple random restarts with greedy + simulated annealing
    restart_count = 0
    while time.time() < DEADLINE:
        # Generate random initial labels
        labels = rng.integers(0, 2, size=n)
        
        # Greedy improvement first
        labels = _greedy_improve(adj, labels, max_passes=30)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Simulated annealing to try to escape local optima
        if time.time() < DEADLINE - 2.0:
            labels_sa, cut_sa = _simulated_annealing(adj, labels, rng, time_limit=3.0)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
        
        restart_count += 1
        
        # Check time
        if time.time() >= DEADLINE:
            break
    
    # Phase 3: If we have time, do more spectral-based restarts with different seeds
    while time.time() < DEADLINE - 1.0:
        # Perturb the spectral initialization
        labels_spec = _spectral_init(adj, n, rng)
        # Add some random noise
        flip_mask = rng.random(n) < 0.1
        labels_spec[flip_mask] = 1 - labels_spec[flip_mask]
        
        labels_spec = _greedy_improve(adj, labels_spec, max_passes=50)
        cut_spec = _cut_size(adj, labels_spec)
        
        if cut_spec > best_cut:
            best_cut = cut_spec
            best_labels = labels_spec.copy()
        
        if time.time() >= DEADLINE - 1.0:
            break
    
    if best_labels is None:
        # Fallback: return alternating labels
        return np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
