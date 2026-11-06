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
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvalue's eigenvector (Fiedler vector)
        # gives a good spectral partition
        fiedler = eigenvectors[:, 1]
        # Threshold at 0
        labels = (fiedler > 0).astype(np.int8)
        # Ensure both sides are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = (fiedler > np.median(fiedler)).astype(np.int8)
        return labels
    except Exception:
        return np.arange(n) % 2


def _goemans_maxwell(adj, n, deadline):
    """
    Goemans-Williamson style rounding with random hyperplane.
    Returns a 0/1 label array.
    """
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Use the top k eigenvectors to embed vertices
        k = min(n, 20)
        # Use the smallest non-trivial eigenvectors
        # Actually, for MaxCut we want the largest eigenvalues of -L or use the Fiedler approach
        # Let's use a different approach: random projection after spectral embedding
        # Use the first k eigenvectors (smallest eigenvalues) for embedding
        V = eigenvectors[:, 1:k+1]  # skip the first (constant) eigenvector
        
        # Random hyperplane in the embedded space
        random_vec = np.random.randn(k)
        projections = V @ random_vec
        labels = (projections > 0).astype(np.int8)
        
        if labels.sum() == 0 or labels.sum() == n:
            labels = (projections > np.median(projections)).astype(np.int8)
        return labels
    except Exception:
        return np.arange(n) % 2


def _sdp_init(adj, n, deadline, iters=200):
    """
    SDP relaxation via projected gradient ascent.
    Returns a 0/1 label array from random hyperplane rounding.
    """
    A = adj.astype(float)
    # Initialize X as random orthogonal-ish matrix
    np.random.seed(42)
    X = np.random.randn(n, n)
    # Project to correlation matrix: X X^T with diagonal 1
    # Use a simpler approach: start with identity + noise
    X = np.eye(n) + 0.1 * np.random.randn(n, n)
    # Make symmetric
    X = (X + X.T) / 2.0
    
    # Project to correlation matrix (diagonal 1, PSD)
    def _project_correlation(X):
        # Make symmetric
        X = (X + X.T) / 2.0
        # Eigen-decompose
        try:
            eigvals, eigvecs = np.linalg.eigh(X)
            # Clip negative eigenvalues
            eigvals = np.clip(eigvals, 0, None)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Make symmetric again
            X = (X + X.T) / 2.0
            # Normalize diagonal to 1
            diag = np.sqrt(np.diag(X))
            diag[diag < 1e-10] = 1e-10
            X = X / np.outer(diag, diag)
            return X
        except Exception:
            return X
    
    X = _project_correlation(X)
    
    # Gradient ascent on Tr(A X)
    step_size = 0.1
    for i in range(iters):
        if time.time() > deadline:
            break
        # Gradient is 2*A (for symmetric A)
        grad = 2.0 * A
        X = X + step_size * grad
        X = _project_correlation(X)
    
    # Random hyperplane rounding
    # Generate random unit vector
    u = np.random.randn(n)
    u = u / np.linalg.norm(u)
    # Project vertices onto u in the embedded space
    # Each vertex i is represented by row i of X (or column)
    # The embedding is: v_i = row i of X
    # Projection: v_i @ u
    projections = X @ u
    labels = (projections > 0).astype(np.int8)
    
    if labels.sum() == 0 or labels.sum() == n:
        labels = (projections > np.median(projections)).astype(np.int8)
    
    return labels


def _greedy_improve_fast(adj, labels, deadline):
    """Fast greedy improvement with deadline guard."""
    n = len(labels)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    deg = adj_int.sum(axis=1)
    
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


def _simulated_annealing(adj, labels, deadline, max_iter=5000):
    """
    Simulated annealing to escape local optima.
    Returns a 0/1 label array.
    """
    n = len(labels)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    deg = adj_int.sum(axis=1)
    
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    T_start = 1.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iter))
    T = T_start
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # Compute delta cut if we flip v
        same = adj_int[v][labels == labels[v]].sum()
        other = deg[v] - same
        delta = (other - same)  # change in cut size if we flip v
        
        # Accept with probability based on delta and temperature
        if delta > 0 or np.random.rand() < np.exp(delta / max(T, 1e-10)):
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        T *= cooling
    
    return best_labels


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    # Use a safety margin: stop searching 10 seconds before the 120s limit
    deadline = start_time + 110.0
    
    # Convert to numpy array once
    adj = np.asarray(adjacency, dtype=np.int8)
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + greedy
    labels_spec = _spectral_init(adj, n)
    labels_spec = _greedy_improve_fast(adj, labels_spec, deadline)
    cut_spec = _cut_size(adj, labels_spec)
    if cut_spec > best_cut:
        best_cut = cut_spec
        best_labels = labels_spec.copy()
    
    # Strategy 2: Goemans-Williamson style + greedy
    labels_gm = _goemans_maxwell(adj, n, deadline)
    labels_gm = _greedy_improve_fast(adj, labels_gm, deadline)
    cut_gm = _cut_size(adj, labels_gm)
    if cut_gm > best_cut:
        best_cut = cut_gm
        best_labels = labels_gm.copy()
    
    # Strategy 3: SDP initialization + greedy
    if time.time() < deadline - 10.0:
        labels_sdp = _sdp_init(adj, n, deadline - 5.0, iters=100)
        labels_sdp = _greedy_improve_fast(adj, labels_sdp, deadline)
        cut_sdp = _cut_size(adj, labels_sdp)
        if cut_sdp > best_cut:
            best_cut = cut_sdp
            best_labels = labels_sdp.copy()
    
    # Strategy 4: Alternating + greedy
    labels_alt = np.arange(n) % 2
    labels_alt = _greedy_improve_fast(adj, labels_alt, deadline)
    cut_alt = _cut_size(adj, labels_alt)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()
    
    # Strategy 5: Random restarts with greedy
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = _greedy_improve_fast(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Strategy 6: Simulated annealing from best found
    if time.time() < deadline - 5.0:
        labels_sa = _simulated_annealing(adj, best_labels, deadline, max_iter=3000)
        cut_sa = _cut_size(adj, labels_sa)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
    
    # Final greedy polish
    if time.time() < deadline:
        best_labels = _greedy_improve_fast(adj, best_labels, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
