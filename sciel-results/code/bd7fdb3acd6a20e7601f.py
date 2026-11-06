import numpy as np
import time

def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # adj is a boolean or int matrix
    # labels is a 0/1 array
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)

def _spectral_init_gw(adj, n, k=10, num_hyperplanes=20):
    """
    Generate multiple initial partitions using spectral embedding and random hyperplane rounding.
    Returns a list of label arrays.
    """
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Use the smallest non-trivial eigenvectors
        # The first eigenvector is constant (for connected graph)
        # We take the next k eigenvectors
        k_eff = min(k, n - 1)
        V = eigenvectors[:, 1:1+k_eff]
        
        candidates = []
        for _ in range(num_hyperplanes):
            # Random hyperplane in the embedded space
            random_vec = np.random.randn(k_eff)
            projections = V @ random_vec
            labels = (projections > 0).astype(np.int8)
            
            # Ensure both sides are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels = (projections > np.median(projections)).astype(np.int8)
            
            candidates.append(labels)
        
        return candidates
    except Exception:
        return [np.arange(n) % 2]

def _greedy_improve(adj, labels, deadline):
    """
    Fast greedy improvement: flip a vertex if it increases the cut size.
    Repeats until no improvement is found or deadline is reached.
    """
    n = len(labels)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    
    # Precompute degrees
    deg = adj_int.sum(axis=1)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        for v in range(n):
            if time.time() > deadline:
                return labels
            
            # Count edges to same side and other side
            # same_side_count = sum(adj_int[v, labels == labels[v]])
            # other_side_count = sum(adj_int[v, labels != labels[v]])
            
            # Vectorized calculation for speed
            same_mask = (labels == labels[v])
            same_count = adj_int[v, same_mask].sum()
            other_count = deg[v] - same_count
            
            # If flipping v increases the cut, flip it
            # Change in cut = other_count - same_count
            if other_count > same_count:
                labels[v] = 1 - labels[v]
                improved = True
                
    return labels

def _simulated_annealing(adj, labels, deadline, max_iter=10000):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    deg = adj_int.sum(axis=1)
    
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    T_start = 5.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iter))
    T = T_start
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # Compute delta cut if we flip v
        same_mask = (labels == labels[v])
        same_count = adj_int[v, same_mask].sum()
        other_count = deg[v] - same_count
        delta = other_count - same_count
        
        # Accept with probability
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
    deadline = start_time + 110.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=np.int8)
    
    best_labels = None
    best_cut = -1
    
    # 1. Generate multiple spectral GW candidates
    candidates = _spectral_init_gw(adj, n, k=15, num_hyperplanes=50)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        refined = _greedy_improve(adj, cand, deadline)
        cut = _cut_size(adj, refined)
        if cut > best_cut:
            best_cut = cut
            best_labels = refined.copy()
    
    # 2. Add a few random restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        refined = _greedy_improve(adj, labels, deadline)
        cut = _cut_size(adj, refined)
        if cut > best_cut:
            best_cut = cut
            best_labels = refined.copy()
    
    # 3. Simulated Annealing on the best found
    if time.time() < deadline - 10.0:
        sa_labels = _simulated_annealing(adj, best_labels, deadline, max_iter=20000)
        cut_sa = _cut_size(adj, sa_labels)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = sa_labels.copy()
    
    # 4. Final greedy polish
    if time.time() < deadline:
        best_labels = _greedy_improve(adj, best_labels, deadline)
    
    return best_labels
