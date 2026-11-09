"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Use spectral method (Fiedler vector) to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    idx = np.argsort(eigvals)
    fiedler = eigvecs[:, idx[1]]
    labels = (fiedler >= 0).astype(np.int32)
    
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _gwg_spectral_partition(adjacency, n_hyperplanes=50):
    """
    Generate high-quality initial partitions using Goemans-Williamson style
    random hyperplane cuts from Laplacian eigenvectors.
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    idx = np.argsort(eigvals)
    eigvecs_sorted = eigvecs[:, idx]
    
    k = min(10, n - 1)
    embedding = eigvecs_sorted[:, 1:k+1]
    
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embedding_normalized = embedding / norms
    
    best_labels = None
    best_cut = -1
    
    for _ in range(n_hyperplanes):
        normal = np.random.randn(k)
        normal /= (np.linalg.norm(normal) + 1e-10)
        projections = embedding_normalized @ normal
        labels = (projections >= 0).astype(np.int32)
        
        if labels.sum() == 0 or labels.sum() == n:
            continue
        
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
    
    if best_labels is None:
        fiedler = eigvecs_sorted[:, 1]
        best_labels = (fiedler >= 0).astype(np.int32)
        if best_labels.sum() == 0 or best_labels.sum() == n:
            best_labels = np.random.randint(0, 2, size=n)
    
    return best_labels


def _random_labels(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, size=n)


def _vectorized_local_search(adjacency, labels, deadline, max_rounds=200):
    """
    Vectorized local search: compute all gains at once per round using matrix ops.
    Gain of flipping vertex v = (num same-partition neighbors) - (num opposite-partition neighbors)
    If gain > 0, flipping increases cut size.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    # Precompute: for each vertex, how many neighbors are in the same partition
    # same[v] = sum over neighbors u of (labels[u] == labels[v])
    # We can compute: for each vertex v, count neighbors with same label
    # Using matrix: A @ (labels == labels[v]) but that's per-vertex
    
    # Better approach: 
    # For vertex v with label l: same_count = sum(A[v,:]) where labels == l
    #                    opp_count = sum(A[v,:]) where labels != l
    # gain = same_count - opp_count = 2*same_count - degree[v]
    
    degrees = A.sum(axis=1).astype(np.int32)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Compute for all vertices simultaneously
        # labels is 0/1. For vertex v:
        #   if labels[v] == 0: same_count = A[v, labels==0].sum()
        #   if labels[v] == 1: same_count = A[v, labels==1].sum()
        # 
        # Vectorized: 
        # same_matrix[i,j] = A[i,j] if labels[i]==labels[j] else 0
        # same_counts = same_matrix.sum(axis=1)
        # 
        # But we can be smarter:
        # For label 0 vertices: same_count[i] = A[i, :].sum() where labels==0
        # For label 1 vertices: same_count[i] = A[i, :].sum() where labels==1
        
        # Compute same-partition neighbor counts
        # Create mask: same[i,j] = (labels[i] == labels[j])
        same_mask = (labels[:, None] == labels[None, :])
        same_counts = (A * same_mask).sum(axis=1)
        
        # gain[v] = same_counts[v] - (degrees[v] - same_counts[v]) = 2*same_counts[v] - degrees[v]
        gains = 2 * same_counts - degrees
        
        # Find vertices where flipping improves
        flip_mask = gains > 0
        if not flip_mask.any():
            break
        
        # Flip all improving vertices simultaneously (parallel update)
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels


def _adaptive_simulated_annealing(adjacency, labels, deadline, max_iterations=10000):
    """
    Simulated annealing with adaptive cooling based on acceptance rate.
    Uses vectorized gain computation for speed.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    degrees = A.sum(axis=1).astype(np.int32)
    
    # Adaptive temperature: start high, cool based on acceptance rate
    T = 10.0
    T_min = 0.01
    T_max = 20.0
    cooling_rate = 0.95
    heating_rate = 1.1
    
    # Track acceptance in a window
    window_size = 100
    accept_count = 0
    total_count = 0
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Adaptive temperature adjustment
        if total_count >= window_size:
            acceptance_rate = accept_count / total_count
            if acceptance_rate < 0.1:
                # Too few accepts - heat up slightly
                T = min(T * heating_rate, T_max)
            elif acceptance_rate > 0.5:
                # Too many accepts - cool faster
                T = max(T * (cooling_rate * 0.9), T_min)
            else:
                # Normal cooling
                T = max(T * cooling_rate, T_min)
            accept_count = 0
            total_count = 0
        
        # Pick a random vertex
        v = np.random.randint(n)
        old_label = labels[v]
        
        # Compute gain for flipping v
        # same_count = number of neighbors with same label as v
        # opp_count = degree - same_count
        # gain = same_count - opp_count = 2*same_count - degree[v]
        if degrees[v] == 0:
            delta = 0
        else:
            same_count = (A[v] * (labels == old_label)).sum()
            delta = 2 * same_count - degrees[v]
        
        total_count += 1
        
        if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
            labels[v] = 1 - old_label
            current_cut += delta
            accept_count += 1
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def _multi_start_sa(adjacency, candidates, deadline, n_starts=3):
    """
    Run multiple SA restarts from different candidates, keeping the best.
    """
    best_labels = None
    best_cut = -1
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Give more time to earlier starts
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.8
        
        sa_labels, sa_cut = _adaptive_simulated_annealing(adjacency, cand, sa_deadline)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """Main partition function using spectral initialization, vectorized local search, and multi-start SA."""
    n = len(adjacency)
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Standard spectral partition (Fiedler vector)
    spec_labels = _spectral_partition(adjacency)
    candidates.append(spec_labels)
    
    # 2. Goemans-Williamson style spectral partition with random hyperplanes
    gw_labels = _gwg_spectral_partition(adjacency, n_hyperplanes=30)
    candidates.append(gw_labels)
    
    # 3. Random partitions
    for _ in range(3):
        rand_labels = _random_labels(n)
        candidates.append(rand_labels)
    
    # 4. Alternating partition
    alt_labels = np.arange(n) % 2
    candidates.append(alt_labels)
    
    # Phase 1: Vectorized local search on each candidate
    best_labels = None
    best_cut = -1
    
    ls_candidates = []
    for cand in candidates:
        if time.time() > deadline:
            break
        
        ls_labels = _vectorized_local_search(adjacency, cand, deadline)
        ls_cut = _cut_size(adjacency, ls_labels)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()
        
        ls_candidates.append(ls_labels)
    
    # Phase 2: Multi-start simulated annealing from top LS results
    # Take the top 3 distinct LS results as SA starts
    if time.time() < deadline - 15.0 and len(ls_candidates) > 0:
        # Sort by cut size and take top few
        ls_with_cuts = [(c, _cut_size(adjacency, c)) for c in ls_candidates]
        ls_with_cuts.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3 (or fewer if not enough)
        sa_starts = [c for c, _ in ls_with_cuts[:3]]
        
        # Remove duplicates (same label array)
        unique_starts = []
        seen = set()
        for s in sa_starts:
            key = tuple(s.tolist())
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)
        
        if unique_starts:
            sa_labels, sa_cut = _multi_start_sa(adjacency, unique_starts, deadline, n_starts=len(unique_starts))
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    # Phase 3: Final high-precision polish pass
    if time.time() < deadline - 5.0 and best_labels is not None:
        # Run vectorized local search with more rounds for final polish
        final_labels = _vectorized_local_search(adjacency, best_labels, deadline, max_rounds=500)
        final_cut = _cut_size(adjacency, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()
        
        # One more short SA burst if time permits
        if time.time() < deadline - 3.0:
            sa_labels, sa_cut = _adaptive_simulated_annealing(adjacency, best_labels, deadline, max_iterations=5000)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
            
            # Final polish after SA
            if time.time() < deadline - 1.0:
                final_labels = _vectorized_local_search(adjacency, best_labels, deadline, max_rounds=200)
                final_cut = _cut_size(adjacency, final_labels)
                if final_cut > best_cut:
                    best_cut = final_cut
                    best_labels = final_labels.copy()
    
    # Ensure we return a valid label array
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
