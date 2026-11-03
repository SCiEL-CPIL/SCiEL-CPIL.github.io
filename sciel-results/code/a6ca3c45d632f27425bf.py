"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    return np.sum(a * b[np.ix_(p, p)])


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in total cost when swapping facilities at positions i and j.
    """
    pi = p[i]
    pj = p[j]
    
    delta = 0.0
    
    # Diagonal terms
    delta += (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
    
    # Off-diagonal i,j terms
    delta += (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Cross terms with k != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    
    if len(ks) > 0:
        diff1 = b[pj, p[ks]] - b[pi, p[ks]]
        diff2 = a[i, ks] - a[j, ks]
        delta += np.dot(diff2, diff1)
        
        diff3 = b[p[ks], pj] - b[p[ks], pi]
        diff4 = a[ks, i] - a[ks, j]
        delta += np.dot(diff4, diff3)
    
    return delta


def _spectral_init(a, b, k=10, seed=42):
    """
    Generate initial permutation using spectral projection.
    Project flow and distance matrices onto top-k principal components,
    then use rank-matching on the projected matrices.
    """
    n = len(a)
    rng = np.random.RandomState(seed)
    
    # Center the matrices
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    
    # SVD decomposition
    Ua, Sa, Vt_a = np.linalg.svd(a_centered, full_matrices=False)
    Ub, Sb, Vt_b = np.linalg.svd(b_centered, full_matrices=False)
    
    # Take top-k components
    Ua_k = Ua[:, :k]
    Ub_k = Ub[:, :k]
    
    # Project onto k-dimensional space
    # For facility i, its "position" in spectral space is Ua_k[i, :]
    # For location j, its "position" is Ub_k[j, :]
    
    # Create surrogate cost matrix: distance between spectral projections
    # This captures non-linear structural correlations
    surrogate = np.zeros((n, n), dtype=np.float64)
    
    # Compute pairwise distances in spectral space
    # d(i,j) = ||Ua_k[i,:] - Ub_k[j,:]||^2
    # = ||Ua_k[i,:]||^2 + ||Ub_k[j,:]||^2 - 2 * Ua_k[i,:] @ Ub_k[j,:]^T
    
    norm_a = np.sum(Ua_k ** 2, axis=1)  # (n,)
    norm_b = np.sum(Ub_k ** 2, axis=1)  # (n,)
    
    cross_term = Ua_k @ Ub_k.T  # (n, n)
    
    surrogate = norm_a[:, np.newaxis] + norm_b[np.newaxis, :] - 2 * cross_term
    
    # Rank-matching: assign high-flow facilities to low-distance locations
    # Sort facilities by their row sum in original flow matrix (descending)
    flow_rank = np.argsort(-a.sum(axis=1), kind="stable")
    
    # Sort locations by their row sum in surrogate matrix (ascending - prefer low cost)
    loc_rank = np.argsort(surrogate.sum(axis=1), kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[flow_rank] = loc_rank
    
    return p


def _local_search_fast(a, b, p, max_time=100.0):
    """
    Fast 2-opt local search with vectorized delta computation.
    """
    n = len(p)
    p = p.copy()
    start_time = time.time()
    
    cost = _qap_cost(a, b, p)
    
    # Precompute row sums for quick delta estimation
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() - start_time < max_time:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Find best improving swap
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return p, cost


def _perturb(p, n, strategy=0, seed=42):
    """
    Perturb a permutation using different strategies.
    """
    rng = np.random.RandomState(seed)
    p2 = p.copy()
    
    if strategy == 0:
        # Single random swap
        i, j = rng.choice(n, size=2, replace=False)
        p2[i], p2[j] = p2[j], p2[i]
    elif strategy == 1:
        # Multiple random swaps
        for _ in range(rng.randint(3, 7)):
            i, j = rng.choice(n, size=2, replace=False)
            p2[i], p2[j] = p2[j], p2[i]
    elif strategy == 2:
        # Or-opt: reverse a segment
        i = rng.randint(n)
        length = rng.randint(2, min(6, n - i))
        segment = p2[i:i+length].copy()
        segment = segment[::-1]
        p2[i:i+length] = segment
    elif strategy == 3:
        # 2-opt: reverse two segments
        i1 = rng.randint(n)
        i2 = rng.randint(n)
        if i1 > i2:
            i1, i2 = i2, i1
        length1 = rng.randint(2, min(4, n - i1))
        length2 = rng.randint(2, min(4, n - i2))
        
        seg1 = p2[i1:i1+length1].copy()
        seg2 = p2[i2:i2+length2].copy()
        
        p2[i1:i1+length1] = seg1[::-1]
        p2[i2:i2+length2] = seg2[::-1]
    else:
        # Random shuffle of a subset
        subset_size = rng.randint(10, 20)
        indices = rng.choice(n, size=subset_size, replace=False)
        vals = p2[indices].copy()
        rng.shuffle(vals)
        p2[indices] = vals
    
    return p2


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate multiple diverse initial solutions using spectral projection
    # with different random seeds and k values
    best_p = None
    best_cost = np.inf
    
    # Try different spectral configurations
    configs = [
        (10, 42),
        (15, 123),
        (20, 456),
        (8, 789),
        (12, 321),
    ]
    
    for k, seed in configs:
        if time.time() > deadline - 15:
            break
        
        # Generate spectral initialization
        p = _spectral_init(a, b, k=k, seed=seed)
        
        # Local search
        remaining_time = deadline - time.time() - 5
        if remaining_time < 5:
            break
        
        p, cost = _local_search_fast(a, b, p, max_time=remaining_time)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # If we have more time, try iterated local search with perturbations
    if best_p is not None and time.time() < deadline - 10:
        for iteration in range(10):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            strategy = iteration % 5
            p2 = _perturb(best_p, n, strategy=strategy, seed=42 + iteration)
            
            # Local search on perturbed solution
            remaining_time = deadline - time.time() - 3
            if remaining_time < 3:
                break
            
            p2, cost2 = _local_search_fast(a, b, p2, max_time=remaining_time)
            
            if cost2 < best_cost:
                best_cost = cost2
                best_p = p2.copy()
    
    # Final polish: one more local search pass on the best solution
    if best_p is not None and time.time() < deadline - 2:
        best_p, best_cost = _local_search_fast(a, b, best_p, max_time=deadline - time.time() - 1)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
