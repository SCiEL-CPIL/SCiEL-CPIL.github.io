"""
Strategy: Cluster-Based Iterated Local Search for QAP.
1. Partition facilities and locations into K clusters using K-means on row sums.
2. Initial assignment aligns high-flow facility clusters with low-distance location clusters.
3. Iterated Local Search:
   a. Perform fast vectorized 2-opt local search.
   b. Apply a structured perturbation: randomly swap elements within or between clusters.
   c. Accept new solution if better, or with probability exp(-delta/T) (Simulated Annealing style).
4. Use a time budget to balance exploration (perturbation) and exploitation (local search).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym):
    """
    Compute the change in cost when swapping facilities at positions i and j.
    p[i] is the location of facility i.
    Uses the fast vectorized formula assuming b is symmetric.
    """
    Lx = p[i]
    Ly = p[j]
    if Lx == Ly:
        return 0
    diff_b = b[Ly, p] - b[Lx, p]
    coeff = a_sym[i, :] - a_sym[j, :]
    return np.dot(coeff, diff_b)

def _kmeans(X, k, n_iters=10, seed=0):
    """Simple K-means implementation."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    # Initialize centroids by picking k random points
    centroids = X[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iters):
        # Assign points to nearest centroid
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update centroids
        for c in range(k):
            mask = labels == c
            if np.sum(mask) > 0:
                centroids[c] = X[mask].mean(axis=0)
    return labels

def _cluster_init(a, b, n, k=8, seed=0):
    """
    Initialize permutation using cluster alignment.
    Facilities are clustered by flow row sums, locations by distance row sums.
    We align the 'highest flow' facility cluster with the 'lowest distance' location cluster, etc.
    """
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Normalize for k-means
    a_norm = (a_sums - a_sums.min()) / (a_sums.max() - a_sums.min() + 1e-9)
    b_norm = (b_sums - b_sums.min()) / (b_sums.max() - b_sums.min() + 1e-9)
    
    fac_clusters = _kmeans(a_norm.reshape(-1, 1), k, seed=seed)
    loc_clusters = _kmeans(b_norm.reshape(-1, 1), k, seed=seed+1)
    
    # Sort clusters by their mean value
    # Facility clusters: higher mean -> higher flow
    fac_cluster_means = [a_sums[fac_clusters == c].mean() for c in range(k)]
    # Location clusters: lower mean -> lower distance (central)
    loc_cluster_means = [b_sums[loc_clusters == c].mean() for c in range(k)]
    
    # We want to assign high flow clusters to low distance clusters.
    # Sort facility clusters descending by mean flow.
    fac_order = np.argsort(-fac_cluster_means)
    # Sort location clusters ascending by mean distance.
    loc_order = np.argsort(loc_cluster_means)
    
    # Map facility cluster index to location cluster index
    # fac_cluster fac_order[0] (highest flow) -> loc_cluster loc_order[0] (lowest dist)
    cluster_map = {}
    for i in range(k):
        cluster_map[fac_order[i]] = loc_order[i]
    
    # Assign facilities to locations
    p = np.empty(n, dtype=int)
    # For each facility, get its cluster, find the target location cluster,
    # and assign it to an available location in that cluster.
    
    # Precompute lists of locations in each cluster
    locs_in_cluster = {c: np.where(loc_clusters == c)[0] for c in range(k)}
    # Shuffle locations within each cluster to break ties randomly
    for c in range(k):
        np.random.shuffle(locs_in_cluster[c])
    # Use a pointer for each cluster to assign locations sequentially
    pointers = {c: 0 for c in range(k)}
    
    for i in range(n):
        fc = fac_clusters[i]
        lc = cluster_map[fc]
        locs = locs_in_cluster[lc]
        if pointers[lc] < len(locs):
            p[i] = locs[pointers[lc]]
            pointers[lc] += 1
        else:
            # Fallback: assign random location (shouldn't happen if cluster sizes match, but safety)
            p[i] = np.random.randint(n)
            
    # Ensure p is a valid permutation (unique locations)
    # The above might have collisions if cluster sizes don't match exactly or if we ran out.
    # A simple repair: find duplicates and fix them.
    # Actually, the pointer method guarantees uniqueness within a cluster, but if a cluster is exhausted,
    # we might assign a location already used.
    # Let's do a robust check and repair.
    used = set()
    for i in range(n):
        if p[i] in used:
            # Find a free location
            for loc in range(n):
                if loc not in used:
                    p[i] = loc
                    break
        used.add(p[i])
        
    return p

def _local_search(a, b, p, a_sym, time_limit):
    """
    Fast 2-opt local search.
    """
    n = len(p)
    best_cost = _qap_cost(a, b, p)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Random order of pairs to avoid bias
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for i, j in pairs:
            delta = _swap_delta(a, b, p, i, j, a_sym)
            if delta < 0:
                p[i], p[j] = p[j], p[i]
                best_cost += delta
                improved = True
    return p, best_cost

def _perturb(p, n, strength=10):
    """
    Apply a random perturbation of 'strength' swaps.
    """
    p_new = p.copy()
    for _ in range(strength):
        i, j = random.sample(range(n), 2)
        p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new

def _solve_tai256c(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    a_sym = a + a.T
    
    # Initial solution using cluster alignment
    p = _cluster_init(a, b, n, k=16, seed=42)
    current_cost = _qap_cost(a, b, p)
    
    best_p = p.copy()
    best_cost = current_cost
    
    # Iterated Local Search
    T = 1000.0  # Initial temperature
    cooling_rate = 0.95
    
    while time.time() < time_limit - 5.0:
        # Local search
        p, current_cost = _local_search(a, b, p, a_sym, time.time() + 2.0)
        
        if current_cost < best_cost:
            best_p = p.copy()
            best_cost = current_cost
            T = 1000.0  # Reset temperature on improvement
            
        # Perturbation
        p_new = _perturb(p, n, strength=random.randint(5, 15))
        new_cost = _qap_cost(a, b, p_new)
        
        # Acceptance criterion
        if new_cost < current_cost:
            p = p_new
            current_cost = new_cost
        else:
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    p = p_new
                    current_cost = new_cost
            T *= cooling_rate
            
        if time.time() > time_limit - 5.0:
            break

    # Final local search on best solution
    if time.time() < time_limit - 2.0:
        best_p, best_cost = _local_search(a, b, best_p, a_sym, time_limit - 1.0)

    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
