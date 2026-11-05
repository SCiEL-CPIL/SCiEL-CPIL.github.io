import time
import random
import numpy as np
from itertools import combinations

def _verify_no_regular_subset(adj_matrix, n):
    """Check if any subset of size >= 6 induces a regular graph using NumPy."""
    # Precompute: for each possible subset mask, we need to check degree uniformity
    # Instead, iterate over all subsets of size 6..n and check degrees
    
    # For efficiency, we'll check subsets of size 6 first (most likely to be regular),
    # then larger sizes
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            # Get induced subgraph adjacency
            sub_adj = adj_matrix[np.ix_(subset, subset)]
            # Compute degrees within the induced subgraph
            degrees = sub_adj.sum(axis=1)
            # Check if all degrees are equal
            if np.all(degrees == degrees[0]):
                return False  # Found a regular subset
    return True

def _construct_comparability(n, rng):
    """Construct a comparability graph from a random total order.
    
    Vertices are ordered 0..n-1. Edge (i,j) exists if |i-j| > threshold.
    This creates a graph where vertices at different positions have different
    degrees, making regular induced subgraphs unlikely.
    """
    # Generate a random permutation to create a non-trivial order
    order = rng.permutation(n)
    # Create adjacency matrix
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Try different thresholds to find one that works
    # The key insight: if we use a threshold that depends on position,
    # we can break regularity
    
    # Strategy: for each pair (i, j), include edge if the "distance" in the
    # permuted order exceeds a threshold. But to make it more robust, we can
    # use a weighted approach.
    
    # Alternative structural approach: use a random graph but with degree 
    # heterogeneity built in. Let's try: assign each vertex a random weight,
    # and connect i,j if weight[i] + weight[j] > threshold.
    
    weights = rng.random(n)
    # Sort weights to create a natural ordering
    sorted_idx = np.argsort(weights)
    rank = np.zeros(n, dtype=int)
    for r, idx in enumerate(sorted_idx):
        rank[idx] = r
    
    # Edge (i, j) exists if |rank[i] - rank[j]| > n/3
    threshold = max(1, n // 3)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(rank[i] - rank[j]) > threshold:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj

def _construct_random_heterogeneous(n, rng):
    """Construct a random graph with built-in degree heterogeneity.
    
    Assign each vertex a random 'activity' level. Higher activity vertices
    have higher connection probability. This creates a natural degree spread.
    """
    # Assign random activity levels
    activities = rng.random(n)
    # Scale activities to create meaningful differences
    activities = activities * 0.6 + 0.2  # Range [0.2, 0.8]
    
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            # Connection probability based on both vertices' activities
            p = activities[i] * activities[j] * 2.5  # Scale up
            if rng.random() < min(0.95, p):
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj

def _construct_paley_like(n, rng):
    """Construct a Paley-like graph using quadratic residues, modified for heterogeneity."""
    # Use a prime p >= n
    primes = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    p = primes[0]
    for prime in primes:
        if prime >= n:
            p = prime
            break
    
    # Quadratic residues mod p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in residues or (i - j) % p in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj

def _construct_mixed(n, rng):
    """Mix multiple construction strategies."""
    strategy = rng.randint(0, 2)
    if strategy == 0:
        return _construct_comparability(n, rng)
    elif strategy == 1:
        return _construct_random_heterogeneous(n, rng)
    else:
        return _construct_paley_like(n, rng)

def _local_search(adj, n, rng, max_steps=500):
    """Local search: try to fix regular subsets by flipping edges."""
    adj = adj.copy()
    
    for step in range(max_steps):
        # Find a regular subset of size 6
        found_subset = None
        for subset in combinations(range(n), 6):
            sub_adj = adj[np.ix_(subset, subset)]
            degrees = sub_adj.sum(axis=1)
            if np.all(degrees == degrees[0]):
                found_subset = subset
                break
        
        if found_subset is None:
            # No regular 6-subset found, but need to check larger subsets too
            # For now, assume if no 6-subset is regular, we're good (approximation)
            # Actually we need to verify all sizes
            if _verify_no_regular_subset(adj, n):
                return adj
            # Otherwise, try flipping a random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i != j:
                adj[i, j] = 1 - adj[i, j]
                adj[j, i] = adj[i, j]
            continue
        
        # Found a regular subset. Try to break it by flipping an edge within it
        subset_list = list(found_subset)
        # Pick a random pair in the subset to flip
        a_idx = rng.randint(0, len(subset_list) - 1)
        b_idx = rng.randint(0, len(subset_list) - 1)
        if a_idx == b_idx:
            b_idx = (b_idx + 1) % len(subset_list)
        
        u = subset_list[a_idx]
        v = subset_list[b_idx]
        adj[u, v] = 1 - adj[u, v]
        adj[v, u] = adj[u, v]
    
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    best_n = 5
    best_adj = None
    
    # Try to construct valid graphs for n = 21, 20, 19, ...
    for n in range(21, 5, -1):
        if time.time() > deadline - 10:
            break
        
        found_valid = False
        
        # Try multiple construction strategies
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(attempt * 12345 + n * 67890)
            
            # Try different construction methods
            if attempt < 50:
                adj = _construct_comparability(n, rng)
            elif attempt < 100:
                adj = _construct_random_heterogeneous(n, rng)
            elif attempt < 150:
                adj = _construct_paley_like(n, rng)
            else:
                adj = _construct_mixed(n, rng)
            
            # Quick check: verify no regular subset of size 6
            has_regular_6 = False
            for subset in combinations(range(n), 6):
                sub_adj = adj[np.ix_(subset, subset)]
                degrees = sub_adj.sum(axis=1)
                if np.all(degrees == degrees[0]):
                    has_regular_6 = True
                    break
            
            if has_regular_6:
                # Try local search to fix it
                adj = _local_search(adj, n, rng, max_steps=100)
            
            # Full verification
            if _verify_no_regular_subset(adj, n):
                found_valid = True
                break
        
        if found_valid:
            best_n = n
            best_adj = adj
            break
        else:
            # Store best attempt for this n (the one closest to valid)
            if best_adj is None or n > best_n:
                best_n = n
                best_adj = adj
    
    # If we didn't find a valid graph, try smaller n with more attempts
    if best_adj is None or not _verify_no_regular_subset(best_adj, best_n):
        for n in range(19, 5, -1):
            if time.time() > deadline - 5:
                break
            
            for attempt in range(300):
                if time.time() > deadline - 3:
                    break
                
                rng = random.Random(attempt * 98765 + n * 54321)
                adj = _construct_random_heterogeneous(n, rng)
                
                if _verify_no_regular_subset(adj, n):
                    best_n = n
                    best_adj = adj
                    break
            
            if best_adj is not None and _verify_no_regular_subset(best_adj, best_n):
                break
    
    # Convert adjacency matrix to edge list
    if best_adj is not None and best_n >= 6:
        edges = []
        for i in range(best_n):
            for j in range(i + 1, best_n):
                if best_adj[i, j] == 1:
                    edges.append([i, j])
        return {'n': best_n, 'edges': edges}
    
    # Fallback: empty graph on 5 vertices
    return {'n': 5, 'edges': []}
