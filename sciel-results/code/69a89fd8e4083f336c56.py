import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _random_1_factorization(n):
    """Generate a random 1-factorization of K_n (n must be even).
    Returns a list of n-1 perfect matchings, each a list of (u,v) pairs."""
    vertices = list(range(n))
    matchings = []
    for i in range(n - 1):
        # Use the standard cyclic construction with random rotation
        # Or just build a random perfect matching from remaining edges
        # Simpler: use a fixed 1-factorization and randomly permute matchings
        # Actually, let's use a proper random 1-factorization via random permutation
        
        # Use the known 1-factorization: for round r in 0..n-2,
        # fix vertex r, pair up the rest
        # But to get different factorizations, we can randomly assign vertices to positions
        
        # Simple approach: generate random perfect matching repeatedly
        # This doesn't guarantee a full factorization, so let's use the cyclic method
        # with a random initial permutation
        
        # Better: use the standard 1-factorization of K_n
        # Fix vertex 0, then for round r, pair (r, r+1), (r+2, r-1), etc. mod (n-1)
        # But we want random ones. Let's just use the cyclic one with random vertex labels.
        
        pass
    
    # Use the standard 1-factorization with a random vertex permutation
    perm = list(range(n))
    random.shuffle(perm)
    
    # Standard 1-factorization: vertices 0..n-1
    # Round i: fix vertex i, pair up the remaining n-1 vertices
    # The remaining vertices in order: i+1, i+2, ..., n-1, 0, 1, ..., i-1
    # Pair: (i+1, i-1), (i+2, i-2), ..., (i+(n/2), i-(n/2))
    
    matchings = []
    for i in range(n - 1):
        matching = []
        # The n-1 vertices other than i, in cyclic order starting from i+1
        others = []
        for j in range(1, n):
            others.append((i + j) % n)
        
        # Pair them up: others[0] with others[n-2], others[1] with others[n-3], etc.
        for j in range(n // 2):
            u = others[j]
            v = others[n - 2 - j]
            matching.append((perm[u], perm[v]))
        matchings.append(matching)
    
    return matchings

def _is_regular_subgraph_fast(adj_bits, n, subset):
    """Check if induced subgraph on subset is regular. adj_bits[i] is a bitmask."""
    k = len(subset)
    if k < 6:
        return False
    degrees = [0] * k
    for i in range(k):
        u = subset[i]
        mask = adj_bits[u]
        # Count neighbors in subset
        cnt = 0
        for j in range(k):
            if i != j:
                v = subset[j]
                if mask & (1 << v):
                    cnt += 1
        degrees[i] = cnt
    d0 = degrees[0]
    for d in degrees[1:]:
        if d != d0:
            return False
    return True

def _find_violation(adj_bits, n, min_size=6):
    """Find a regular induced subgraph of size >= min_size.
    Returns (subset_list, size) or (None, 0)."""
    for k in range(min_size, n + 1):
        for subset in combinations(range(n), k):
            if _is_regular_subgraph_fast(adj_bits, n, subset):
                return list(subset), k
    return None, 0

def _build_adj_from_matchings(n, selected_matchings):
    """Build adjacency bitmask array from selected perfect matchings."""
    adj_bits = [0] * n
    for matching in selected_matchings:
        for (u, v) in matching:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
    return adj_bits

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_result = None
    
    # Try n = 21 first (odd, so we can't use perfect matchings directly)
    # For odd n, we can use a near-1-factorization (one vertex unmatched per matching)
    # Or we can try n=20 (even) which is known to be achievable.
    
    for n_target in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5.0:
            break
        
        found = False
        
        if n_target % 2 == 0:
            # Even n: use 1-factorization approach
            # Try many random 1-factorizations and random subsets of matchings
            max_attempts = 200
            for attempt in range(max_attempts):
                if time.time() > deadline - 2.0:
                    break
                
                factorization = _random_1_factorization(n_target)
                # Randomly select a subset of matchings
                # Try different densities
                n_matchings = n_target - 1
                # Random number of matchings to include
                num_selected = random.randint(1, n_matchings)
                selected_idx = random.sample(range(n_matchings), num_selected)
                selected = [factorization[i] for i in selected_idx]
                
                adj_bits = _build_adj_from_matchings(n_target, selected)
                
                # Check for violations
                subset, size = _find_violation(adj_bits, n_target, 6)
                if subset is None:
                    # Valid!
                    edges = []
                    for u in range(n_target):
                        for v in range(u + 1, n_target):
                            if adj_bits[u] & (1 << v):
                                edges.append([u, v])
                    if best_result is None or n_target > best_result[0]:
                        best_result = (n_target, edges)
                    found = True
                    break
        
        else:
            # Odd n: use a different approach
            # We can add one extra vertex to an even-vertex graph
            # Or use random graph with local search
            # For now, use random graph with faster checking
            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() > deadline - 2.0:
                    break
                
                # Random graph
                adj_bits = [0] * n_target
                for u in range(n_target):
                    for v in range(u + 1, n_target):
                        if random.random() < 0.5:
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                
                # Quick check
                subset, size = _find_violation(adj_bits, n_target, 6)
                if subset is None:
                    edges = []
                    for u in range(n_target):
                        for v in range(u + 1, n_target):
                            if adj_bits[u] & (1 << v):
                                edges.append([u, v])
                    if best_result is None or n_target > best_result[0]:
                        best_result = (n_target, edges)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        n_found, edges_found = best_result
        return {'n': n_found, 'edges': edges_found}
    else:
        return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
