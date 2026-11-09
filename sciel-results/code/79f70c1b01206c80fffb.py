import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    # Try deterministic algebraic constructions first, then local search
    for target_n in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break

        # Try Paley-type construction
        if target_n == 20:
            adj = _paley_like_graph(20)
            if _is_valid_fast(adj, 20, deadline):
                return _adj_to_result(adj, 20)

        # Try local search from random start
        num_attempts = 30 if target_n >= 19 else 50
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break

            rng = random.Random(attempt * 7919 + 13)
            n = target_n
            adj = np.zeros((n, n), dtype=np.uint8)
            
            # Random initial graph
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1

            # Local search to eliminate regular induced subgraphs
            adj = _local_search(adj, n, deadline)
            
            if _is_valid_fast(adj, n, deadline):
                return _adj_to_result(adj, n)

    # Fallback: return a trivially valid small graph
    return {'n': 5, 'edges': []}

def _paley_like_graph(n):
    """
    Construct a graph using quadratic residues modulo a prime (Paley graph).
    For n=20, use a modified construction.
    """
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Use a prime p >= n for Paley construction
    # Find smallest prime >= n
    p = n
    while not _is_prime(p):
        p += 1
    
    # Quadratic residues modulo p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Build graph on vertices 0..n-1
    for i in range(n):
        for j in range(i + 1, n):
            # Edge if (j - i) mod p is a quadratic residue
            diff = (j - i) % p
            if diff in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _local_search(adj, n, deadline):
    """
    Local search: find violations (regular induced subgraphs of size >= 6)
    and flip edges to break them.
    """
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Find a violating subset
        violation = _find_violation(adj, n)
        if violation is None:
            break
        
        # Flip a random edge in the violating subset to break regularity
        subset = violation
        u, v = random.sample(subset, 2)
        adj[u, v] = 1 - adj[u, v]
        adj[v, u] = adj[u, v]
    
    return adj

def _find_violation(adj, n):
    """
    Find a subset of size >= 6 that induces a regular graph.
    Returns the subset as a list, or None if no violation found.
    """
    if n < 6:
        return None
    
    # Check size 6 first (most numerous, most likely to be regular in bad graphs)
    for subset in combinations(range(n), 6):
        mask = np.zeros(n, dtype=bool)
        mask[list(subset)] = True
        degs = (adj & mask[None, :]).sum(axis=1)
        if len(set(degs[mask].tolist())) == 1:
            return list(subset)
    
    # Check larger sizes (fewer subsets, but still need to check)
    for size in range(7, n + 1):
        for subset in combinations(range(n), size):
            mask = np.zeros(n, dtype=bool)
            mask[list(subset)] = True
            degs = (adj & mask[None, :]).sum(axis=1)
            if len(set(degs[mask].tolist())) == 1:
                return list(subset)
    
    return None

def _is_valid_fast(adj, n, deadline):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Uses vectorized NumPy operations.
    Returns True if valid, False if violation found or timeout.
    """
    if n < 6:
        return True

    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                return False
            
            mask = np.zeros(n, dtype=bool)
            mask[list(subset)] = True
            degs = (adj & mask[None, :]).sum(axis=1)
            
            if len(set(degs[mask].tolist())) == 1:
                return False

    return True

def _adj_to_result(adj, n):
    """Convert adjacency matrix to the required result format."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
