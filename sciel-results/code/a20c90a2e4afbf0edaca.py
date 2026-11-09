"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _is_regular_subset(adj, subset):
    """Check if the induced subgraph on 'subset' is regular using bitmask adj."""
    k = len(subset)
    idx = list(range(k))
    # For each vertex in subset, count neighbors also in subset
    degs = []
    for i in range(k):
        v = subset[i]
        cnt = 0
        for j in range(k):
            if i != j:
                u = subset[j]
                if adj[v] & (1 << u):
                    cnt += 1
        degs.append(cnt)
    return all(d == degs[0] for d in degs)

def _find_violation(adj, n):
    """Find any subset of size >=6 that induces a regular subgraph. Return subset or None."""
    # Check 6-sets first (most common violation)
    for subset in combinations(range(n), 6):
        if _is_regular_subset(adj, subset):
            return subset
    # Check 7-sets
    for subset in combinations(range(n), 7):
        if _is_regular_subset(adj, subset):
            return subset
    # Check 8-sets
    for subset in combinations(range(n), 8):
        if _is_regular_subset(adj, subset):
            return subset
    # Check 9-sets
    for subset in combinations(range(n), 9):
        if _is_regular_subset(adj, subset):
            return subset
    # Check 10-sets
    for subset in combinations(range(n), 10):
        if _is_regular_subset(adj, subset):
            return subset
    # Check larger sets up to n
    for size in range(11, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_subset(adj, subset):
                return subset
    return None

def _repair_graph(adj, n, max_iters=5000, deadline=None):
    """Try to repair a graph by removing edges from regular subsets."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return False
        viol = _find_violation(adj, n)
        if viol is None:
            return True
        # Remove a random edge within the violating subset
        edges_in_subset = []
        for i in range(len(viol)):
            for j in range(i + 1, len(viol)):
                u, v = viol[i], viol[j]
                if adj[u] & (1 << v):
                    edges_in_subset.append((u, v))
        if not edges_in_subset:
            # Shouldn't happen if it's regular with degree 0, but handle it
            # If all degrees are 0, any edge removal won't help; try adding? No, we can only remove.
            # Actually if it's a 0-regular graph (empty), we can't fix by removing edges.
            # This means we need a different approach. For now, shuffle and retry.
            return False
        u, v = random.choice(edges_in_subset)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    return False

def _verify(adj, n):
    """Full verification that no subset of size >=6 induces a regular subgraph."""
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_subset(adj, subset):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try n=20 first (known to be achievable)
    for target_n in [20, 21]:
        if time.time() > deadline:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Start with a random sparse graph
            # Use edge probability ~0.3 for sparsity
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.3:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if _repair_graph(adj, target_n, max_iters=3000, deadline=deadline):
                # Verify
                if _verify(adj, target_n):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
        if found:
            break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
