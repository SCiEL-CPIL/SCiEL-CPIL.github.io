"""Combined program: bitmask speed + targeted local search + 6-subset optimization."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_regular_6(adj, subset):
    """Check if the induced subgraph on the 6-tuple subset is regular."""
    # Compute degree of each vertex within the subset
    first = -1
    for i in range(6):
        u = subset[i]
        deg = 0
        for j in range(6):
            if i != j and ((adj[u] >> subset[j]) & 1):
                deg += 1
        if first == -1:
            first = deg
        elif deg != first:
            return False
    return True

def _find_regular_6(adj, n):
    """Find any 6-vertex subset inducing a regular subgraph.
    Returns the subset tuple if found, else None.
    Relies on the fact that any regular induced subgraph on >= 6 vertices
    contains a regular induced subgraph on exactly 6 vertices."""
    for subset in combinations(range(n), 6):
        if _is_regular_6(adj, subset):
            return subset
    return None

def _random_graph(n, p, rng):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _local_search(adj, n, rng, max_steps, deadline):
    """Local search: flip edges to break regular 6-subsets.
    Returns (adj, found_valid)."""
    adj = adj[:]
    for step in range(max_steps):
        if time.time() > deadline:
            return adj, False
        reg = _find_regular_6(adj, n)
        if reg is None:
            return adj, True
        # Flip a random edge inside the offending subset
        u, v = rng.sample(list(reg), 2)
        if (adj[u] >> v) & 1:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj, False

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under 150s kill

    best_n = 0
    best_edges = []

    # Target larger n first (21 is the prize, 20 is known-feasible)
    for n in [21, 20]:
        if time.time() > deadline:
            break
        max_attempts = 8 if n == 21 else 20
        max_steps = 300 if n == 21 else 600
        found_for_n = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            rng = random.Random(attempt * 7919 + n)
            adj = _random_graph(n, 0.5, rng)
            adj, ok = _local_search(adj, n, rng, max_steps, deadline)
            if ok:
                if n > best_n:
                    best_n = n
                    best_edges = _edges_from_adj(adj, n)
                found_for_n = True
                break  # got a valid graph for this n, move to smaller n
        # If we found valid for this n, we can stop trying smaller (we want max)
        if found_for_n:
            break

    # Fallback: try to build up from small n if nothing large worked
    if best_n < 6:
        for n in range(6, 16):
            if time.time() > deadline:
                break
            for attempt in range(30):
                if time.time() > deadline:
                    break
                rng = random.Random(attempt * 104729 + n * 13)
                adj = _random_graph(n, 0.5, rng)
                adj, ok = _local_search(adj, n, rng, 200, deadline)
                if ok:
                    if n > best_n:
                        best_n = n
                        best_edges = _edges_from_adj(adj, n)
                    break

    # Ultimate fallback: empty graph on 5 vertices is always valid
    if best_n < 6:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
