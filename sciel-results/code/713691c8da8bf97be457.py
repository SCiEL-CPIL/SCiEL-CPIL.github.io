"""Rewritten program: search for a graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import itertools
import numpy as np

# Deadline guard: evaluator kills at 150s; keep safety margin.
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _is_regular_subset(adj, subset):
    """Check if the induced subgraph on 'subset' (list of ints) is regular.
    Uses bit operations for speed."""
    n = len(subset)
    if n < 6:
        return False
    # Build adjacency restricted to subset
    # For each vertex in subset, count its degree within subset
    degs = []
    for v in subset:
        d = 0
        row = adj[v]
        for u in subset:
            if u != v and row[u]:
                d += 1
        degs.append(d)
    # Check if all degrees equal
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def _has_regular_6plus(adj, n):
    """Check if any subset of size >= 6 induces a regular subgraph.
    Returns True if violation found."""
    # Check all subsets of size 6, 7, ..., n
    # Start with size 6 (most common violation), then larger
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            if _is_regular_subset(adj, subset):
                return True
    return False

def _has_regular_6plus_fast(adj, n):
    """Faster check: only check size-6 subsets first (most likely violations),
    then size 7, etc. Uses early termination."""
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            # Quick check: compute degrees
            degs = []
            for v in subset:
                d = 0
                row = adj[v]
                for u in subset:
                    if u != v and row[u]:
                        d += 1
                degs.append(d)
            if len(set(degs)) == 1:
                return True
    return False

def _validate(adj, n):
    """Full validation: no subset of size >= 6 induces a regular subgraph."""
    return not _has_regular_6plus_fast(adj, n)

def _spectral_cost(adj, n):
    """Compute a spectral-based irregularity cost.
    For each 6-vertex subset, compute degree variance; sum over all subsets.
    Approximate using eigenvalue spread of adjacency matrix."""
    A = np.array(adj, dtype=float)
    # Eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(A)
    except:
        return float('inf')
    # Cost: sum of squared deviations from mean eigenvalue
    mean_ev = np.mean(eigvals)
    cost = np.sum((eigvals - mean_ev) ** 2)
    return cost

def _flip_edge(adj, n, u, v):
    """Flip edge (u,v) in adjacency matrix."""
    adj[u][v] = 1 - adj[u][v]
    adj[v][u] = adj[u][v]

def _local_search(adj, n, max_iter=200):
    """Local search: flip edges to reduce spectral cost, then validate."""
    best_adj = [row[:] for row in adj]
    best_cost = _spectral_cost(best_adj, n)
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        improved = False
        # Try flipping each edge
        for u in range(n):
            for v in range(u + 1, n):
                _flip_edge(best_adj, n, u, v)
                new_cost = _spectral_cost(best_adj, n)
                if new_cost < best_cost:
                    best_cost = new_cost
                    improved = True
                else:
                    _flip_edge(best_adj, n, u, v)  # revert
                if _time_left() < 5.0:
                    return best_adj, best_cost
        if not improved:
            break
    return best_adj, best_cost

def _random_graph(n):
    """Generate a random graph on n vertices."""
    adj = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                adj[u][v] = 1
                adj[v][u] = 1
    return adj

def _try_n(n, num_trials=50):
    """Try to find a valid graph on n vertices."""
    best_adj = None
    for trial in range(num_trials):
        if _time_left() < 10.0:
            break
        adj = _random_graph(n)
        # Local search to reduce cost
        adj, cost = _local_search(adj, n, max_iter=100)
        # Validate
        if _validate(adj, n):
            return adj
        if best_adj is None or cost < _spectral_cost(best_adj, n):
            best_adj = adj
    return None

def construct():
    """Find the largest n such that a valid graph exists.
    Returns {'n': n, 'edges': [[u,v], ...]}"""
    random.seed(42)
    
    # Try from n=21 down to n=5
    # n=20 is known to work; try 21 first
    for n in range(21, 4, -1):
        if _time_left() < 15.0:
            break
        adj = _try_n(n, num_trials=30 if n <= 18 else 10)
        if adj is not None:
            # Build edge list
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    # Fallback: return empty graph on 5 vertices
    return {'n': 5, 'edges': []}
