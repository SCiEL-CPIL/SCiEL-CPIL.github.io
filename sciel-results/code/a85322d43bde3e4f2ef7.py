"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    rng = random.Random(42)

    # Start with a known valid small graph as fallback
    best = _make_fallback(rng)
    best_n = best['n']

    # Try target sizes from largest to smallest
    for target_n in [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 15:
            break
        if target_n <= best_n:
            continue
        result = _local_search(target_n, deadline - 10, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']
            if best_n >= 21:
                break

    return best


def _make_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    for _ in range(5000):
        adj = [0] * (n * n)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i * n + j] = adj[j * n + i] = 1
                    edges.append([i, j])
        if _is_valid_fast(adj, n, rng):
            return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}


def _is_valid_fast(adj, n, rng):
    """Check if no subset of size >= 6 induces a regular graph."""
    if n < 6:
        return True
    
    # Check all 6-subsets (most common violations)
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset, n):
            return False
    
    # Check 7-subsets
    if n >= 7:
        for subset in itertools.combinations(range(n), 7):
            if _is_regular_subset(adj, subset, n):
                return False
    
    # For n >= 8, check a sample of larger subsets
    if n >= 8:
        for size in range(8, n + 1):
            for _ in range(200):
                subset = tuple(sorted(rng.sample(range(n), size)))
                if _is_regular_subset(adj, subset, n):
                    return False
    
    return True


def _is_regular_subset(adj, subset, n):
    """Check if induced subgraph on subset is regular."""
    s = len(subset)
    if s < 2:
        return True
    
    degrees = [0] * s
    for i in range(s):
        ui = subset[i]
        for j in range(i + 1, s):
            uj = subset[j]
            if adj[ui * n + uj]:
                degrees[i] += 1
                degrees[j] += 1
    
    d0 = degrees[0]
    for i in range(1, s):
        if degrees[i] != d0:
            return False
    return True


def _find_violations_fast(adj, n, rng, limit=30):
    """Find up to 'limit' subsets of size >= 6 that induce regular graphs."""
    violations = []
    if n < 6:
        return violations
    
    # Check 6-subsets
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset, n):
            violations.append(subset)
            if len(violations) >= limit:
                return violations
    
    # Check 7-subsets
    if n >= 7 and len(violations) < limit:
        for subset in itertools.combinations(range(n), 7):
            if _is_regular_subset(adj, subset, n):
                violations.append(subset)
                if len(violations) >= limit:
                    return violations
    
    return violations


def _local_search(target_n, deadline, rng):
    """Local search to find a valid graph on target_n vertices."""
    n = target_n
    start_time = time.time()
    
    # Initialize with random graph
    adj = [0] * (n * n)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i * n + j] = adj[j * n + i] = 1
                edges.append((i, j))
    
    # Check if already valid
    if _is_valid_fast(adj, n, rng):
        return _to_result(adj, n, edges)
    
    # Local search: iteratively fix violations
    max_iters = 30000
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find violations
        violations = _find_violations_fast(adj, n, rng, limit=20)
        if not violations:
            return _to_result(adj, n, edges)
        
        # Pick a random violation and flip a random edge within it
        subset = rng.choice(violations)
        s = len(subset)
        i, j = rng.sample(range(s), 2)
        u, v = subset[i], subset[j]
        
        # Flip the edge
        idx = u * n + v
        old_val = adj[idx]
        adj[idx] = adj[v * n + u] = 1 - old_val
        
        # Update edge list
        if old_val:
            edges = [(a, b) for a, b in edges if not ((a == u and b == v) or (a == v and b == u))]
        else:
            edges.append((u, v))
        
        # Occasionally flip a random edge for diversity
        if rng.random() < 0.15:
            a, b = rng.sample(range(n), 2)
            idx2 = a * n + b
            old_val2 = adj[idx2]
            adj[idx2] = adj[b * n + a] = 1 - old_val2
            if old_val2:
                edges = [(x, y) for x, y in edges if not ((x == a and y == b) or (x == b and y == a))]
            else:
                edges.append((a, b))
    
    # Final check
    if _is_valid_fast(adj, n, rng):
        return _to_result(adj, n, edges)
    return None


def _to_result(adj, n, edges):
    """Convert adjacency and edge list to result format."""
    edge_list = [[u, v] for u, v in edges]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
