import time
import random
from itertools import combinations

def _is_regular_subset(adj_mask, subset_mask, n):
    """Check if the induced subgraph on subset_mask is regular."""
    # Count vertices in subset
    k = bin(subset_mask).count('1')
    if k < 6:
        return True  # not checked per problem, but safe
    # For each vertex in subset, compute degree within subset
    degrees = []
    for i in range(n):
        if subset_mask & (1 << i):
            # degree of i within subset = popcount(adj_mask[i] & subset_mask)
            deg = bin(adj_mask[i] & subset_mask).count('1')
            degrees.append(deg)
    # Check all degrees equal
    first = degrees[0]
    for d in degrees[1:]:
        if d != first:
            return False
    return True

def _has_regular_subset(adj_mask, n):
    """Check if any subset of size >= 6 induces a regular graph."""
    # Generate all subsets of size 6 to n
    # For efficiency, iterate by size
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subset(adj_mask, mask, n):
                return True, subset
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_n = 5
    best_edges = []
    
    # Try random graphs for increasing n
    for n in range(22, 5, -1):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            # Generate random graph
            edges = []
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            
            # Check for regular subsets
            found, subset = _has_regular_subset(adj_mask, n)
            if not found:
                return {'n': n, 'edges': edges}
    
    # Fallback: return empty graph on 5 vertices
    return {'n': 5, 'edges': []}
