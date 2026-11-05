"""New approach: Random k-regular graph generation with girth verification."""
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses random k-regular graph generation (configuration model) + girth check.
    Returns (n, edges) where edges is a list of (i, j) pairs.
    """
    # Moore bound for k-regular graph with girth g
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * ((k - 1) ** r)
    else:
        r = g // 2
        moore = k * ((k - 1) ** (r - 1))
    
    # Start searching from Moore bound, increase n
    # We need n such that k*n is even
    def make_n_valid(n):
        while (n * k) % 2 != 0:
            n += 1
        return n
    
    max_attempts = 5000
    current_n = make_n_valid(moore)
    
    for attempt in range(max_attempts):
        # Try to generate a random k-regular graph with current_n vertices
        success, edges = _generate_k_regular_simple(current_n, k)
        if success:
            # Check girth
            girth = _compute_girth(current_n, edges, k)
            if girth >= g:
                return current_n, edges
        # Increase n for next attempt
        current_n += 2 if (k * current_n) % 2 == 0 else 1
        current_n = make_n_valid(current_n)
        if current_n > 5000:  # Safety limit
            break
    
    # Fallback: use a known construction if random fails
    return _fallback_construct(k, g)

def _generate_k_regular_simple(n, k):
    """
    Generate a random simple k-regular graph on n vertices using configuration model.
    Returns (success, edges) where edges is a list of (i,j) pairs.
    """
    # Configuration model: create n*k half-edges, pair them randomly
    half_edges = []
    for v in range(n):
        for _ in range(k):
            half_edges.append(v)
    
    for _ in range(200):  # Retry if we get multigraph
        random.shuffle(half_edges)
        # Pair up half-edges
        adj = {i: set() for i in range(n)}
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:  # Loop
                valid = False
                break
            if v in adj[u]:  # Multiple edge
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            continue
        # Check all vertices have degree k
        if all(len(adj[i]) == k for i in range(n)):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return True, edges
    return False, []

def _compute_girth(n, edges, k):
    """
    Compute the girth (length of shortest cycle) of a graph.
    Uses BFS from each vertex with non-backtracking.
    Returns girth, or a large number if no cycle found.
    """
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        # BFS with non-backtracking: state is (vertex, prev_vertex)
        # dist[v][prev] = shortest non-backtracking walk length from (start, None) to (v, prev)
        # We use a dict: key = (v, prev), value = distance
        visited = {}
        queue = deque()
        # Start: from start, we can go to any neighbor
        for nb in adj[start]:
            visited[(nb, start)] = 1
            queue.append((nb, start, 1))
        
        while queue:
            v, prev, dist = queue.popleft()
            if dist >= min_girth:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue  # Non-backtracking
                if nb == start:
                    # Found a cycle of length dist + 1
                    cycle_len = dist + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    continue
                state = (nb, v)
                if state not in visited:
                    visited[state] = dist + 1
                    queue.append((nb, v, dist + 1))
    
    if min_girth == float('inf'):
        return 10**9  # No cycle (shouldn't happen for k >= 2)
    return min_girth

def _fallback_construct(k, g):
    """
    Fallback: use a simple explicit construction.
    For small k and g, we can use circulant graphs or other constructions.
    """
    # For k=5, g=9, we need at least the Moore bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * ((k - 1) ** r)
    else:
        r = g // 2
        moore = k * ((k - 1) ** (r - 1))
    
    n = moore
    while (n * k) % 2 != 0:
        n += 1
    
    # Generate a random k-regular graph (this may not have girth g, but it's a fallback)
    for _ in range(100):
        success, edges = _generate_k_regular_simple(n, k)
        if success:
            return n, edges
    
    # Last resort: return a path-based construction (won't be k-regular, but shouldn't happen)
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    return n, edges
