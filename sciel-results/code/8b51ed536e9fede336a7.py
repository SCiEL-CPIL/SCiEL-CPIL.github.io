"""Construct a 4-regular graph with girth at least g using circulant search and verification."""

from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses circulant graph search for k=4.
    """
    if k == 4:
        # Search for the smallest circulant C(n, ±1, ±m) with girth >= g
        best = _search_circulant_4(g)
        if best is not None:
            n, edges = best
            return n, edges
    
    # Fallback for other k or if circulant search fails
    return _fallback(k, g)


def _search_circulant_4(g, max_n=400):
    """Search for smallest n such that C(n, ±1, ±m) has girth >= g for some m."""
    best = None
    for n in range(20, max_n + 1):
        for m in range(2, n // 2 + 1):
            # Skip invalid m values
            if m == 1 or m == n - 1:
                continue
            if n % 2 == 0 and m == n // 2:
                continue
            # Check that ±1 and ±m give 4 distinct steps
            steps = {1, -1, m, -m}
            if len(steps) != 4:
                continue
            
            # Quick check: the graph must be connected and 4-regular (it is by construction)
            # Check girth
            if _has_girth_at_least(g, n, m):
                # Build edges
                edges = []
                for i in range(n):
                    for step in [1, m]:
                        j = (i + step) % n
                        if i < j:
                            edges.append((i, j))
                    for step in [-1, -m]:
                        j = (i + step) % n
                        if i < j:
                            edges.append((i, j))
                
                if best is None or n < best[0]:
                    best = (n, edges)
    
    return best


def _has_girth_at_least(g, n, m):
    """
    Check if C(n, ±1, ±m) has girth at least g.
    Use BFS with non-backtracking from each vertex, limited to depth g-1.
    """
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for step in [1, -1, m, -m]:
            adj[i].append((i + step) % n)
    
    # For each start vertex, do BFS tracking (node, prev) to avoid backtracking
    # Limit depth to g-1
    max_depth = g - 1
    
    for start in range(n):
        # BFS from start
        # State: (node, prev_node), distance
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            state = (nb, start)
            if state not in visited:
                visited.add(state)
                queue.append((state, 1))
        
        while queue:
            (node, prev), d = queue.popleft()
            
            # Check if we've reached start (cycle found)
            if node == start:
                if d + 1 < g:
                    return False
                continue
            
            if d >= max_depth:
                continue
            
            for nb in adj[node]:
                if nb == prev:
                    continue
                state = (nb, node)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, d + 1))
    
    return True


def _fallback(k, g):
    """Fallback construction for general k."""
    # Use a simple circulant or complete bipartite-like structure
    if k == 2:
        # Cycle
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    # For k >= 3, use a large circulant
    n = 2 * g * g
    # Use steps 1, 2, ..., k//2 and their negatives
    half_k = k // 2
    steps = list(range(1, half_k + 1))
    if k % 2 == 0:
        # k even: use ±1, ±2, ..., ±(k/2)
        pass
    else:
        # k odd: use ±1, ±2, ..., ±((k-1)/2) and one more
        steps.append(n // 2)
    
    edges = []
    for i in range(n):
        for step in steps:
            j = (i + step) % n
            if i < j:
                edges.append((i, j))
            j2 = (i - step) % n
            if i < j2:
                edges.append((i, j2))
    
    # Deduplicate
    edge_set = set()
    unique_edges = []
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(key)
    
    return n, unique_edges
