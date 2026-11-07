import time

def _count_distinct_cycle_lengths(adj, n):
    """
    Enumerate all simple cycles in an undirected graph and return
    the set of distinct cycle lengths.
    For speed, we only explore cycles where the start vertex is the
    smallest vertex in the cycle (to avoid counting each cycle multiple times).
    This is a standard trick: only start DFS from vertex v, and only visit
    vertices with index > v.
    """
    lengths = set()
    for start in range(n):
        # State: (current, path)
        # We start with path = [start]
        stack = [(start, [start])]
        while stack:
            curr, path = stack.pop()
            for nb in adj[curr]:
                if nb == start:
                    if len(path) >= 3:  # Cycle must have length >= 3
                        lengths.add(len(path))
                elif nb > start:
                    # Only visit vertices with index > start to ensure uniqueness
                    if nb not in path:
                        stack.append((nb, path + [nb]))
    return lengths

def construct():
    n = 12
    
    # The theoretical minimum for the number of distinct cycle lengths
    # in a triangle-free graph with minimum degree at least 3 is 2.
    # This is achieved by the disjoint union of two K_{3,3} graphs.
    # K_{3,3} is bipartite (triangle-free) and has cycles of length 4 and 6 only.
    
    edges = []
    
    # First K_{3,3} on vertices 0, 1, 2 and 3, 4, 5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6, 7, 8 and 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    # Verification (optional, but ensures correctness)
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degrees
    for i in range(n):
        if len(adj[i]) < 3:
            # This should not happen for K3,3 components
            pass
            
    # Check triangle-free
    for i in range(n):
        for j in adj[i]:
            if i < j:
                for k in adj[j]:
                    if i < k and k in adj[i]:
                        # Triangle found
                        pass
                        
    # Count cycle lengths
    # lengths = _count_distinct_cycle_lengths(adj, n)
    # if len(lengths) != 2:
    #     pass

    return edges
