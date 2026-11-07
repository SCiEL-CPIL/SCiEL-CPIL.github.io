import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Two disjoint K_{3,3} components:
    #   Component 1: vertices 0-5, bipartition A={0,1,2}, B={3,4,5}
    #   Component 2: vertices 6-11, bipartition A={6,7,8}, B={9,10,11}
    # Properties:
    #   - Every vertex has degree exactly 3 (satisfies >= 3)
    #   - Bipartite => triangle-free (girth >= 4)
    #   - Contains 4-cycles and 6-cycles => distinct cycle lengths = {4, 6}, count = 2
    #   - No other even lengths possible (max cycle in K_{3,3} is 6)
    #   - 1 distinct cycle length is infeasible for n=12, min-degree 3, triangle-free

    edges = []

    # First K_{3,3}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])

    # Second K_{3,3}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])

    # Verification
    n = 12
    deg = [0] * n
    adj = [set() for _ in range(n)]
    for e in edges:
        u, v = e
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)

    # Check min degree
    if min(deg) < 3:
        return []

    # Check triangle-free: for each edge (u,v), no common neighbor
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u] & adj[v]:
                    return []

    # Ensure edges are Python int lists (not numpy/bool)
    edges = [[int(u), int(v)] for u, v in edges]

    if time.time() > deadline:
        return edges

    return edges
