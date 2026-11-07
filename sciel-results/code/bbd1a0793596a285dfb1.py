"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    base = set()
    for e in P['edges']:
        base.add((e[0], e[1]))
        base.add((e[1], e[0]))
    
    added = [(0, 12), (6, 18), (8, 16), (10, 14)]
    
    # Build full graph
    adj = {i: set() for i in range(P['n'])}
    for u, v in base:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    
    # Verify triangle-free
    for i in range(P['n']):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj[i]:
                    return []  # triangle found, should not happen
    
    # Verify diameter <= 4 via BFS
    for start in range(P['n']):
        dist = {start: 0}
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if len(dist) != P['n']:
            return []  # not connected
        if max(dist.values()) > 4:
            return []  # diameter too large
    
    # Verify no duplicate or existing edges
    for u, v in added:
        if (u, v) in base or (v, u) in base:
            return []
    
    return [[u, v] for u, v in added]
# EVOLVE-BLOCK-END
