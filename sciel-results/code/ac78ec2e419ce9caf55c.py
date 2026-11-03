def _gen_random_3_reg(n):
    stubs = list(range(3 * n))
    random.shuffle(stubs)
    edges = []
    for i in range(0, 3 * n, 2):
        u = stubs[i] // 3
        v = stubs[i + 1] // 3
        if u == v:
            return None
        if (u, v) in edges or (v, u) in edges:
            return None
        edges.append((u, v))
    return edges

def _check_girth(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g // 2 - 1: # Max depth 7 for g=16
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a visited node
                    if v != parent[u] and u != parent[v]:
                        # Cycle length
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True
