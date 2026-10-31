<<<<<<< SEARCH
def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])

    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
=======
def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    # Check for cycles shorter than g using non-backtracking walks
    for src in range(n):
        if _nb_detect_short_cycle(src, adj, n, g):
            return False

    return True


def _nb_detect_short_cycle(src, adj, n, g):
    """
    Non-backtracking walk from src to detect cycles shorter than g.
    """
    # State: (current, previous)
    visited = set()
    queue = deque()
    
    for v in adj[src]:
        state = (v, src)
        if state not in visited:
            visited.add(state)
            queue.append((state, 1))
    
    while queue:
        (u, p), length = queue.popleft()
        
        if u == src and length + 1 >= 3:
            cycle_len = length + 1
            if cycle_len < g:
                return True
        
        if length + 1 >= g:
            continue
        
        for w in adj[u]:
            if w == p:
                continue
            new_state = (w, u)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, length + 1))
    
    return False

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
