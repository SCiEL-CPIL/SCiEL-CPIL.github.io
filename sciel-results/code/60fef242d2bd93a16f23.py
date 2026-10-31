<<<<<<< SEARCH
def _full_girth_check_circ(n, a, b, g):
    """Full non-backtracking girth check for C_n(1,a,b) from vertex 0."""
    # Since the graph is vertex-transitive, checking from 0 suffices.
    # Non-backtracking BFS: state (u, prev)
    G_mod = [1, n-1, a % n, (n-a) % n, b % n, (n-b) % n]
    
    dist = {}
    queue = deque()
    start = (0, -1)
    dist[start] = 0
    queue.append(start)
    
    max_len = g - 1
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_len:
            continue
        for g_step in G_mod:
            v = (u + g_step) % n
            if v == prev:
                continue
            ns = (v, u)
            nd = d + 1
            if nd > max_len:
                continue
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                if v == 0 and u != -1:
                    return False  # found cycle shorter than g
                queue.append(ns)
    return True
=======
# _full_girth_check_circ removed
>>>>>>> REPLACE

