<<<<<<< SEARCH
def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise. Optimized with early termination."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _verify_girth_array(adj_list, n, g):
    """Faster verification using array-based distance tracking.
    adj_list: list of lists (sorted neighbor lists)
    Returns True if girth >= g."""
    # For each start vertex, do BFS tracking (vertex, prev) -> depth
    # Use a dict keyed by (v, prev) but with early termination
    for start in range(n):
        # dist[v][prev] = depth, but we'll use a flat dict
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True
=======
def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise. Optimized with early termination."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _verify_girth_array(adj_list, n, g):
    """Fast girth verification using array-based distance tracking.
    adj_list: list of lists (sorted neighbor lists)
    Returns True if girth >= g."""
    # Precompute edge index for fast prev lookup
    # For each vertex, store neighbor list and position map
    max_depth = g - 1  # We only need to explore up to depth g-1
    
    for start in range(n):
        # Use a 2D array: dist[v][pos_in_adj] = depth, or -1 if unvisited
        # Since degree is 3, we can use a small array
        # dist[v][i] where i is the position of the "came from" neighbor
        # Actually, let's use a dict but with integer keys for speed
        # Key: v * 4 + prev_index (prev_index = index of prev in adj[v], or 3 for start)
        
        # Better: use a list of dicts, one per vertex
        # dist[v] maps prev_index -> depth
        dist = [{} for _ in range(n)]
        dq = deque()
        dq.append((start, -1, 0))  # (vertex, prev_vertex, depth)
        dist[start][-1] = 0
        
        while dq:
            v, prev, d = dq.popleft()
            if d >= max_depth:
                continue
            for i, w in enumerate(adj_list[v]):
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                if i not in dist[w]:
                    dist[w][i] = d + 1
                    dq.append((w, v, d + 1))
    return True
>>>>>>> REPLACE

