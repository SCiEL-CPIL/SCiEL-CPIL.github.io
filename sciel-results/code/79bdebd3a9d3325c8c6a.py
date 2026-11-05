<<<<<<< SEARCH
def _girth_np(adj, max_len):
    """Girth via non-backtracking walks (BFS over directed edges, no immediate backtrack).
    Vectorized with NumPy.  Returns the shortest cycle length (>= 3) or None."""
    n = len(adj)
    if n == 0:
        return None
    L = [sorted(a) for a in adj]
    # directed-edge id: (u, v) -> u*n + v
    from collections import deque
    prev = {}          # (u, v) -> (w, u)
    depth = {}         # (u, v) -> depth
    q = deque()
    for u in range(n):
        for v in L[u]:
            q.append((u, v))
            prev[(u, v)] = None
            depth[(u, v)] = 1
    # level-by-level vectorized expansion
    level = [(u, v) for (u, v) in q]
    best = None
    while level and (best is None or len(level) < 1 << 20):
        if _deadline():
            return best
        nxt = []
        for (u, v) in level:
            for w in L[v]:
                if w == u:            # no immediate backtrack
                    continue
                if (v, w) in depth:
                    d = depth[(v, w)]
                    if d >= 2:        # u -> v -> w closes a cycle of length d + 1
                        cand = d + 1
                        if best is None or cand < best:
                            best = cand
                        if best == 3:
                            return best
                    continue
                prev[(v, w)] = (u, v)
                depth[(v, w)] = depth[(u, v)] + 1
                nxt.append((v, w))
                if best is not None and depth[(v, w)] + 1 >= best:
                    pass
        level = nxt
    return best
=======
def _girth_np(adj, max_len):
    """Girth via non-backtracking walks (BFS over directed edges, no immediate backtrack).
    Returns the shortest cycle length (>= 3) or None."""
    n = len(adj)
    if n == 0:
        return None
    L = [sorted(a) for a in adj]
    # Use integer encoding: (u, v) -> u * n + v
    # BFS from all directed edges simultaneously
    # depth[eid] = depth of directed edge eid
    # We track visited directed edges
    # Key insight: for girth detection, we need BFS where we don't backtrack
    # Start: all (u, v) for edges (u,v) with depth 1
    # From (u, v), go to (v, w) for all w in L[v], w != u
    # If (v, w) was visited at depth d >= 2, cycle of length d+1
    
    # For efficiency, use arrays
    # Total directed edges = 2 * |E|
    # Map (u, v) to index
    # Since graph is k-regular, each vertex has k neighbors
    # directed edge id: u * n + v
    
    # Use a dict for depth since n can be large but sparse
    # Actually, let's use a list of size n*n only if n is small, else dict
    # For n up to ~1000, n*n = 1M which is fine
    
    if n <= 2000:
        # Use array-based approach
        depth_arr = np.full(n * n, -1, dtype=np.int32)
        # BFS level by level
        # Current level: list of directed edge ids (u*n + v)
        curr = []
        for u in range(n):
            for v in L[u]:
                eid = u * n + v
                depth_arr[eid] = 1
                curr.append(eid)
        best = None
        level_num = 1
        while curr:
            if _deadline():
                return best
            if best is not None and level_num + 1 >= best:
                break
            nxt = []
            for eid in curr:
                u = eid // n
                v = eid % n
                for w in L[v]:
                    if w == u:
                        continue
                    neid = v * n + w
                    if depth_arr[neid] >= 0:
                        d = depth_arr[neid]
                        if d >= 2:
                            cand = d + 1
                            if best is None or cand < best:
                                best = cand
                            if best == 3:
                                return best
                        continue
                    depth_arr[neid] = level_num + 1
                    nxt.append(neid)
            curr = nxt
            level_num += 1
        return best
    else:
        # Fallback to dict-based
        from collections import deque
        depth = {}
        curr = []
        for u in range(n):
            for v in L[u]:
                eid = u * n + v
                depth[eid] = 1
                curr.append(eid)
        best = None
        level_num = 1
        while curr:
            if _deadline():
                return best
            if best is not None and level_num + 1 >= best:
                break
            nxt = []
            for eid in curr:
                u = eid // n
                v = eid % n
                for w in L[v]:
                    if w == u:
                        continue
                    neid = v * n + w
                    if neid in depth:
                        d = depth[neid]
                        if d >= 2:
                            cand = d + 1
                            if best is None or cand < best:
                                best = cand
                            if best == 3:
                                return best
                        continue
                    depth[neid] = level_num + 1
                    nxt.append(neid)
            curr = nxt
            level_num += 1
        return best
>>>>>>> REPLACE

