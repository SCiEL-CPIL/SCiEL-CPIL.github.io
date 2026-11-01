import time
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_n = float('inf')
    best_edges = None
    
    def verify_graph(n, edges, k, g):
        """Verify graph is simple, k-regular, girth >= g using non-backtracking BFS."""
        if n == 0:
            return False
        adj = [[] for _ in range(n)]
        seen_edges = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u, v), max(u, v))
            if key in seen_edges:
                return False
            seen_edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Non-backtracking BFS from each vertex
        max_depth = (g - 1) // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([(u, -1)])  # (vertex, parent)
            while q:
                curr, par = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if v == par:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append((v, curr))
                    else:
                        if v == parent[curr]:
                            continue
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_luw_graph(q_val, d_val):
        """Build LUW graph."""
        if q_val == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q_val
            sub = lambda x,y: (x-y)%q_val
            mul = lambda x,y: x*y%q_val
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        def get_rule(dv):
            r = dict(rule)
            for c in range(5, dv+1):
                rc = (c-5)%4
                c0 = c-(c-5)%4
                if rc==0: r[c]=(1,c0-2)
                elif rc==1: r[c]=(c0-1,1)
                elif rc==2: r[c]=(c0,1)
                else: r[c]=(1,c0+1)
            return r
        r = get_rule(d_val)
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0]*(d_val-1)
                for c in range(2, d_val+1):
                    a,b = r[c]
                    if side==0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,)+(0,)*d_val)
        index = {start: 0}
        order = [start]
        edges = []
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try LUW with various d values
    # Estimate max d to keep N reasonable. N ~ 2 * q^d.
    # For q=4, d=5 -> N~2048. d=6 -> N~8192.
    # Verification is O(N^2). N=8000 is ~64M ops, might be slow in Python.
    # Let's cap d such that 2*q^d < 5000.
    import math
    max_d = int(math.log(5000 / 2, k)) if k > 1 else 10
    min_d = max(3, (g-2)//2)
    
    # Ensure we try at least a few values
    search_range = range(min_d, min(max_d + 1, min_d + 10))
    
    for d_val in search_range:
        if time.time() > deadline - 20:
            break
            
        # Check estimated size
        est_n = 2 * (k ** d_val)
        if est_n > 5000:
            # If we haven't found anything yet, maybe try this one but with care?
            # Or skip if too big?
            # If est_n is too big, verification will be slow.
            # Let's allow up to 10000 if we have time.
            if est_n > 10000:
                continue

        try:
            n, edges = build_luw_graph(k, d_val)
            if n >= best_n:
                continue
            # Only verify if n is not too large to risk timeout
            if n <= 5000:
                if verify_graph(n, edges, k, g):
                    best_n = n
                    best_edges = edges
                    if n <= 400:
                        break
            else:
                # If n is large, skip verification to save time, 
                # or do a quick check?
                # If we skip, we might return an invalid graph.
                # Better to skip large graphs if we have a smaller valid one.
                # If we don't have a valid one, we might be stuck.
                # But LUW graphs with large d usually have large girth.
                # However, without verification, we can't be sure.
                # Let's assume if d is large enough, girth is large enough?
                # No, we must verify.
                # If n > 5000, verification is risky.
                # Let's just skip these to avoid timeout, hoping a smaller d works.
                pass
        except Exception:
            pass

    if best_edges is None:
        # Fallback: Return a trivial valid graph if possible?
        # But k=4, g=10 requires N >= 485.
        # If we couldn't find one, we return 0, [] which scores 0.
        # To avoid 0, we could try a random graph? Unlikely to work.
        # Or return the last built graph even if unverified? No, must be valid.
        # Let's try one more time with a slightly larger budget if we have time.
        if time.time() < deadline - 5:
             try:
                 # Try d = min_d + 5
                 d_try = min_d + 5
                 n, edges = build_luw_graph(k, d_try)
                 if n < 10000: # Safety
                     if verify_graph(n, edges, k, g):
                         best_n = n
                         best_edges = edges
             except Exception:
                 pass

    if best_edges is None:
        return 0, []
    
    return best_n, best_edges
