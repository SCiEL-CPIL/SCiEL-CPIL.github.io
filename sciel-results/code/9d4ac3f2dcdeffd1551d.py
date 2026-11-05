import time
import random
import numpy as np
from collections import deque

def _build_luw(k, g):
    """Construct Lazebnik-Ustimenko-Woldar graph as fallback."""
    q = k
    if q == 4:
        d = 5 if g <= 10 else 6
        tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def _add(a,b): return a ^ b
        def _sub(a,b): return a ^ b
        def _mul(a,b): return tbl[a][b]
    else:
        d = max(3, g - 5 if (g-5) % 2 else g - 4)
        def _add(a,b): return (a+b) % q
        def _sub(a,b): return (a-b) % q
        def _mul(a,b): return (a*b) % q
    
    rules = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        rules[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def _nbrs(v):
        side, x = v
        for f in range(q):
            y = [None, f] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rules[c]
                if side == 0:
                    y[c] = _add(x[c], _mul(y[a], x[b]))
                else:
                    y[c] = _sub(x[c], _mul(x[a], y[b]))
            yield (1-side, tuple(y))
    
    start = (0, (None,) + (0,)*d)
    idx = {start: 0}
    ordered = [start]
    edge_list = []
    for v in ordered:
        for nb in _nbrs(v):
            if nb not in idx:
                idx[nb] = len(ordered)
                ordered.append(nb)
            if v[0] == 0:
                edge_list.append((idx[v], idx[nb]))
    return len(ordered), edge_list

def _gen_4reg(n):
    """Generate a random simple 4-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    return adj

def _has_short_cycle(adj, n, max_len=9):
    """Check if there's a cycle of length <= max_len using BFS from each vertex."""
    for s in range(n):
        dist = [-1]*n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= max_len:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
                elif dist[v] <= dist[u]:
                    # Cycle of length dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 <= max_len:
                        return True
    return False

def _find_cycle_edge(adj, n, max_len=9):
    """Find an edge (u,v) that lies on a short cycle. Returns (u,v) or None."""
    for s in range(n):
        dist = [-1]*n
        parent = [-1]*n
        dist[s] = 0
        parent[s] = -1
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                elif dist[v] <= dist[u]:
                    clen = dist[u] + dist[v] + 1
                    if clen <= max_len and clen >= 3:
                        return (u, v)
    return None

def _try_swap(adj, a, b, c, d):
    """Attempt 2-opt swap: replace edges (a,b),(c,d) with (a,c),(b,d)."""
    if a == c or a == d or b == c or b == d:
        return False
    if c in adj[a] or d in adj[a] or a in adj[c] or a in adj[d]:
        return False
    # Perform swap
    adj[a].remove(b)
    adj[b].remove(a)
    adj[c].remove(d)
    adj[d].remove(c)
    adj[a].append(c)
    adj[c].append(a)
    adj[b].append(d)
    adj[d].append(b)
    return True

def _revert_swap(adj, a, b, c, d):
    """Revert a 2-opt swap."""
    adj[a].remove(c)
    adj[c].remove(a)
    adj[b].remove(d)
    adj[d].remove(b)
    adj[a].append(b)
    adj[b].append(a)
    adj[c].append(d)
    adj[d].append(c)

def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # girth check using non-backtracking walks
    INF = 10 ** 9
    for (u, v) in edges:
        dist = {}
        q = []
        for w in adj[u]:
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v and L >= 1:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True

def construct(k, g):
    """Construct k-regular graph with girth at least g, minimizing vertices."""
    
    # For k=4, g=10: search for a smaller cage
    if k == 4 and g == 10:
        t0 = time.time()
        hard_limit = t0 + 115  # Leave 5s buffer for 120s limit
        
        # Try to find a graph with n < LUW size (2048)
        # Start from known optimal 320
        for n in range(320, 500, 2):
            if time.time() > hard_limit - 5:
                break
            
            # Allocate time for this n
            time_per_n = 10.0
            n_start = time.time()
            
            found = False
            while time.time() < n_start + time_per_n and not found:
                adj = _gen_4reg(n)
                if adj is None:
                    continue
                
                # Repair phase
                for step in range(200):
                    if time.time() > n_start + time_per_n:
                        break
                    
                    # Check if already good
                    if not _has_short_cycle(adj, n, 9):
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        if _verify(n, edges, k, g):
                            return n, edges
                        else:
                            # Should not happen if _has_short_cycle is correct
                            break
                    
                    ce = _find_cycle_edge(adj, n, 9)
                    if ce is None:
                        break
                    
                    u, v = ce
                    nbrs_u = [x for x in adj[u] if x != v]
                    nbrs_v = [y for y in adj[v] if y != u]
                    
                    if not nbrs_u or not nbrs_v:
                        continue
                    
                    # Try swaps
                    for _ in range(20):
                        if time.time() > n_start + time_per_n:
                            break
                        x = random.choice(nbrs_u)
                        y = random.choice(nbrs_v)
                        
                        if _try_swap(adj, u, v, x, y):
                            # Check improvement
                            if not _has_short_cycle(adj, n, 9):
                                edges = []
                                for u2 in range(n):
                                    for v2 in adj[u2]:
                                        if u2 < v2:
                                            edges.append((u2, v2))
                                if _verify(n, edges, k, g):
                                    return n, edges
                                break
                            # Keep swap, continue
                            break
                    else:
                        continue
            
        # Fallback to LUW construction
        return _build_luw(k, g)
    else:
        return _build_luw(k, g)
