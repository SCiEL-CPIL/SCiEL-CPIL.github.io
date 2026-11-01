import sys
import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 5 and g == 9:
        # Try q=3, d=5 LUW (3-regular, girth >= 10, 1458 vertices)
        # Then add 2 perfect matchings between points and lines at dist >= 8
        try:
            n, edges = construct_luw_q3_d5_plus_2matchings()
            if n is not None and verify_girth(n, edges, k, g):
                return n, edges
        except Exception:
            pass

        # Fallback: q=5, d=4 LUW (5-regular, girth >= 9, 6250 vertices)
        try:
            n, edges = construct_luw_q5_d4()
            if n is not None and verify_girth(n, edges, k, g):
                return n, edges
        except Exception:
            pass

    return construct_default(k, g)

def construct_luw_q5_d4():
    """LUW construction: q=5, d=4, giving 5-regular bipartite graph with girth >= 9.
    Total vertices: 2 * 5^5 = 6250"""
    q = 5
    d = 4
    
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                u_idx = index[vertex]
                v_idx = index[other]
                edges.append((u_idx, v_idx))
                
    return len(order), edges

def construct_luw_q3_d5_plus_2matchings():
    """LUW q=3, d=5: 3-regular bipartite graph, girth >= 10, 1458 vertices.
    Add 2 perfect matchings at distance >= 8 to get 5-regular, girth >= 9."""
    q = 3
    d = 5
    
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # For c=5: r=(5-5)%4=0, c0=5-0=5
    # rule[5] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][0] = (1, 3)
    rule[5] = (1, 3)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    adj = [[] for _ in range(2000)]
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                u_idx = index[vertex]
                v_idx = index[other]
                edges.append((u_idx, v_idx))
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
                
    n = len(order)
    points = [i for i in range(n) if order[i][0] == 0]
    lines = [i for i in range(n) if order[i][0] == 1]
    n_p = len(points)
    n_l = len(lines)
    
    if n_p != n_l:
        return None, None
    
    # Compute all-pairs distances using BFS from each vertex
    dist = np.full((n, n), -1, dtype=np.int32)
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    
    # Build bipartite adjacency for matching: edges where dist >= 8
    match_adj = [[] for _ in range(n_p)]
    p_idx = {p: i for i, p in enumerate(points)}
    l_idx = {l: i for i, l in enumerate(lines)}
    
    for p in points:
        for l in lines:
            if dist[p][l] >= 8:
                match_adj[p_idx[p]].append(l_idx[l])
                
    for i in range(n_p):
        if not match_adj[i]:
            return None, None

    def find_perfect_matching(adj_list, n_left, n_right, removed_edges=None):
        if removed_edges is None:
            removed_edges = set()
        
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist_hk = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist_hk[u] = 0
                    queue.append(u)
                else:
                    dist_hk[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if (u, v) in removed_edges:
                        continue
                    if pair_v[v] == -1:
                        found = True
                    elif dist_hk[pair_v[v]] == float('inf'):
                        dist_hk[pair_v[v]] = dist_hk[u] + 1
                        queue.append(pair_v[v])
            return found
            
        def dfs(u):
            for v in adj_list[u]:
                if (u, v) in removed_edges:
                    continue
                if pair_v[v] == -1 or (dist_hk[pair_v[v]] == dist_hk[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist_hk[u] = float('inf')
            return False
            
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pair_u, pair_v
    
    m1, pair_u1, pair_v1 = find_perfect_matching(match_adj, n_p, n_l)
    if m1 < n_p:
        return None, None
        
    new_edges = list(edges)
    removed = set()
    for u in range(n_p):
        v = pair_u1[u]
        p = points[u]
        l = lines[v]
        new_edges.append((p, l))
        removed.add((u, v))
        
    m2, pair_u2, pair_v2 = find_perfect_matching(match_adj, n_p, n_l, removed)
    if m2 < n_p:
        return None, None
        
    for u in range(n_p):
        v = pair_u2[u]
        p = points[u]
        l = lines[v]
        new_edges.append((p, l))
        
    return n, new_edges

def construct_default(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def verify_girth(n, edges, k, g):
    """Verify degree and girth using non-backtracking walk analysis."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth: for each edge, the shortest non-backtracking path between its endpoints
    # must be at least g-1 (so cycle length >= g)
    for u, v in edges:
        dist = [-1] * n
        dist[u] = 0
        queue = deque([u])
        found = False
        while queue:
            cur = queue.popleft()
            if cur == v:
                if dist[v] + 1 < g:
                    return False
                found = True
                break
            for nb in adj[cur]:
                # Non-backtracking: don't go back to where we came from
                if (cur == u and nb == v) or (cur == v and nb == u):
                    continue
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        if not found:
            return False
    
    return True
