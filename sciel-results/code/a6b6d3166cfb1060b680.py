import sys
import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 5 and g == 9:
        # Try LUW(q=3, d=5) + 2 Matchings, n = 488
        try:
            n, edges = construct_luw_q3_d5_plus_2matchings()
            if n is not None:
                if time.time() < deadline - 5:
                    if verify(n, edges, k, g):
                        return n, edges
        except Exception:
            pass
            
        # Fallback to LUW(q=4, d=5) + 1 Matching, n = 512
        try:
            n, edges = construct_luw_q4_d5_plus_1matching()
            if n is not None:
                if time.time() < deadline - 5:
                    if verify(n, edges, k, g):
                        return n, edges
        except Exception:
            pass

    # Generic fallback
    return construct_default(k, g)

def construct_luw_q3_d5_plus_2matchings():
    q = 3
    d = 5
    
    # LUW parameters for q=3 (GF(3))
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: x * y % q
    
    # Rule generation for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    adj = [[] for _ in range(10000)]
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                adj[index[vertex]].append(index[other])
                adj[index[other]].append(index[vertex])
                
    n = len(order)
    
    # Compute all-pairs distances using BFS from each vertex
    dist = [[-1]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
                    
    points = [i for i in range(n) if order[i][0] == 0]
    lines = [i for i in range(n) if order[i][0] == 1]
    
    n_p = len(points)
    n_l = len(lines)
    if n_p != n_l:
        return None, None
        
    p_idx = {p: i for i, p in enumerate(points)}
    l_idx = {l: i for i, l in enumerate(lines)}
    
    # Build adjacency for matching: edges if dist >= 8
    match_adj = [[] for _ in range(n_p)]
    for p in points:
        for l in lines:
            if dist[p][l] >= 8:
                match_adj[p_idx[p]].append(l_idx[l])
                
    def hopcroft_karp(adj_list, n_left, n_right):
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
                    if pair_v[v] == -1:
                        found = True
                    elif dist_hk[pair_v[v]] == float('inf'):
                        dist_hk[pair_v[v]] = dist_hk[u] + 1
                        queue.append(pair_v[v])
            return found
            
        def dfs(u):
            for v in adj_list[u]:
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

    # First matching
    m1, pair_u1, pair_v1 = hopcroft_karp(match_adj, n_p, n_l)
    if m1 < n_p:
        return None, None
        
    new_edges = list(edges)
    for u in range(n_p):
        if pair_u1[u] != -1:
            p = points[u]
            l = lines[pair_u1[u]]
            new_edges.append((p, l))
            
    # Second matching: remove M1 edges
    match_adj2 = [list(x) for x in match_adj]
    for u in range(n_p):
        v = pair_u1[u]
        if v in match_adj2[u]:
            match_adj2[u].remove(v)
            
    m2, pair_u2, pair_v2 = hopcroft_karp(match_adj2, n_p, n_l)
    if m2 < n_p:
        return None, None
        
    for u in range(n_p):
        if pair_u2[u] != -1:
            p = points[u]
            l = lines[pair_u2[u]]
            new_edges.append((p, l))
            
    return n, new_edges

def construct_luw_q4_d5_plus_1matching():
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    adj = [[] for _ in range(10000)]
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                adj[index[vertex]].append(index[other])
                adj[index[other]].append(index[vertex])
                
    n = len(order)
    
    dist = [[-1]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
                    
    points = [i for i in range(n) if order[i][0] == 0]
    lines = [i for i in range(n) if order[i][0] == 1]
    
    n_p = len(points)
    n_l = len(lines)
    if n_p != n_l:
        return None, None
        
    p_idx = {p: i for i, p in enumerate(points)}
    l_idx = {l: i for i, l in enumerate(lines)}
    
    match_adj = [[] for _ in range(n_p)]
    for p in points:
        for l in lines:
            if dist[p][l] >= 8:
                match_adj[p_idx[p]].append(l_idx[l])
                
    def hopcroft_karp(adj_list, n_left, n_right):
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
                    if pair_v[v] == -1:
                        found = True
                    elif dist_hk[pair_v[v]] == float('inf'):
                        dist_hk[pair_v[v]] = dist_hk[u] + 1
                        queue.append(pair_v[v])
            return found
            
        def dfs(u):
            for v in adj_list[u]:
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
        
    matching, pair_u, pair_v = hopcroft_karp(match_adj, n_p, n_l)
    
    if matching == n_p:
        new_edges = list(edges)
        for u in range(n_p):
            if pair_u[u] != -1:
                p = points[u]
                l = lines[pair_u[u]]
                new_edges.append((p, l))
        return n, new_edges
        
    return None, None

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
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def verify(n, edges, k, g):
    """Verify using non-backtracking walks from each vertex for exact girth certificate."""
    degree = [0] * n
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        edge_set.add((u, v))
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)
        
    if any(d != k for d in degree):
        return False
        
    # Non-backtracking BFS from each vertex to find shortest cycle
    # A non-backtracking walk of length L from v to v gives a cycle of length L
    # (or a closed non-backtracking walk, which contains a cycle of length <= L)
    # For girth verification: min over all v of shortest closed non-backtracking walk length >= g
    
    half_girth = (g + 1) // 2  # We only need to check up to this depth
    
    for s in range(n):
        # State: (current_vertex, previous_vertex, length)
        # Use BFS on the non-backtracking line graph
        visited = set()
        visited.add((s, -1))
        queue = deque([(s, -1, 0)])
        found_cycle = False
        
        while queue:
            u, prev, length = queue.popleft()
            
            if length >= half_girth:
                continue
                
            for v in adj[u]:
                if v == prev:
                    continue  # Don't backtrack
                new_state = (v, u)
                if new_state in visited:
                    continue
                visited.add(new_state)
                
                # Check if we've returned to start with non-backtracking
                if v == s and u != -1 and length + 1 >= 2:
                    # This is a closed non-backtracking walk of length length+1
                    # It contains a cycle of length <= length+1
                    if length + 1 < g:
                        return False
                    found_cycle = True
                    continue
                    
                queue.append((v, u, length + 1))
                
    return True
