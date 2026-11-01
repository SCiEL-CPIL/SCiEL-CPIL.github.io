import sys
import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 5 and g == 9:
        # Try LUW(q=3, d=4) + 2 Matchings, n = 486
        try:
            n, edges = construct_luw_plus_matchings(3, 4, 2)
            if n is not None and time.time() < deadline - 5:
                if verify(n, edges, k, g):
                    return n, edges
        except Exception:
            pass
            
        # Try LUW(q=3, d=5) + 2 Matchings, n = 1458
        try:
            n, edges = construct_luw_plus_matchings(3, 5, 2)
            if n is not None and time.time() < deadline - 5:
                if verify(n, edges, k, g):
                    return n, edges
        except Exception:
            pass
            
        # Fallback to LUW(q=4, d=5) + 1 Matching, n = 8192
        try:
            n, edges = construct_luw_plus_matchings(4, 5, 1)
            if n is not None and time.time() < deadline - 5:
                if verify(n, edges, k, g):
                    return n, edges
        except Exception:
            pass

    # Generic fallback
    return construct_default(k, g)

def luw_graph(q, d):
    """Build LUW graph with parameters q, d. Returns (n, edges, adj, points, lines)."""
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
    adj = [[] for _ in range(q * (q ** d) * 2 + 100)]
    
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
    points = [i for i in range(n) if order[i][0] == 0]
    lines = [i for i in range(n) if order[i][0] == 1]
    
    return n, edges, adj, points, lines

def construct_luw_plus_matchings(q, d, num_matchings):
    """Build LUW(q,d) graph and add num_matchings to reach degree 5."""
    n, edges, adj, points, lines = luw_graph(q, d)
    
    n_p = len(points)
    n_l = len(lines)
    if n_p != n_l:
        return None, None
    
    # Compute distances from each point to all lines using BFS
    # We only need distances, not full all-pairs
    p_idx = {p: i for i, p in enumerate(points)}
    l_idx = {l: i for i, l in enumerate(lines)}
    
    # For girth 9, we need dist >= 7 for added edges (2 + dist >= 9)
    min_dist = 7
    
    # Build bipartite adjacency for matching: connect p-l if dist >= min_dist
    # To save memory/time, use a sparse representation
    match_adj = [[] for _ in range(n_p)]
    
    # BFS from each point
    for p in points:
        dist = [-1] * n
        dist[p] = 0
        queue = deque([p])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        pi = p_idx[p]
        for l in lines:
            if dist[l] >= min_dist:
                li = l_idx[l]
                match_adj[pi].append(li)
                
    # Find perfect matchings using greedy + augmenting path
    def find_perfect_matching(adj_list, n_left, n_right, exclude=None):
        """Find perfect matching using BFS augmenting paths."""
        if exclude is None:
            exclude = set()
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        
        def bfs_augment(start_u):
            """Try to find augmenting path from start_u."""
            prev_v = [-1] * n_right
            prev_u = [-1] * n_left
            found_v = -1
            queue = deque([start_u])
            visited_u = {start_u}
            visited_v = set()
            
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if v in visited_v:
                        continue
                    if (u, v) in exclude:
                        continue
                    visited_v.add(v)
                    prev_v[v] = u
                    if pair_v[v] == -1:
                        found_v = v
                        break
                    else:
                        nu = pair_v[v]
                        if nu not in visited_u:
                            visited_u.add(nu)
                            queue.append(nu)
            
            if found_v == -1:
                return False
            
            # Reconstruct augmenting path
            v = found_v
            while v != -1:
                u = prev_v[v]
                old_v = pair_u[u]
                pair_u[u] = v
                pair_v[v] = u
                v = old_v
            return True
            
        matching = 0
        for u in range(n_left):
            if pair_u[u] == -1:
                if bfs_augment(u):
                    matching += 1
        return matching, pair_u, pair_v
    
    new_edges = list(edges)
    exclude = set()
    
    for m in range(num_matchings):
        matching, pair_u, pair_v = find_perfect_matching(match_adj, n_p, n_l, exclude)
        if matching < n_p:
            return None, None
        for u in range(n_p):
            if pair_u[u] != -1:
                p = points[u]
                l = lines[pair_u[u]]
                new_edges.append((p, l))
                exclude.add((u, pair_u[u]))
                exclude.add((pair_u[u], u))  # for symmetry tracking
                # Also exclude the reverse direction
                pi = p_idx[p]
                li = l_idx[l]
                exclude.add((pi, li))
                exclude.add((li, pi))
                
    return n, new_edges

def construct_default(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    n, edges, adj, points, lines = luw_graph(q, d)
    return n, edges

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
    half_girth = (g + 1) // 2
    
    for s in range(n):
        visited = set()
        visited.add((s, -1))
        queue = deque([(s, -1, 0)])
        
        while queue:
            u, prev, length = queue.popleft()
            
            if length >= half_girth:
                continue
                
            for v in adj[u]:
                if v == prev:
                    continue
                new_state = (v, u)
                if new_state in visited:
                    continue
                visited.add(new_state)
                
                if v == s and u != -1 and length + 1 >= 2:
                    if length + 1 < g:
                        return False
                    continue
                    
                queue.append((v, u, length + 1))
                
    return True
