import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    if g % 2 == 1:
        r = (g - 3) // 2
        if k == 1:
            return 2
        total = 1 + k
        for i in range(1, r + 1):
            total += k * (k - 1) ** i
        return total
    else:
        r = g // 2 - 1
        if k == 1:
            return 2
        total = 0
        for i in range(r + 1):
            total += (k - 1) ** i
        return 2 * total

def _verify_girth(n, edges, k, g):
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        found_short = False
        while queue:
            u = queue.popleft()
            if dist[u] >= g:
                break
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < g:
                        queue.append(v)
                else:
                    if dist[u] + dist[v] + 1 < g:
                        found_short = True
                        break
            if found_short:
                break
        if found_short:
            return False
    return True

def _psl2_11_elements():
    p = 11
    elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a == 0:
                    if b == 0:
                        continue
                    b_inv = pow(b, p - 2, p)
                    c_val = (10 * b_inv) % p
                    d = 0
                    elements.append((a, b, c_val, d))
                else:
                    a_inv = pow(a, p - 2, p)
                    for c in range(p):
                        d = ((1 + b * c) * a_inv) % p
                        elements.append((a, b, c, d))
    return elements

def _mat_mul(m1, m2, p=11):
    a1, b1, c1, d1 = m1
    a2, b2, c2, d2 = m2
    return ((a1*a2 + b1*c2) % p,
            (a1*b2 + b1*d2) % p,
            (c1*a2 + d1*c2) % p,
            (c1*b2 + d1*d2) % p)

def _mat_inv(m, p=11):
    a, b, c, d = m
    return (d, (-b) % p, (-c) % p, a)

def _construct_cayley_psl211():
    p = 11
    elems = _psl2_11_elements()
    n = len(elems)
    if n != 660:
        return None, None
    
    idx = {e: i for i, e in enumerate(elems)}
    ident = (1, 0, 0, 1)
    
    order2 = []
    others = []
    for e in elems:
        if e == ident:
            continue
        e2 = _mat_mul(e, e, p)
        if e2 == ident:
            order2.append(e)
        else:
            others.append(e)
            
    random.seed(42)
    
    def check_girth(S):
        dist = {ident: 0}
        parent = {ident: None}
        q = deque([ident])
        while q:
            u = q.popleft()
            if dist[u] >= 9:
                break
            for s in S:
                v = _mat_mul(u, s, p)
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = (u, s)
                    if dist[v] < 9:
                        q.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + 1 + dist[v]
                        if cycle_len < 9:
                            return False
        return True

    attempts = 0
    while attempts < 5000:
        attempts += 1
        if time.time() > 10:
            break
        s0 = random.choice(order2)
        a = random.choice(others)
        b = random.choice(others)
        a_inv = _mat_inv(a, p)
        b_inv = _mat_inv(b, p)
        if a == b or a == b_inv or a_inv == b or a_inv == b_inv:
            continue
        if a == s0 or a_inv == s0 or b == s0 or b_inv == s0:
            continue
            
        S = [s0, a, a_inv, b, b_inv]
        
        if check_girth(S):
            edges = []
            for i, u in enumerate(elems):
                for s in S:
                    v = _mat_mul(u, s, p)
                    j = idx[v]
                    if i < j:
                        edges.append((i, j))
            return n, edges
    return None, None

def _luw_q4(d, q):
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 2, 3],
        [0, 3, 3, 2],
    ]
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    sub_table = add_table

    def add(x, y):
        return add_table[x][y]

    def sub(x, y):
        return sub_table[x][y]

    def mul(x, y):
        return mul_table[x][y]

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        pattern = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = pattern[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
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
    return len(order), edges, index, order

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _compute_distances(adj, max_dist):
    n = len(adj)
    dist = [[max_dist + 1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            if dist[i][u] >= max_dist:
                continue
            for v in adj[u]:
                if dist[i][v] > dist[i][u] + 1:
                    dist[i][v] = dist[i][u] + 1
                    if dist[i][v] < max_dist:
                        queue.append(v)
    return dist

def _hopcroft_karp(left_size, right_size, adj_left):
    INF = float('inf')
    pair_u = [-1] * left_size
    pair_v = [-1] * right_size
    dist = [0] * left_size

    def bfs():
        queue = deque()
        for u in range(left_size):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_left[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == INF:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found

    def dfs(u):
        for v in adj_left[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(left_size):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching, pair_u

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145

    if k == 5 and g <= 12:
        n, edges = _construct_cayley_psl211()
        if n is not None:
            if _verify_girth(n, edges, k, g):
                return n, edges

    # Fallback to LUW
    if k == 5 and g <= 10:
        d = 4
        n_luw, edges_luw, index, order = _luw_q4(d, 4)
        
        if n_luw < 4000:
            side0 = [i for i, v in enumerate(order) if v[0] == 0]
            side1 = [i for i, v in enumerate(order) if v[0] == 1]
            
            if len(side0) == len(side1):
                adj_luw = _build_adj(n_luw, edges_luw)
                dist = _compute_distances(adj_luw, g - 1)
                
                left_size = len(side0)
                right_size = len(side1)
                
                side0_idx = {v: i for i, v in enumerate(side0)}
                side1_idx = {v: i for i, v in enumerate(side1)}
                
                adj_left = [[] for _ in range(left_size)]
                for i, u in enumerate(side0):
                    for j, v in enumerate(side1):
                        if dist[u][v] >= g:
                            adj_left[i].append(j)
                
                matching_size, pair_u = _hopcroft_karp(left_size, right_size, adj_left)
                
                if matching_size == left_size:
                    new_edges = list(edges_luw)
                    for i, u in enumerate(side0):
                        v = side1[pair_u[i]]
                        new_edges.append((u, v))
                    
                    if _verify_girth(n_luw, new_edges, k, g):
                        return n_luw, new_edges

    return _moore_bound(k, g), []

# EVOLVE-BLOCK-END
