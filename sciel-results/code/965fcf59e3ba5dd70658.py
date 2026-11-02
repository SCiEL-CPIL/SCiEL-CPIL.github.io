import time
import random
import heapq
import sys
import numpy as np

def _luw_construct(k, g):
    # Copy of the seed logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

def _construct_lps(p):
    """Construct the LPS graph for prime p. Returns n, edges."""
    # Generate SL(2, p)
    # Matrices are [[a, b], [c, d]] with ad - bc = 1
    mats = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # ad - bc = 1 => ad = 1 + bc
                # If a == 0, then -bc = 1 => bc = -1
                # If a != 0, d = (1 + bc) * a^{-1}
                if a == 0:
                    if (b * c) % p == (p - 1) % p: # bc = -1
                        for d in range(p):
                            if (a * d - b * c) % p == 1:
                                mats.append((a, b, c, d))
                else:
                    inv_a = pow(a, p - 2, p)
                    d = (1 + b * c) * inv_a % p
                    if (a * d - b * c) % p == 1:
                        mats.append((a, b, c, d))
    
    n = len(mats)
    index = {m: i for i, m in enumerate(mats)}
    
    # Generators
    # A = [[0, -1], [1, 0]] = [[0, p-1], [1, 0]]
    # B = [[1, 1], [0, 1]]
    # A_inv = [[0, 1], [-1, 0]] = [[0, 1], [p-1, 0]]
    # B_inv = [[1, -1], [0, 1]] = [[1, p-1], [0, 1]]
    
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p, (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
        
    A = (0, p-1, 1, 0)
    B = (1, 1, 0, 1)
    A_inv = (0, 1, p-1, 0)
    B_inv = (1, p-1, 0, 1)
    
    generators = [A, B, A_inv, B_inv]
    
    edges = set()
    for i, m in enumerate(mats):
        for g in generators:
            m2 = mat_mul(m, g)
            if m2 in index:
                j = index[m2]
                if i < j:
                    edges.add((i, j))
    
    return n, list(edges)

def _cartesian_product_k2(n, edges):
    """Take Cartesian product with K2. Returns new_n, new_edges."""
    new_n = 2 * n
    new_edges = []
    # Edges in layer 0 and 1
    for u, v in edges:
        new_edges.append((u, v))
        new_edges.append((u + n, v + n))
    # Vertical edges
    for i in range(n):
        new_edges.append((i, i + n))
    return new_n, new_edges

def _check_girth(n, edges, g_target):
    """Check if girth >= g_target. Returns True if valid."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            if dist[u] >= g_target:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v == parent[u]:
                        continue
                    L = dist[u] + dist[v] + 1
                    if L < g_target:
                        return False
    return True

def construct(k, g):
    if k == 5 and g == 9:
        # Try LPS p=7
        # X_7 has 168 vertices.
        # If girth >= 9, product with K2 has 336 vertices.
        n_lps, edges_lps = _construct_lps(7)
        if _check_girth(n_lps, edges_lps, g):
            n_final, edges_final = _cartesian_product_k2(n_lps, edges_lps)
            return n_final, edges_final
            
        # Try LPS p=13? n=1092. Product 2184. Too big.
        # Fall back to seed
        return _luw_construct(k, g)
        
    return _luw_construct(k, g)
