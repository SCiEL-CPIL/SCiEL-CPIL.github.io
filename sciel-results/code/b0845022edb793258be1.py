import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def seed_construct(k, g):
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

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def check_connected(n, adj):
        visited = [False] * n
        q = deque([0])
        visited[0] = True
        count = 1
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    q.append(v)
        return count == n

    def try_sl2_11_cayley():
        if time.time() > deadline:
            return None
            
        mod = 11
        # Generate all elements of SL(2, 11)
        # Matrix [[a, b], [c, d]] with ad - bc = 1 mod 11
        elements = []
        for a in range(mod):
            for b in range(mod):
                for c in range(mod):
                    # ad - bc = 1 => ad = 1 + bc
                    # If a == 0, then -bc = 1 => bc = -1 = 10.
                    # If a != 0, d = (1 + bc) * a^-1
                    if a == 0:
                        if b == 0:
                            continue
                        # bc = 10 mod 11
                        # c = 10 * b^-1
                        try:
                            b_inv = pow(b, -1, mod)
                        except:
                            continue
                        c_val = (10 * b_inv) % mod
                        # d can be anything? No, ad - bc = 1.
                        # 0*d - b*c = 1 => -bc = 1 => bc = 10.
                        # d is free? No, the determinant is 1.
                        # Wait, SL(2, q) definition: det=1.
                        # If a=0, det = -bc = 1 => bc = -1.
                        # d is not constrained by det if a=0?
                        # det = ad - bc. If a=0, det = -bc.
                        # So -bc = 1.
                        # d is free?
                        # Yes, if a=0, the condition is on b and c only.
                        # So for each valid (b, c), d can be any value?
                        # No, the matrix is [[0, b], [c, d]].
                        # det = 0*d - b*c = -bc.
                        # So -bc = 1.
                        # d is indeed free in SL(2, q) if a=0?
                        # Let's check.
                        # In SL(2, q), the number of elements is q(q^2-1).
                        # If we fix a, b, c, d is determined by ad = 1+bc.
                        # If a=0, 0 = 1+bc => bc = -1.
                        # Then d is NOT determined.
                        # This implies there are q choices for d for each valid (b,c) with a=0.
                        # So yes, d is free.
                        for d in range(mod):
                            elements.append((0, b, c_val, d))
                    else:
                        a_inv = pow(a, -1, mod)
                        # d = (1 + bc) * a_inv
                        # b and c can be anything?
                        # Yes.
                        for b in range(mod):
                            for c in range(mod):
                                d = ((1 + b * c) * a_inv) % mod
                                elements.append((a, b, c, d))
        
        # Remove duplicates (shouldn't be any, but just in case)
        elements = list(set(elements))
        
        # Map element to index
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        n = len(elements)
        
        # Identity
        identity = (1, 0, 0, 1)
        
        # Precompute multiplication table? No, just function.
        def mul(m1, m2):
            a, b, c, d = m1
            e, f, g, h = m2
            return (
                (a*e + b*g) % mod,
                (a*f + b*h) % mod,
                (c*e + d*g) % mod,
                (c*f + d*h) % mod
            )
            
        def inv(m):
            a, b, c, d = m
            return (d, (-b) % mod, (-c) % mod, a)

        # We need a symmetric set S of size 5.
        # Structure: 2 pairs of inverses + 1 involution.
        # x, x^-1, y, y^-1, z (z=z^-1)
        
        # Get list of elements excluding identity
        non_id = [e for e in elements if e != identity]
        
        # Random search for S
        for _ in range(1000):
            if time.time() > deadline:
                return None
                
            # Pick 2 pairs and 1 involution
            # To ensure distinctness, we pick indices
            
            # Pick involution z
            # z = z^-1 => z^2 = I
            # We can just pick a random element and check if it's an involution
            # Or pick from a precomputed list of involutions.
            # Let's just pick random elements and filter.
            
            candidates = non_id[:]
            random.shuffle(candidates)
            
            # Find 2 pairs
            x = None
            y = None
            z = None
            
            # We need x != x^-1, y != y^-1, z = z^-1.
            # And all 5 distinct.
            
            # Pick z (involution)
            for e in candidates:
                if e == inv(e):
                    z = e
                    break
            if z is None:
                continue
                
            # Pick x (not involution, not z)
            x_candidates = [e for e in candidates if e != z and e != inv(z) and e != inv(e)]
            if not x_candidates:
                continue
            x = x_candidates[0]
            
            # Pick y (not involution, not x, not x^-1, not z, not z^-1)
            excluded = {z, inv(z), x, inv(x)}
            y_candidates = [e for e in x_candidates if e not in excluded]
            if not y_candidates:
                continue
            y = y_candidates[0]
            
            S = [x, inv(x), y, inv(y), z]
            
            # Check distinctness
            if len(set(S)) != 5:
                continue
                
            # Build adjacency
            adj = [[] for _ in range(n)]
            for i, g_elem in enumerate(elements):
                for s in S:
                    neighbor = mul(g_elem, s)
                    j = elem_to_idx[neighbor]
                    adj[i].append(j)
                    
            # Verify
            if not check_connected(n, adj):
                continue
                
            if exact_girth_check(n, adj, g):
                # Convert adj to edges
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
        return None

    # Try voltage graph construction
    result = try_sl2_11_cayley()
    if result is not None:
        n, edges = result
        # Final verification
        adj = build_adj(edges, n)
        if exact_girth_check(n, adj, g):
            return n, edges
    
    # Fallback to seed
    seed_n, seed_edges = seed_construct(k, g)
    return seed_n, seed_edges
