"""LPS and LUW algebraic constructions for 3-regular graphs of large girth."""
import time
import random
import sys
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    def verify_girth(n, edges, min_girth):
        """Return True iff girth >= min_girth."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        max_depth = min_girth // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def build_lps(q):
        """Build a 3-regular LPS graph over F_q."""
        # q must be odd prime or prime power, q ≡ ±1 mod 6 for degree 3?
        # Actually, LPS works for any q ≡ ±1 mod 6 for the standard generators.
        # Let's assume q is a prime for simplicity.
        
        # Matrices in SL(2, F_q)
        # A = [[0, 1], [-1, 0]]
        # B = [[0, 1], [1, 0]]
        # C = [[1, 1], [0, 1]]
        
        # Vertices are pairs (x, y) in F_q^2 \ {(0,0)} modulo scalar multiplication?
        # No, LPS vertices are cosets of PSL(2, F_q) in SL(2, F_q) ?
        # Actually, the LPS graph vertices are the elements of PSL(2, F_q) ?
        # No, the number of vertices is q(q^2-1)/2.
        # This is the number of elements in PSL(2, F_q).
        
        # We represent elements of SL(2, F_q) as tuples (a, b, c, d) with ad-bc=1.
        # We identify M and -M.
        
        # Generators:
        # A = (0, 1, -1, 0)
        # B = (0, 1, 1, 0)
        # C = (1, 1, 0, 1)
        
        # We need to map group elements to indices 0..n-1.
        
        # Generate all elements of SL(2, F_q)
        elements = []
        seen = set()
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    for d in range(q):
                        if (a*d - b*c) % q == 1:
                            # Normalize: take (a,b,c,d) or (-a,-b,-c,-d)
                            # We want a canonical representative.
                            # Let's say if a != 0, we make a > 0? No, F_q.
                            # We identify M and -M.
                            # Canonical: if (a,b,c,d) < (-a,-b,-c,-d) lexicographically, use it.
                            rep1 = (a, b, c, d)
                            rep2 = ((-a)%q, (-b)%q, (-c)%q, (-d)%q)
                            if rep1 < rep2:
                                rep = rep1
                            else:
                                rep = rep2
                            if rep not in seen:
                                seen.add(rep)
                                elements.append(rep)
        
        n = len(elements)
        index = {e: i for i, e in enumerate(elements)}
        
        # Multiplication of matrices
        def mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            a = (a1*a2 + b1*c2) % q
            b = (a1*b2 + b1*d2) % q
            c = (c1*a2 + d1*c2) % q
            d = (c1*b2 + d1*d2) % q
            # Normalize
            rep1 = (a, b, c, d)
            rep2 = ((-a)%q, (-b)%q, (-c)%q, (-d)%q)
            if rep1 < rep2:
                return rep1
            else:
                return rep2

        A = (0, 1, q-1, 0) # -1 mod q
        B = (0, 1, 1, 0)
        C = (1, 1, 0, 1)
        
        # Normalize generators
        def norm(m):
            a, b, c, d = m
            rep1 = (a, b, c, d)
            rep2 = ((-a)%q, (-b)%q, (-c)%q, (-d)%q)
            if rep1 < rep2:
                return rep1
            else:
                return rep2
        
        A = norm(A)
        B = norm(B)
        C = norm(C)
        
        edges = []
        for i, e in enumerate(elements):
            for gen in [A, B, C]:
                # e * gen
                prod = mul(e, gen)
                j = index[prod]
                if i < j:
                    edges.append((i, j))
        
        return n, edges

    def build_luw(q, d):
        """Build a 3-regular LUW graph."""
        if q == 3:
            add_f = lambda a, b: (a + b) % 3
            sub_f = lambda a, b: (a - b) % 3
            mul_f = lambda a, b: (a * b) % 3
        elif q == 4:
            ta = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
            tm = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add_f = lambda a, b: ta[a][b]
            sub_f = lambda a, b: ta[a][b]
            mul_f = lambda a, b: tm[a][b]
        else:
            return None, None

        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            options = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
            rule[c] = options[r]

        start = (0, tuple([0] * (d + 1)))
        index = {start: 0}
        order = [start]
        edges = []
        
        for idx, vertex in enumerate(order):
            if time.time() - t0 > deadline - 5:
                return None, None
            side, x = vertex
            x_list = list(x)
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_f(x[c], mul_f(y[a], x[b]))
                    else:
                        y[c] = sub_f(x[c], mul_f(x[a], y[b]))
                neighbor = (1 - side, tuple(y))
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                if side == 0:
                    edges.append((idx, index[neighbor]))
        
        return len(order), edges

    best_n = float('inf')
    best_edges = None

    # Try LPS constructions
    for q in [7, 13, 19, 25, 31, 37, 43]:
        if time.time() - t0 > deadline - 20:
            break
        n, edges = build_lps(q)
        if n is not None:
            if verify_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    break

    # Try LUW constructions if LPS didn't find a good one
    if best_edges is None or best_n > 1000:
        for q in [3, 4]:
            for d in range(6, 12):
                if time.time() - t0 > deadline - 10:
                    break
                n, edges = build_luw(q, d)
                if n is not None:
                    if verify_girth(n, edges, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    # Fallback
    return 0, []
# EVOLVE-BLOCK-END
