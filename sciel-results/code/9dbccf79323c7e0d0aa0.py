"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import math
import numpy as np

# Cache for displacements
_DISP_CACHE = None

def _precompute_displacements(max_L=9):
    """
    Precompute the set of possible net displacements (k_u, k_v)
    for non-backtracking walks of length L over generators {u, -u, v, -v}.
    """
    displacements = [set() for _ in range(max_L + 1)]
    
    # Generators represented as (du, dv)
    S = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for L in range(2, max_L + 1):
        def gen(start, length, sum_u, sum_v, prev):
            if length == 0:
                displacements[start].add((sum_u, sum_v))
                return
            for s in S:
                if s == (-prev[0], -prev[1]):
                    continue
                gen(start, length - 1, sum_u + s[0], sum_v + s[1], s)
        
        for s in S:
            gen(L, L - 1, s[0], s[1], s)
            
    return displacements

def _get_displacements():
    global _DISP_CACHE
    if _DISP_CACHE is None:
        _DISP_CACHE = _precompute_displacements(9)
    return _DISP_CACHE

def _check_girth_full(a, b, u1, u2, v1, v2):
    displacements = _get_displacements()
    for L in range(3, 10):
        for ku, kv in displacements[L]:
            if (ku * u1 + kv * v1) % a == 0 and (ku * u2 + kv * v2) % b == 0:
                return False
    return True

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    if g != 10:
        return _construct_fallback(k, g)

    # Search for a Cayley graph on Z_a x Z_b with n < 2048
    # Moore bound is 322.
    best_n = None
    best_edges = None
    
    # Search n from 322 to 1500
    for n in range(322, 1500):
        for a in range(1, int(math.isqrt(n)) + 1):
            if n % a != 0:
                continue
            b = n // a
            
            # Check connectedness conditions
            # We iterate u, v
            # To speed up, we can break early if we find one.
            
            # We can restrict u1, v1 to be in [0, a)
            # And u2, v2 in [0, b)
            
            # Optimization: Only check if gcd(u1, v1, a) == 1 and gcd(u2, v2, b) == 1
            # This ensures the projections generate the factors.
            
            for u1 in range(a):
                for v1 in range(a):
                    if math.gcd(math.gcd(u1, v1), a) != 1:
                        continue
                    for u2 in range(b):
                        for v2 in range(b):
                            if math.gcd(math.gcd(u2, v2), b) != 1:
                                continue
                            
                            # Check u != +/- v
                            if u1 == v1 and u2 == v2: continue
                            if (a - u1) % a == v1 and (b - u2) % b == v2: continue
                            
                            # Fast check
                            # L=3
                            if (3 * u1) % a == 0 and (3 * u2) % b == 0: continue
                            if (3 * v1) % a == 0 and (3 * v2) % b == 0: continue
                            if (u1 + 2 * v1) % a == 0 and (u2 + 2 * v2) % b == 0: continue
                            if (2 * u1 + v1) % a == 0 and (2 * u2 + v2) % b == 0: continue
                            
                            # Full check
                            if _check_girth_full(a, b, u1, u2, v1, v2):
                                # Construct edges
                                edges = set()
                                for i in range(n):
                                    # Map i to (x, y) in Z_a x Z_b
                                    x = i % a
                                    y = i // a
                                    
                                    # Neighbor 1: (x+u1, y+u2)
                                    nx1 = (x + u1) % a
                                    ny1 = (y + u2) % b
                                    n1 = nx1 + ny1 * a
                                    
                                    # Neighbor 2: (x+v1, y+v2)
                                    nx2 = (x + v1) % a
                                    ny2 = (y + v2) % b
                                    n2 = nx2 + ny2 * a
                                    
                                    if x < nx1 or (x == nx1 and y < ny1):
                                        edges.add((i, n1))
                                    elif x > nx1 or (x == nx1 and y > ny1):
                                        edges.add((n1, i))
                                    else:
                                        # If x==nx1 and y==ny1, it's a self loop? No, u!=0
                                        # If x==nx1, y!=ny1
                                        if y < ny1:
                                            edges.add((i, n1))
                                        else:
                                            edges.add((n1, i))
                                        
                                    if x < nx2 or (x == nx2 and y < ny2):
                                        edges.add((i, n2))
                                    elif x > nx2 or (x == nx2 and y > ny2):
                                        edges.add((n2, i))
                                    else:
                                        if y < ny2:
                                            edges.add((i, n2))
                                        else:
                                            edges.add((n2, i))
                                        
                                best_n = n
                                best_edges = sorted(edges)
                                break
                        if best_n is not None: break
                    if best_n is not None: break
                if best_n is not None: break
            if best_n is not None: break
        if best_n is not None: break
        
    if best_n is not None:
        return best_n, best_edges
        
    return _construct_fallback(4, g)

def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
