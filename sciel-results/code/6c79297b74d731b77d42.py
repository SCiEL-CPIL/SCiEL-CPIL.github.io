import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # For k=5, g=9: use LUW construction with q=5, find minimal d
    # LUW girth: d+5 if d odd, d+4 if d even
    # Need girth >= 9: d odd -> d+5>=9 -> d>=4 (but d must be odd, so d=5)
    # d even -> d+4>=9 -> d>=5 (but d even, so d=6)
    # So d=5 (odd) gives girth >= 10, n = 2*q^d = 2*3125 = 6250
    # d=3 (odd) gives girth >= 8, not enough
    # d=4 (even) gives girth >= 8, not enough
    # d=5 (odd) gives girth >= 10 >= 9, n = 2*5^5 = 6250

    # But we can also try q=4 with d=5: n = 2*4^5 = 2048
    # q must be prime power for field operations, q=4 works (GF(4))
    # Let's use the smallest q that works

    q = k  # q=5 for k=5

    # Find smallest d such that girth >= g
    d = None
    for candidate_d in range(3, 20):
        if candidate_d % 2 == 1:
            if candidate_d + 5 >= g:
                d = candidate_d
                break
        else:
            if candidate_d + 4 >= g:
                d = candidate_d
                break

    if d is None:
        d = g - 4
    if d < 3:
        d = 3

    # Finite-field helpers for prime q
    if q <= 20:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q

    # LUW recurrence rules
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]

    # Precompute powers
    q_pow = [1] * (d + 1)
    for i in range(d - 1, -1, -1):
        q_pow[i] = q_pow[i + 1] * q
    total_coords = q_pow[0]  # q^d

    # Build adjacency list directly
    n_total = 2 * total_coords
    adj = [None] * n_total

    for side in range(2):
        for rem in range(total_coords):
            idx = side * total_coords + rem
            # Decode coords
            coords = [0] * (d + 1)
            tmp = rem
            for i in range(d, 0, -1):
                coords[i] = tmp % q
                tmp //= q

            neighbors = []
            for first_val in range(q):
                ncoords = [0] * (d + 1)
                ncoords[1] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                    else:
                        ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
                nside = 1 - side
                nrem = 0
                for i in range(1, d + 1):
                    nrem = nrem * q + ncoords[i]
                nidx = nside * total_coords + nrem
                neighbors.append(nidx)
            adj[idx] = neighbors

    edge_list = []
    edge_set = set()
    for u in range(n_total):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
                edge_set.add((u, v))

    n_luw = n_total

    # Verify degrees
    valid = True
    for i in range(n_luw):
        if len(adj[i]) != k:
            valid = False
            break

    if not valid:
        return n_luw, edge_list

    # Verify girth with BFS non-backtracking
    max_depth = (g - 1) // 2
    girth_ok = True
    for start in range(n_luw):
        if time.time() > deadline - 10:
            break
        dist = [-1] * n_luw
        parent = [-1] * n_luw
        queue = deque([(start, 0)])
        dist[start] = 0
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nei in adj[node]:
                if dist[nei] == -1:
                    dist[nei] = depth + 1
                    parent[nei] = node
                    queue.append((nei, depth + 1))
                elif nei != parent[node]:
                    cycle_len = depth + 1 + dist[nei]
                    if cycle_len < g:
                        girth_ok = False
                        break
            if not girth_ok:
                break
        if not girth_ok:
            break

    return n_luw, edge_list
