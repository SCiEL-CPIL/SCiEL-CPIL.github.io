import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # LUW Construction with optimized coordinate handling
    q = k

    # Find smallest d such that girth >= g
    # For LUW: girth >= d+5 if d odd, d+4 if d even
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

    # Finite-field helpers
    if q == 4:
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q

    # Recurrence rule for the LUW construction
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]

    # Precompute q powers for encoding
    q_pow = [1] * (d + 1)
    for i in range(d - 1, -1, -1):
        q_pow[i] = q_pow[i+1] * q
    total_coords = q_pow[0]  # q^d

    def encode(side, coords):
        val = side * total_coords
        for i in range(1, d + 1):
            val += coords[i] * q_pow[i]
        return val

    # Precompute all vertex keys and their (side, coords)
    all_keys = []
    all_side = []
    all_coords = []
    for side in range(2):
        base_offset = side * total_coords
        for rem in range(total_coords):
            coords = [0] * (d + 1)
            tmp = rem
            for i in range(d, 0, -1):
                coords[i] = tmp % q
                tmp //= q
            all_keys.append(base_offset + rem)
            all_side.append(side)
            all_coords.append(coords)

    n_total = len(all_keys)

    # Build adjacency via LUW rules
    # For each vertex, compute its q neighbors
    adj = [list() for _ in range(n_total)]
    edge_set = set()

    for idx in range(n_total):
        side = all_side[idx]
        coords = all_coords[idx]
        for first_val in range(q):
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = encode(1 - side, ncoords)
            # Find index of nkey
            # nkey = (1-side)*total_coords + rem
            nside = 1 - side
            nrem = nkey - nside * total_coords
            nidx = nside * total_coords + nrem
            adj[idx].append(nidx)
            edge = (idx, nidx) if idx < nidx else (nidx, idx)
            if edge not in edge_set:
                edge_set.add(edge)

    # Build edge list from adjacency (each undirected edge once)
    edge_list = []
    for u in range(n_total):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))

    n_luw = n_total

    if n_luw == 0 or not edge_list:
        return n_luw, edge_list

    # Check all vertices have degree exactly k
    valid = True
    for i in range(n_luw):
        if len(adj[i]) != k:
            valid = False
            break

    if not valid:
        return n_luw, edge_list

    # Verify girth using non-backtracking walks
    max_depth = (g - 1) // 2
    girth_ok = True
    for start in range(n_luw):
        if time.time() > deadline - 5:
            break
        visited = [-1] * n_luw
        parent = [-1] * n_luw
        queue2 = deque([(start, 0)])
        visited[start] = 0
        while queue2:
            node, depth = queue2.popleft()
            if depth >= max_depth:
                continue
            for nei in adj[node]:
                if visited[nei] == -1:
                    visited[nei] = depth + 1
                    parent[nei] = node
                    queue2.append((nei, depth + 1))
                elif nei != parent[node]:
                    cycle_len = depth + 1 + visited[nei]
                    if cycle_len < g:
                        girth_ok = False
                        break
            if not girth_ok:
                break
        if not girth_ok:
            break

    return n_luw, edge_list
