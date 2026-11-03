import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145.0

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

    seed_n, seed_edges = seed_construct(k, g)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check_fast(n, adj, g):
        """Fast girth check using BFS from each vertex with early stopping."""
        half_g = g // 2
        for s in range(n):
            if time.time() > deadline - 2:
                return False
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                du = dist[u]
                if du >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        if du + dist[v] + 1 < g:
                            return False
        return True

    def heuristic_girth_check(n, adj, g, sample_size=10):
        """Check girth from random sources."""
        sources = random.sample(range(n), min(sample_size, n))
        half_g = g // 2
        for s in sources:
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                du = dist[u]
                if du >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        if du + dist[v] + 1 < g:
                            return False
        return True

    def generate_simple_regular(n, k):
        """Generate a simple k-regular graph using configuration model."""
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            stubs = list(range(n * k))
            for i in range(n):
                for j in range(k):
                    stubs[i * k + j] = i
            random.shuffle(stubs)

            edges = []
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                edges.append((u, v))

            # Repair self-loops and multi-edges
            edge_set = set()
            bad_indices = []

            for i, (u, v) in enumerate(edges):
                if u == v:
                    bad_indices.append(i)
                    continue
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    bad_indices.append(i)
                else:
                    edge_set.add(key)

            if not bad_indices:
                return edges

            # Try to fix bad edges
            max_fix_iters = n * 2
            for _ in range(max_fix_iters):
                if time.time() > deadline - 5:
                    return None
                if not bad_indices:
                    break

                bi = random.choice(bad_indices)
                u1, v1 = edges[bi]

                fixed = False
                for _ in range(30):
                    j = random.randint(0, len(edges) - 1)
                    if j == bi:
                        continue
                    u2, v2 = edges[j]
                    if u2 == v2:
                        continue

                    # Try swap: (u1,v1) and (u2,v2) -> (u1,v2) and (u2,v1)
                    new_e1 = (u1, v2)
                    new_e2 = (u2, v1)

                    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                        continue

                    key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                    key2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))

                    old_key1 = (min(u1, v1), max(u1, v1)) if u1 != v1 else None
                    old_key2 = (min(u2, v2), max(u2, v2)) if u2 != v2 else None

                    if old_key1 in edge_set:
                        edge_set.discard(old_key1)
                    if old_key2 in edge_set:
                        edge_set.discard(old_key2)

                    if key1 in edge_set or key2 in edge_set:
                        if old_key1:
                            edge_set.add(old_key1)
                        if old_key2:
                            edge_set.add(old_key2)
                        continue

                    edge_set.add(key1)
                    edge_set.add(key2)
                    edges[bi] = new_e1
                    edges[j] = new_e2
                    bad_indices.remove(bi)
                    fixed = True
                    break

                if not fixed:
                    break

            # Final check
            if not bad_indices:
                edge_count = set()
                valid = True
                for u, v in edges:
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_count:
                        valid = False
                        break
                    edge_count.add(key)
                if valid:
                    return edges

        return None

    def local_search_improve_girth(n, edges, k, g, max_iters=500):
        """Try to increase girth by swapping edges."""
        adj = build_adj(edges, n)
        current_girth = g  # Assume we start with girth >= g

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Pick a random edge
            ei = random.randint(0, len(edges) - 1)
            u1, v1 = edges[ei]

            # Pick another random edge not sharing vertices
            for _ in range(20):
                ej = random.randint(0, len(edges) - 1)
                if ej == ei:
                    continue
                u2, v2 = edges[ej]
                if u2 in (u1, v1) or v2 in (u1, v1):
                    continue

                # Try swap: (u1,v1),(u2,v2) -> (u1,v2),(u2,v1)
                new_e1 = (u1, v2)
                new_e2 = (u2, v1)

                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue

                # Check no multi-edges
                key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                key2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))

                # Build edge set for quick lookup
                es = set()
                for a, b in edges:
                    if a == b:
                        continue
                    es.add((min(a, b), max(a, b)))

                old_key1 = (min(u1, v1), max(u1, v1))
                old_key2 = (min(u2, v2), max(u2, v2))
                es.discard(old_key1)
                es.discard(old_key2)

                if key1 in es or key2 in es:
                    continue

                # Apply swap
                edges[ei] = new_e1
                edges[ej] = new_e2

                # Check if girth is still >= g using heuristic
                adj = build_adj(edges, n)
                if not heuristic_girth_check(n, adj, g, sample_size=3):
                    # Revert
                    edges[ei] = (u1, v1)
                    edges[ej] = (u2, v2)

        return edges

    # Search
    best_n = seed_n
    best_edges = seed_edges

    search_candidates = []
    step = max(2, (seed_n - moore_bound) // 20)
    for n in range(moore_bound, min(seed_n, moore_bound + 200), step):
        if n % 2 == 0:
            search_candidates.append(n)
    for n in [428, 430, 450, 480, 500, 550, 600]:
        if n % 2 == 0 and n < seed_n and n >= moore_bound:
            if n not in search_candidates:
                search_candidates.append(n)
    search_candidates.sort()

    for n in search_candidates:
        if time.time() > deadline - 15:
            break
        attempts = max(2, 8 - (n - moore_bound) // 30)
        for _ in range(attempts):
            if time.time() > deadline - 15:
                break
            edges = generate_simple_regular(n, k)
            if edges is None:
                continue
            adj = build_adj(edges, n)
            if heuristic_girth_check(n, adj, g, sample_size=5):
                if exact_girth_check_fast(n, adj, g):
                    return n, edges
    return seed_n, seed_edges
