"""Clique partition via iterative maximal-clique extraction with bitset adjacency."""
import time
from random import Random


def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    if not edges:
        return []

    # Bitset adjacency: adj_bits[v] is an int where bit u is set if (u,v) is an edge
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    remaining = set(edges)
    cliques = []

    while remaining:
        if time.time() > deadline:
            for e in remaining:
                cliques.append(list(e))
            break

        # Score each remaining edge by the size of the common-neighbor bitset
        best_edge = None
        best_score = -1
        for e in remaining:
            u, v = e
            common = adj_bits[u] & adj_bits[v]
            score = bin(common).count('1')
            if score > best_score:
                best_score = score
                best_edge = e

        if best_edge is None:
            break

        u0, v0 = best_edge
        # Greedy maximal clique via bitset: start from edge, expand
        C = [u0, v0]
        P = (adj_bits[u0] & adj_bits[v0]) & ~((1 << u0) | (1 << v0))

        while P:
            if time.time() > deadline:
                break
            # Pick vertex with most connections to C
            best_v = -1
            best_deg = -1
            mask = P
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                deg = bin(adj_bits[v] & ((1 << u0) | (1 << v0) | (sum((1 << c) for c in C[2:])) if len(C) > 2 else 0)).count('1')
                # Simpler: count common neighbors with all of C
                common_with_C = P & adj_bits[v]
                cnt = bin(common_with_C).count('1')
                if cnt > best_deg:
                    best_deg = cnt
                    best_v = v
                mask ^= lb

            if best_v < 0:
                break
            C.append(best_v)
            P = P & adj_bits[best_v]

        cliques.append(C)

        # Remove edges within C
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                a, b = C[i], C[j]
                e = (a, b) if a < b else (b, a)
                if e in remaining:
                    remaining.discard(e)
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)

    # Merge phase
    adj_set = [set() for _ in range(n)]
    for u in range(n):
        mask = adj_bits[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            adj_set[u].add(v)
            mask ^= lb

    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        i = 0
        while i < len(cliques):
            merged = False
            for j in range(i + 1, len(cliques)):
                if time.time() > deadline:
                    return cliques
                C1, C2 = cliques[i], cliques[j]
                ok = all(b in adj_set[a] for a in C1 for b in C2)
                if ok:
                    cliques[i] = C1 + C2
                    del cliques[j]
                    merged = True
                    changed = True
                    break
            if not merged:
                i += 1

    return cliques


def construct():
    deadline = time.time() + 110
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
        if time.time() > deadline - 10:
            edges = _random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            break
    return result
