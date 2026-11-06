"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 110

    def max_book_fast(adj, edges):
        mb = 0
        for (u, v) in edges:
            common = adj[u] & adj[v]
            bs = common.bit_count()
            if bs > mb:
                mb = bs
        return mb

    best_edges = None
    best_max = 999

    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        adj = [0] * n
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                key = (min(u,v), max(u,v))
                if key not in edges:
                    edges.add(key)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        current_max = max_book_fast(adj, edges)

        stuck = 0
        while time.time() < deadline and current_max > 7:
            e1 = random.choice(list(edges))
            u1, v1 = e1
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e1)

            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 != v2 and (min(u2,v2), max(u2,v2)) not in edges:
                    break

            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add((min(u2,v2), max(u2,v2)))

            new_max = max_book_fast(adj, edges)

            if new_max > current_max:
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edges.add(e1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                edges.discard((min(u2,v2), max(u2,v2)))
                stuck += 1
            else:
                current_max = new_max
                stuck = 0

            if stuck > 300:
                break

        if current_max < best_max:
            best_max = current_max
            best_edges = edges.copy()
            if best_max <= 7:
                break

    if best_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return [[u, v] for (u, v) in sorted(best_edges)]
# EVOLVE-BLOCK-END
