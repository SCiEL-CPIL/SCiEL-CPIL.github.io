import time
import random
import sys
from collections import deque

N = 40
DEADLINE = 120.0
SAFETY = 15.0
START = time.perf_counter()
CUTOFF = START + (DEADLINE - SAFETY)
RNG = random.Random(20240607)


def _c6_exists(adj):
    """Return True iff the graph (adjacency bitmasks) contains a 6-cycle.
    Counts all 6-cycles; if any exist, count > 0."""
    total = 0
    for a in range(N):
        na = adj[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            nb = adj[b]
            for c in range(b + 1, N):
                if not (nb >> c) & 1:
                    continue
                nc = adj[c]
                for d in range(c + 1, N):
                    if not (nc >> d) & 1:
                        continue
                    nd = adj[d]
                    for e in range(d + 1, N):
                        if not (nd >> e) & 1:
                            continue
                        ne = adj[e]
                        for f in range(e + 1, N):
                            if not (ne >> f) & 1:
                                continue
                            # close: f-a and f-b
                            if (adj[f] >> a) & 1 and (adj[f] >> b) & 1:
                                total += 1
                                if total > 4:  # early stop: C6 definitely exists
                                    return True
    return total > 0


def _greedy(adj):
    """Greedy edge addition: add edges in a random order if they keep the graph C6-free."""
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    RNG.shuffle(pairs)
    for u, v in pairs:
        if (adj[u] >> v) & 1:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if _c6_exists(adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        if time.perf_counter() > CUTOFF:
            break
    return adj


def _local_search(adj):
    """1-swap local search: try removing one edge and adding another to gain an edge.
    Returns the best adjacency matrix found."""
    best_adj = list(adj)
    best_m = sum(bin(x).count("1") for x in best_adj) // 2
    no_improve = 0
    while time.perf_counter() < CUTOFF:
        improved = False
        # candidate remove/insert edges
        present = [(i, j) for i in range(N) for j in range(i + 1, N) if (adj[i] >> j) & 1]
        absent = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] >> j) & 1]
        RNG.shuffle(absent)
        for u, v in absent:
            if time.perf_counter() > CUTOFF:
                break
            for i, j in present:
                if i == u or i == v or j == u or j == v:
                    continue
                # remove (i,j), add (u,v)
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _c6_exists(adj):
                    # revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                else:
                    improved = True
                    best_m += 1
                    break
            if improved:
                break
        if improved:
            no_improve = 0
            if best_m > sum(bin(x).count("1") for x in best_adj) // 2:
                best_adj = list(adj)
        else:
            no_improve += 1
            if no_improve >= 3:
                break
    return best_adj, best_m


def construct():
    t0 = time.perf_counter()
    best_m = 0
    best_adj = [0] * N

    # Start from a dense-ish seed: try several random greedy starts, keep best, then local search.
    for attempt in range(200):
        if time.perf_counter() > CUTOFF:
            break
        adj = [0] * N
        adj = _greedy(adj)
        m = sum(bin(x).count("1") for x in adj) // 2
        if m > best_m:
            best_m = m
            best_adj = list(adj)
        # local search on this graph
        if time.perf_counter() > CUTOFF - 2:
            break
        ls_adj, ls_m = _local_search(adj)
        if ls_m > best_m:
            best_m = ls_m
            best_adj = ls_adj

    # Final safety: ensure no C6 in best_adj
    if _c6_exists(best_adj):
        # fallback: rebuild a safe graph (8 copies of K5 = 80 edges, no C6)
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([5 * k + a, 5 * k + b])
        return edges

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
