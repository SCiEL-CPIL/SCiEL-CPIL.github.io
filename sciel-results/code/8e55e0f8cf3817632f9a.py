"""Construct a 5-regular graph of girth >= g on as few vertices as possible.
Approach: search random k-regular graphs from a low vertex count upward,
certifying girth exactly via non-backtracking walks from every vertex.
This is a different family (randomized search + exact girth certification)
than the seed's explicit algebraic construction, and can find much smaller
valid graphs.
"""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque


def _has_short_cycle(adj, g):
    """Exact: return True iff there is a non-backtracking cycle of length < g."""
    n = len(adj)
    adj_sets = [set(a) for a in adj]
    directed_edges = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                directed_edges.append((u, v))
                directed_edges.append((v, u))
    for (su, sv) in directed_edges:
        q = deque()
        q.append((sv, su, 1))
        seen = set()
        seen.add((sv, su))
        while q:
            u, prev, length = q.popleft()
            if length >= g - 1:
                continue
            for w in adj_sets[u]:
                if w == prev:
                    continue
                if w == su and length + 1 >= 3:
                    return True
                if (w, u) not in seen:
                    seen.add((w, u))
                    q.append((w, u, length + 1))
    return False


def _random_k_regular(n, k, rng):
    if (n * k) % 2 != 0:
        return None
    half = n * k // 2
    for _ in range(60):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, half, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if ok:
            # ensure exact degree k
            if all(len(a) == k for a in adj):
                return [sorted(a) for a in adj]
    return None


def construct(k, g):
    start_time = time.time()
    TIME_LIMIT = 112.0
    rng = random.Random(20260501)

    # Floor: use the task's stated best lower bound (428) for (5,9); for other
    # (k,g) fall back to the Moore bound. We never go below the Moore bound.
    def moore_bound(k, g):
        if g % 2 == 0:
            r = g // 2
            return k * (k - 1) ** (r - 1)
        else:
            r = (g - 1) // 2
            return 2 + k * (k - 1) ** r

    mb = moore_bound(k, g)
    # For the specific (5,9) the task cites a stronger lower bound of 428;
    # use max(mb, 428) only when (k,g)==(5,9) to respect the stated bound.
    if (k, g) == (5, 9):
        floor = max(mb, 428)
    else:
        floor = mb

    best_n = None
    best_adj = None

    # Search upward from floor for a valid k-regular graph of girth >= g.
    max_n = 2000
    for n in range(floor, min(floor + 800, max_n + 1)):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        attempts = 12
        for _ in range(attempts):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            adj = _random_k_regular(n, k, rng)
            if adj is None:
                continue
            if not _has_short_cycle(adj, g):
                best_n = n
                best_adj = adj
                break
        if best_adj is not None:
            break

    # Fallback: if nothing found in the window, do a wider random search at
    # increasing sizes until we hit the time limit, always keeping the best.
    if best_adj is None:
        n = floor
        while n <= max_n and time.time() - start_time < TIME_LIMIT - 5:
            for _ in range(15):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                adj = _random_k_regular(n, k, rng)
                if adj is not None and not _has_short_cycle(adj, g):
                    best_n = n
                    best_adj = adj
                    break
            if best_adj is not None:
                break
            n += 24  # jump to next plausible size

    # Absolute last resort: grow a definitely-valid graph by taking a large
    # random regular graph and, if it has short cycles, keep increasing size
    # (random regular graphs' girth grows with n, so this terminates).
    if best_adj is None:
        n = max(floor, 600)
        while time.time() - start_time < TIME_LIMIT - 2:
            for _ in range(20):
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                adj = _random_k_regular(n, k, rng)
                if adj is not None and not _has_short_cycle(adj, g):
                    best_n = n
                    best_adj = adj
                    break
            if best_adj is not None:
                break
            n += 100

    if best_adj is None:
        # Should not happen; return a trivial valid k-regular graph if k allows
        # (e.g., complete graph only works for small k). As a safety net,
        # return a cycle-based k-regular graph with large girth by using a
        # very large random regular graph.
        n = 1500
        for _ in range(200):
            adj = _random_k_regular(n, k, rng)
            if adj is not None and not _has_short_cycle(adj, g):
                best_n, best_adj = n, adj
                break
        if best_adj is None:
            raise RuntimeError("failed to construct valid graph")

    # Build edge list
    adj = best_adj
    edges = []
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return len(adj), edges
# EVOLVE-BLOCK-END
