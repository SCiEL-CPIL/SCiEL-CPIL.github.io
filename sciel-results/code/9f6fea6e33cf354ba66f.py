"""Construct a large graph with no regular induced subgraph on 6+ vertices."""
import time
import random
from itertools import combinations

START = time.time()
DEADLINE = START + 140.0  # safety margin under the 150s kill


def _is_regular(adj, sub):
    """Check whether the induced subgraph on vertex set `sub` (a bitmask)
    is regular (all vertices have the same degree within the subgraph)."""
    k = sub.bit_count()
    if k < 6:
        return False
    # degrees of each vertex in the induced subgraph
    first = None
    m = sub
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        deg = (adj[v] & sub).bit_count()
        if first is None:
            first = deg
        elif deg != first:
            return False
        m ^= lsb
    return True


def _any_regular(adj):
    """Return True if any subset of >=6 vertices induces a regular subgraph."""
    n = len(adj)
    for k in range(6, n + 1):
        for sub in combinations(range(n), k):
            mask = 0
            for v in sub:
                mask |= 1 << v
            if _is_regular(adj, mask):
                return True
    return False


def _count_violations(adj, n):
    """Count how many subsets of >=6 vertices induce a regular subgraph."""
    cnt = 0
    for k in range(6, n + 1):
        for sub in combinations(range(n), k):
            mask = 0
            for v in sub:
                mask |= 1 << v
            if _is_regular(adj, mask):
                cnt += 1
    return cnt


def _greedy_insert(n, seed, start_edges=None, max_checks=200000):
    """Start from a graph on n vertices (start_edges or empty) and greedily
    add edges that do not create a regular induced subgraph on >=6 vertices.
    Returns (adj, edges) where adj is a list of int bitmasks."""
    rng = random.Random(seed)
    if start_edges is not None:
        adj = [0] * n
        for (u, v) in start_edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        edges = list(start_edges)
    else:
        adj = [0] * n
        edges = []

    # candidate edges in random order
    cands = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(cands)
    checks = 0
    for (u, v) in cands:
        if checks > max_checks:
            break
        if adj[u] & (1 << v):
            continue
        # tentatively add
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        checks += 1
        if _any_regular(adj):
            # undo
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
        else:
            edges.append((u, v))
    return adj, edges


def _local_search(adj, n, seed, iters=2000):
    """Try to add more edges via local search: attempt random single-edge
    additions and keep them if they don't create a violation."""
    rng = random.Random(seed)
    checks = 0
    for _ in range(iters):
        if time.time() > DEADLINE - 15:
            break
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        if adj[u] & (1 << v):
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        checks += 1
        if _any_regular(adj):
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
    return adj, checks


def construct():
    best_n = 5
    best_edges = []

    # Try to build graphs of increasing size via greedy insertion.
    # Known: 20-vertex graphs exist. Aim to reach as large as possible.
    for target_n in range(10, 21):
        if time.time() > DEADLINE - 30:
            break
        # multiple seeds for robustness
        for seed in range(0, 6):
            if time.time() > DEADLINE - 30:
                break
            adj, edges = _greedy_insert(target_n, seed, max_checks=60000)
            # verify no violation
            if not _any_regular(adj):
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                # try to grow a little more via local search
                if time.time() < DEADLINE - 30:
                    adj2, _ = _local_search(adj, target_n, seed + 100, iters=300)
                    if not _any_regular(adj2):
                        e2 = []
                        for u in range(target_n):
                            for v in range(u + 1, target_n):
                                if adj2[u] & (1 << v):
                                    e2.append([u, v])
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = e2

    # Build the final edge list as [u, v] pairs.
    final_edges = []
    if best_n == 5:
        final_edges = []
    else:
        for (u, v) in best_edges:
            final_edges.append([u, v])

    return {'n': best_n, 'edges': final_edges}
