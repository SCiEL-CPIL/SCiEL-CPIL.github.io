"""Suite task: dense odd-order graph avoiding equal-degree 3-path endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

N = 15
DEADLINE = time.time() + 80.0  # safety margin under 90s wall-clock


def _valid(adj):
    """True iff no 3-edge path s-u-v-t (s != t) has deg(s) == deg(t)."""
    deg = [len(adj[i]) for i in range(N)]
    for v in range(N):
        for u in adj[v]:
            if u >= v:
                continue
            du = deg[u]
            for w in adj[v]:
                if w == u:
                    continue
                if deg[w] == du:
                    return False
    return True


def _edge_count(adj):
    return sum(len(adj[i]) for i in range(N)) // 2


def _try_add(adj, e):
    u, v = e
    adj[u].add(v)
    adj[v].add(u)
    ok = _valid(adj)
    if not ok:
        adj[u].discard(v)
        adj[v].discard(u)
    return ok


def _try_remove(adj, e):
    u, v = e
    adj[u].discard(v)
    adj[v].discard(u)


def _greedy_from(adj, rng):
    """Add edges in random order (densest first), then local repair."""
    edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
    rng.shuffle(edges)
    edges.sort(key=lambda e: (-(len(adj[e[0]]) + len(adj[e[1]])), e))
    for e in edges:
        _try_add(adj, e)
    # local repair: try removing one edge to unlock two additions
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        rng.shuffle(edges)
        for rem in edges:
            if time.time() >= DEADLINE:
                break
            before = _edge_count(adj)
            _try_remove(adj, rem)
            added = 0
            for e in edges:
                if e == rem:
                    continue
                if _try_add(adj, e):
                    added += 1
            if _edge_count(adj) > before:
                improved = True
            else:
                # revert: re-add rem and any accidental removals are none
                adj[rem[0]].add(rem[1])
                adj[rem[1]].add(rem[0])
    return _edge_count(adj)


def _init_adj(rng):
    adj = [set() for _ in range(N)]
    style = rng.random()
    if style < 0.5:
        # random sparse
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.15:
                    adj[u].add(v); adj[v].add(u)
    elif style < 0.8:
        # path backbone
        perm = list(range(N)); rng.shuffle(perm)
        for i in range(N - 1):
            a, b = perm[i], perm[i + 1]
            adj[a].add(b); adj[b].add(a)
    else:
        # cycle backbone
        perm = list(range(N)); rng.shuffle(perm)
        for i in range(N):
            a, b = perm[i], perm[(i + 1) % N]
            adj[a].add(b); adj[b].add(a)
    return adj


def construct():
    t0 = time.time()
    best = 0
    best_edges = None
    rng = random.Random(12345)
    # deterministic strong baseline first
    adj = _init_adj(rng)
    c = _greedy_from(adj, rng)
    if c > best:
        best = c
        best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if v in adj[u]]
    # multi-restart
    while time.time() < DEADLINE:
        adj = _init_adj(rng)
        c = _greedy_from(adj, rng)
        if c > best:
            best = c
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if v in adj[u]]
            if best >= 100:
                break
    if best_edges is None:
        best_edges = []
    # exact int, list of lists
    out = []
    for e in best_edges:
        out.append([int(e[0]), int(e[1])])
    return out
# EVOLVE-BLOCK-END
