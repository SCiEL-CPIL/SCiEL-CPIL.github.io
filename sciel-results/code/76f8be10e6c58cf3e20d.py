"""Strong edge colouring: each colour class is an induced matching.
Search-based: greedy init + local improvement with simulated annealing.
"""
import time
from random import Random

_DEADLINE = time.time() + 120.0
_MARGIN = 12.0


def _build_instance(n, d, seed):
    rng = Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            continue
        if degree[u] < d and degree[v] < d:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _build_graph(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _init_coloring(edges, adj, rng):
    m = len(edges)
    color = [-1] * m
    conflict = [set() for _ in range(len(adj))]
    for i, (u, v) in enumerate(edges):
        forbidden = set()
        for x in (u, v):
            forbidden |= conflict[x]
        c = 0
        while c in forbidden:
            c += 1
        color[i] = c
        for x in (u, v):
            conflict[x].add(c)
            for y in adj[x]:
                conflict[y].add(c)
    return color


def _edge_conflict(a, b, adj):
    u, v = a
    x, y = b
    if u == x or u == y or v == x or v == y:
        return True
    if x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
        return True
    return False


def _is_valid(color, edges, adj):
    m = len(edges)
    by_color = {}
    for i in range(m):
        by_color.setdefault(color[i], []).append(i)
    for c, idxs in by_color.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if _edge_conflict(edges[idxs[a]], edges[idxs[b]], adj):
                    return False
    return True


class _State:
    __slots__ = ('color', 'conflict', 'k', 'edges', 'adj', 'm')

    def __init__(self, color, edges, adj):
        self.color = list(color)
        self.edges = edges
        self.adj = adj
        self.m = len(edges)
        self.k = max(color) + 1 if color else 0
        self.conflict = [set() for _ in range(len(adj))]
        for i, (u, v) in enumerate(edges):
            c = self.color[i]
            for x in (u, v):
                self.conflict[x].add(c)
                for y in self.adj[x]:
                    self.conflict[y].add(c)

    def assign(self, i, c):
        u, v = self.edges[i]
        old = self.color[i]
        if old == c:
            return
        for x in (u, v):
            self.conflict[x].discard(old)
            for y in self.adj[x]:
                self.conflict[y].discard(old)
        self.color[i] = c
        for x in (u, v):
            self.conflict[x].add(c)
            for y in self.adj[x]:
                self.conflict[y].add(c)
        if c + 1 > self.k:
            self.k = c + 1

    def feasible(self, i, c):
        u, v = self.edges[i]
        if c in self.conflict[u] or c in self.conflict[v]:
            return False
        return True

    def snapshot(self):
        return list(self.color)

    def restore(self, color):
        self.color = list(color)
        self.conflict = [set() for _ in range(len(self.adj))]
        self.k = 0
        for i, (u, v) in enumerate(self.edges):
            c = self.color[i]
            if c + 1 > self.k:
                self.k = c + 1
            for x in (u, v):
                self.conflict[x].add(c)
                for y in self.adj[x]:
                    self.conflict[y].add(c)


def _local_improve(state, rng, max_iter=100000):
    m = state.m
    best_k = state.k
    snap = state.snapshot()
    it = 0
    while it < max_iter and time.time() < _DEADLINE - _MARGIN:
        it += 1
        i = rng.randrange(m)
        u, v = state.edges[i]
        cur = state.color[i]
        # candidate colours 0..state.k (we allow up to current k)
        # pick a random feasible colour different from cur, or try lowering
        # Strategy: try to assign to a random colour in [0, state.k]
        c = rng.randrange(state.k)
        if c == cur:
            continue
        if not state.feasible(i, c):
            continue
        # accept if it reduces k, or with some probability
        state.assign(i, c)
        if state.k < best_k:
            best_k = state.k
            snap = state.snapshot()
    # restore best
    state.restore(snap)
    return state


def _solve_instance(n, d, seed, rng, time_budget):
    edges = _build_instance(n, d, seed)
    adj = _build_graph(edges, n)
    m = len(edges)
    if m == 0:
        return []

    color = _init_coloring(edges, adj, rng)
    state = _State(color, edges, adj)
    _local_improve(state, rng, max_iter=50000)
    best_color = state.snapshot()
    best_k = state.k

    t_end = time.time() + time_budget
    while time.time() < t_end:
        # re-greedy with random order
        order = list(range(m))
        rng.shuffle(order)
        c3 = [-1] * m
        conflict = [set() for _ in range(len(adj))]
        for i in order:
            u, v = edges[i]
            forbidden = set()
            for x in (u, v):
                forbidden |= conflict[x]
            cc = 0
            while cc in forbidden:
                cc += 1
            c3[i] = cc
            for x in (u, v):
                conflict[x].add(cc)
                for y in adj[x]:
                    conflict[y].add(cc)
        st = _State(c3, edges, adj)
        _local_improve(st, rng, max_iter=20000)
        if st.k < best_k:
            best_k = st.k
            best_color = st.snapshot()
        if time.time() >= t_end:
            break
    return best_color


def construct():
    t0 = time.time()
    rng = Random(12345)
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = _DEADLINE - _MARGIN - time.time()
        if remaining <= 0:
            # fallback: greedy only
            edges = _build_instance(*spec)
            adj = _build_graph(edges, spec[0])
            color = _init_coloring(edges, adj, rng)
            out.append(color)
            continue
        budget = max(0.5, remaining / 3.0)
        color = _solve_instance(spec[0], spec[1], spec[2], rng, budget)
        out.append(color)
    return out
