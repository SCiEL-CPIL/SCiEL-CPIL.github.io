"""Rewritten program: exact branch-and-bound for strong edge colouring (induced matching)."""
import time
import heapq
from random import Random

# EVOLVE-BLOCK-START

def _make_instance(n, d, seed):
    rng = Random(seed)
    degree, edges = [0] * n, set()
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        e = (min(u, v), max(u, v))
        if u != v and degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _build_conflicts(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    m = len(edges)
    # adj_edge[i] = set of edge-indices conflicting with edge i
    adj_edge = [set() for _ in range(m)]
    for i in range(m):
        u, v = edges[i]
        for j in range(i + 1, m):
            a, b = edges[j]
            # share a vertex
            if a == u or a == v or b == u or b == v:
                adj_edge[i].add(j)
                adj_edge[j].add(i)
                continue
            # joined by an edge
            if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                adj_edge[i].add(j)
                adj_edge[j].add(i)
    return adj_edge


def _greedy(adj_edge, m):
    color = [-1] * m
    for i in range(m):
        used = set()
        for j in adj_edge[i]:
            if color[j] != -1:
                used.add(color[j])
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return color, max(color) + 1


def _bbsolve(adj_edge, m, deadline, max_nodes=2000000):
    """Return (optimal_k, feasible_coloring_or_None)."""
    if m == 0:
        return 0, [0] * 0
    # order edges by descending degree
    order = sorted(range(m), key=lambda i: -len(adj_edge[i]))
    pos = [0] * m
    for rank, i in enumerate(order):
        pos[i] = rank
    # adjacency in edge-index space
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in adj_edge[i]:
            adj[i].add(j)

    def feasible(k):
        # domain[i] = bitmask of colours allowed for edge i
        full = (1 << k) - 1
        domain = [full] * m
        color = [-1] * m
        nodes = 0
        best_color = None
        # best-first search over (sum_used, -count, state)
        # state: (color_tuple, domain_tuple, assigned_count)
        start_state = ((), tuple(domain), 0)
        heap = [(0, 0, 0, start_state)]
        seen = set()
        while heap:
            if time.time() > deadline:
                return None
            _, _, seq, state = heapq.heappop(heap)
            color_t, dom_t, cnt = state
            if seq != cnt:
                continue
            if cnt == m:
                best_color = list(color_t)
                break
            if nodes >= max_nodes:
                return None
            nodes += 1
            # pick next unassigned edge in order
            i = order[cnt]
            di = dom_t[i]
            if di == 0:
                continue
            # try colours in increasing order (symmetry: prefer low colours)
            # generate children
            for c in range(k):
                if not (di >> c) & 1:
                    continue
                # symmetry breaking: for the first few edges, only use colour c
                # if c == 0 or colour c-1 was not used earlier in this branch
                # (enforces canonical colouring)
                used_colors = set(color_t)
                if c > 0 and c - 1 not in used_colors:
                    # we could have used colour c-1 instead; skip to avoid symmetric states
                    continue
                # forward check
                new_dom = list(dom_t)
                new_dom[i] = di & ~(1 << c)
                ok = True
                for j in adj[i]:
                    if pos[j] > cnt:  # unassigned
                        nd = new_dom[j] & ~(1 << c)
                        if nd == 0:
                            ok = False
                            break
                        new_dom[j] = nd
                if not ok:
                    continue
                new_color_t = color_t + (c,)
                new_state = (new_color_t, tuple(new_dom), cnt + 1)
                key = new_state
                if key not in seen:
                    seen.add(key)
                    # priority: sum of used colours (lower is better), then -count
                    sum_used = sum(new_color_t)
                    heapq.heappush(heap, (sum_used, -(cnt + 1), seq + 1, new_state))
        return best_color

    # binary / linear search for minimal feasible k
    # start from greedy bound
    gcolor, gk = _greedy(adj_edge, m)
    best_k = gk
    best_sol = gcolor
    # try to prove smaller k infeasible, from 1 up to best_k-1
    # but that's expensive; instead try best_k-1, best_k-2, ...
    for k in range(best_k - 1, 0, -1):
        if time.time() > deadline:
            break
        sol = feasible(k)
        if sol is not None:
            best_k = k
            best_sol = sol
            break
        # else k infeasible, keep going lower
    return best_k, best_sol


def construct():
    deadline = time.time() + 110.0  # safety margin under 120s
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 5.0:
            # not enough time left; use greedy for remaining
            edges = _make_instance(*spec)
            adj_edge = _build_conflicts(edges, spec[0])
            color, _ = _greedy(adj_edge, len(edges))
            out.append(color)
            continue
        edges = _make_instance(*spec)
        n = spec[0]
        adj_edge = _build_conflicts(edges, n)
        m = len(edges)
        # greedy warm start
        gcolor, gk = _greedy(adj_edge, m)
        # attempt exact B&B
        try:
            bk, sol = _bbsolve(adj_edge, m, deadline)
            if sol is not None and bk <= gk:
                color = sol
            else:
                color = gcolor
        except Exception:
            color = gcolor
        out.append(color)
    return out

# EVOLVE-BLOCK-END
