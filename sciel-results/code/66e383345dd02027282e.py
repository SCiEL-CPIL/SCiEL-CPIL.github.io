"""C5-free graph with no independent 9-set; maximize n."""
import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 150.0
SAFETY = 8.0  # return by ~142s


def _remaining():
    return DEADLINE - (time.time() - _START) - SAFETY


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_c5(adj, n):
    """Return True if graph contains a 5-cycle (as a subgraph)."""
    # Count number of (not necessarily induced) 5-cycles via DP on paths.
    # A 5-cycle exists iff there is a set of 5 vertices with a cyclic ordering.
    # Efficient: for each pair, count common neighbors; use cycle enumeration.
    # We enumerate simple 5-cycles by fixing smallest vertex and using
    # backtracking with pruning; for n<=35 and sparse graphs this is fine.
    # Faster: count closed walks of length 5 via adjacency matrix power is heavy.
    # Use: a 5-cycle exists iff there exist distinct a,b,c,d,e with edges
    # a-b,b-c,c-d,d-e,e-a. We can fix a < b < c < d < e and check the two
    # possible "chord-free" cycle shapes? Simpler: enumerate.
    # We'll do a bounded backtracking from each start vertex.
    for start in range(n):
        # neighbors of start
        nbrs = adj[start]
        if len(nbrs) < 2:
            continue
        # path start - v1 - v2 - v3 - v4 - start, all distinct, start smallest
        for v1 in nbrs:
            if v1 < start:
                continue
            for v2 in adj[v1]:
                if v2 == start or v2 == v1:
                    continue
                if v2 < start:
                    continue
                for v3 in adj[v2]:
                    if v3 in (start, v1, v2):
                        continue
                    if v3 < start:
                        continue
                    for v4 in adj[v3]:
                        if v4 in (start, v1, v2, v3):
                            continue
                        if v4 < start:
                            continue
                        if start in adj[v4]:
                            return True
    return False


def _has_c5_fast(adj, n):
    """Faster C5 detection using pair-common-neighbor counting.
    A 5-cycle a-b-c-d-e-a exists. Count over unordered pairs.
    We use the standard: number of 5-cycles = sum over edges (u,v) of
    number of length-3 paths from u to v avoiding... complex.
    Use adjacency-matrix based: C5 count = (1/10) * (trace(A^5) - 5*trace(A^3) - 5*trace(A))
    but that counts closed walks, not simple cycles, and includes chords.
    For detection, backtracking above is acceptable for our sizes.
    """
    return _has_c5(adj, n)


def _alpha_upper(adj, n):
    """Greedy coloring of complement gives an upper bound on alpha (the
    independence number of G = clique number of complement = chromatic lower...
    Actually alpha(G) = chi(complement). Greedy coloring gives a valid
    coloring => upper bound on chi => upper bound on alpha. If upper bound
    < 9 we are safe. If >= 9 we need exact check.
    """
    # complement adjacency
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)
    # greedy coloring on complement (order by degree descending)
    order = sorted(range(n), key=lambda x: -len(comp[x]))
    color = {}
    next_color = 0
    for v in order:
        used = set()
        for u in comp[v]:
            if u in color:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
        next_color = max(next_color, c + 1)
    return next_color  # upper bound on alpha


def _alpha_exact(adj, n):
    """Exact independence number via branch and bound (complement clique)."""
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)

    best = [0]

    def expand(cand, size):
        if not cand:
            if size > best[0]:
                best[0] = size
            return
        if size + len(cand) <= best[0]:
            return
        # pick vertex with max degree in cand (for pruning)
        cand = list(cand)
        # order by degree in cand
        cand.sort(key=lambda x: -sum(1 for y in cand if y != x and y in comp[x]))
        v = cand[0]
        # branch include v
        rest = [u for u in cand[1:] if u in comp[v]]
        expand(set(rest), size + 1)
        # branch exclude v
        expand(set(cand[1:]), size)

    expand(set(range(n)), 0)
    return best[0]


def _alpha_is_at_least_9(adj, n):
    """Return True if alpha >= 9 (i.e., there IS an independent 9-set)."""
    ub = _alpha_upper(adj, n)
    if ub < 9:
        return False
    # need exact
    return _alpha_exact(adj, n) >= 9


def _is_valid(n, edges):
    adj = _build_adj(n, edges)
    if _has_c5(adj, n):
        return False
    if _alpha_is_at_least_9(adj, n):
        return False
    return True


def _seed_graph():
    # 7 disjoint K4
    n = 28
    edges = []
    for b in range(7):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return n, edges


def construct():
    best_n, best_edges = _seed_graph()

    def try_improve(n, edges):
        nonlocal best_n, best_edges
        adj = _build_adj(n, edges)
        # local search: add edges
        changed = True
        while changed and _remaining() > 5:
            changed = False
            # candidate edges: all non-edges
            cand = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        cand.append((u, v))
            random.shuffle(cand)
            for u, v in cand:
                if _remaining() < 5:
                    break
                # tentatively add
                adj[u].add(v)
                adj[v].add(u)
                ok = not _has_c5(adj, n) and not _alpha_is_at_least_9(adj, n)
                if ok:
                    edges.append([u, v])
                    changed = True
                    if n > best_n:
                        best_n, best_edges = n, list(edges)
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        # try adding a new vertex
        while _remaining() > 10 and n < 35:
            new_n = n + 1
            # try connecting new vertex to a random subset
            found = False
            for attempt in range(200):
                if _remaining() < 10:
                    break
                # choose a random neighbor set
                k = random.randint(0, n)
                nbrs = set(random.sample(range(n), k))
                new_edges = list(edges) + [[n, x] for x in nbrs]
                adj2 = _build_adj(new_n, new_edges)
                if not _has_c5(adj2, new_n) and not _alpha_is_at_least_9(adj2, new_n):
                    n, edges = new_n, new_edges
                    found = True
                    if n > best_n:
                        best_n, best_edges = n, list(edges)
                    break
            if not found:
                break

    try_improve(best_n, best_edges)

    # final verification of best
    if _is_valid(best_n, best_edges):
        return {'n': best_n, 'edges': best_edges}
    # fallback to seed
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
