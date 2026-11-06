"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

N = 18
DEADLINE = 110.0  # safety margin before 120s wall clock / 150s kill


def _init_time():
    return time.time()


def _remaining(start):
    return DEADLINE - (time.time() - start)


def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edges_to_set(edges):
    s = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        s.add((u, v))
    return s


def _is_k4_free(adj):
    # K4 exists iff some edge's common neighborhood contains an edge
    for u in range(N):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                for a in common:
                    if adj[a] & common:
                        return False
    return True


def _has_triangle(adj, s):
    for u in s:
        nbrs = adj[u] & s
        for v in nbrs:
            if v > u and (adj[u] & adj[v] & s):
                return True
    return False


def _all_triangles(adj):
    tris = []
    for u in range(N):
        for v in adj[u]:
            if v > u:
                for w in adj[u] & adj[v]:
                    if w > v:
                        tris.append((u, v, w))
    return tris


def _largest_triangle_free(adj):
    # exact: largest s such that some s-subset is triangle-free
    tris = _all_triangles(adj)
    # triangle-free induced set == set that hits NO triangle == independent in hypergraph
    # We want max size of a set containing no triangle.
    # Complement: min hitting set of triangles.
    # max triangle-free = N - (min # vertices hitting all triangles)
    # But hitting set over triangle hyperedges; use branch & bound.
    if not tris:
        return N
    # Find max independent set in the 3-uniform hypergraph (triangles are forbidden).
    # Equivalently, max set with no triangle. Use branch & bound over vertices.
    best = [0]
    # order vertices by triangle-degree descending
    tdeg = [0] * N
    for (a, b, c) in tris:
        tdeg[a] += 1
        tdeg[b] += 1
        tdeg[c] += 1
    order = sorted(range(N), key=lambda x: -tdeg[x])

    def dfs(i, chosen, forbidden):
        # i: index in order; chosen count; forbidden set of vertices that form a triangle with chosen
        if i == N:
            if chosen > best[0]:
                best[0] = chosen
            return
        # upper bound: chosen + remaining
        if chosen + (N - i) <= best[0]:
            return
        v = order[i]
        if v in forbidden:
            dfs(i + 1, chosen, forbidden)
        else:
            # include v: new forbidden = forbidden + vertices that with v make a triangle with another chosen? 
            # Actually including v forbids any w such that {v,w}+some chosen x forms triangle.
            new_forbidden = set(forbidden)
            # For each chosen x, if (v,x,w) triangle then w forbidden
            # Precompute: for pair (v,x) the set of w making triangle
            for x in chosen_set:
                new_forbidden |= tri_with_pair.get((v, x), set())
            # also v itself can't be re-added (handled by order index)
            chosen_set.add(v)
            dfs(i + 1, chosen + 1, new_forbidden)
            chosen_set.remove(v)
            # exclude v
            dfs(i + 1, chosen, forbidden)

    # precompute tri_with_pair
    tri_with_pair = {}
    for (a, b, c) in tris:
        for p, q in ((a, b), (a, c), (b, a), (b, c), (c, a), (c, b)):
            tri_with_pair.setdefault((p, q), set()).add(c)

    chosen_set = set()
    dfs(0, 0, set())
    return best[0]


def _largest_triangle_free_fast(adj):
    # Faster: iterate all 2^18 subsets is 262k, check triangle-free via precomputed triangle mask.
    # Build for each vertex a bitmask of triangles it belongs to.
    tris = _all_triangles(adj)
    if not tris:
        return N
    tri_masks = [0] * N
    for idx, (a, b, c) in enumerate(tris):
        m = 1 << idx
        tri_masks[a] |= m
        tri_masks[b] |= m
        tri_masks[c] |= m
    # For a subset S (bitmask), it's triangle-free iff for every triangle, not all 3 in S.
    # Precompute: a subset is triangle-free iff AND over its vertices of ~tri_masks[v] covers all? 
    # Simpler: subset S hits triangle t iff all of t in S.
    # We want max popcount S with no triangle fully contained.
    # Use: for S, triangle-free iff (union over v in S of tri_masks[v]) has no triangle t where t subset S.
    # Efficient: iterate S by increasing size? Instead iterate all S and test.
    # Test S triangle-free: for each triangle t, if (S & t_mask) == t_mask -> not free.
    # Precompute t_mask list.
    t_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in tris]
    best = 0
    # iterate all subsets
    full = 1 << N
    for S in range(full):
        # quick popcount bound
        pc = bin(S).count('1')
        if pc <= best:
            continue
        ok = True
        for tm in t_masks:
            if (S & tm) == tm:
                ok = False
                break
        if ok:
            best = pc
    return best


def construct():
    start = _init_time()
    rng = random.Random(12345)

    best_edges = None
    best_score = None

    def eval_edges(edges):
        adj = _build_adj(edges)
        if not _is_k4_free(adj):
            return None
        return _largest_triangle_free_fast(adj)

    # Initial: complete tripartite K6,6,6 (known valid, score 12)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    init_edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    init_edges.append([a, b])
    s = eval_edges(init_edges)
    assert s is not None
    best_edges = [list(e) for e in init_edges]
    best_score = s

    # Local search: try adding/removing edges to reduce score, keep K4-free.
    # Represent edge set.
    def local_search(edges, deadline):
        edge_set = _edges_to_set(edges)
        adj = _build_adj(edge_set)
        cur_score = _largest_triangle_free_fast(adj)
        cur_edges = list(edge_set)
        improved = True
        while improved and _remaining(start) > 5.0:
            improved = False
            # candidate moves
            moves = []
            # removals
            for e in cur_edges:
                moves.append((0, e))
            # additions
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) not in edge_set:
                        moves.append((1, (u, v)))
            rng.shuffle(moves)
            for typ, e in moves:
                if _remaining(start) < 5.0:
                    break
                u, v = e
                # apply
                if typ == 1:
                    edge_set.add((u, v))
                    adj[u].add(v); adj[v].add(u)
                else:
                    edge_set.discard((u, v))
                    adj[u].discard(v); adj[v].discard(u)
                if not _is_k4_free(adj):
                    # revert
                    if typ == 1:
                        edge_set.discard((u, v)); adj[u].discard(v); adj[v].discard(u)
                    else:
                        edge_set.add((u, v)); adj[u].add(v); adj[v].add(u)
                    continue
                ns = _largest_triangle_free_fast(adj)
                if ns < cur_score:
                    cur_score = ns
                    cur_edges = list(edge_set)
                    improved = True
                    if ns < best_score:
                        best_score = ns
                        best_edges = [list(x) for x in cur_edges]
                else:
                    # revert
                    if typ == 1:
                        edge_set.discard((u, v)); adj[u].discard(v); adj[v].discard(u)
                    else:
                        edge_set.add((u, v)); adj[u].add(v); adj[v].add(u)
        return cur_edges, cur_score

    # First local search from initial
    cur_edges, cur_score = local_search(best_edges, DEADLINE)

    # Multiple restarts with random K4-free seeds
    while _remaining(start) > 20.0:
        # random seed: random graph then remove edges to make K4-free
        seed_edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.35:
                    seed_edges.append((u, v))
        # make K4-free by removing edges
        adj = _build_adj(seed_edges)
        changed = True
        while changed and _remaining(start) > 15.0:
            changed = False
            for u in range(N):
                for v in list(adj[u]):
                    if v > u:
                        common = adj[u] & adj[v]
                        for a in list(common):
                            if adj[a] & common:
                                # K4 contains {u,v,a,?}
                                adj[u].discard(v); adj[v].discard(u)
                                seed_edges = [(x, y) for (x, y) in seed_edges if not ((x, y) == (u, v) or (x, y) == (v, u))]
                                changed = True
                                break
                            if changed:
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        if not _is_k4_free(adj):
            continue
        s = _largest_triangle_free_fast(adj)
        if s is not None and s < best_score:
            best_score = s
            best_edges = [list(x) for x in seed_edges]
        # local search from this seed
        cur_edges, cur_score = local_search(seed_edges, DEADLINE)

    return [list(e) for e in best_edges]
