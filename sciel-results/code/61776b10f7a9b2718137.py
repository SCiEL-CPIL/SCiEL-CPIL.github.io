"""Triangle-free 4-chromatic graphs on 16 vertices: branch-and-bound over edge set.

Strategy:
- Anchor with the Grötzsch core (vertices 0..10, 20 edges, 4-chromatic, triangle-free)
  so the resulting graph is guaranteed 4-chromatic as long as the core is intact.
- Search over which edges to add involving the 5 "free" vertices (11..15) plus
  optional edges among vertices 0..10 that are not in the core, using
  branch-and-bound on the triangle-free constraint.
- A fast 3-colorability check prunes branches that are 3-colorable.
- Multiple restarts with different orderings to escape local optima.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 16
sys.setrecursionlimit(10000)


def _groetzsch_edges():
    """Grötzsch graph (Mycielski of C5): 11 vertices, 20 edges, 4-chromatic, triangle-free."""
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    orig_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in orig_edges:
        edges.append([v, 5 + u])
        edges.append([u, 5 + v])
    for i in range(5):
        edges.append([10, 5 + i])
    return edges


def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj


def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2


def _is_triangle_free(adj):
    for u in range(N):
        au = adj[u]
        while au:
            lb = au & (-au)
            v = lb.bit_length() - 1
            au ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True


def _is_3_colorable(adj):
    """Fast backtracking 3-colorability check using bitmasks and degree ordering."""
    colors = [-1] * N
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i

    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if order_pos[u] < idx:
                forbidden |= (1 << colors[u])
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)


def _is_not_3_colorable(adj):
    return not _is_3_colorable(adj)


def _is_valid(adj):
    return _is_triangle_free(adj) and _is_not_3_colorable(adj)


def _try_add_edge(adj, u, v):
    if (adj[u] >> v) & 1:
        return False
    if adj[u] & adj[v]:
        return False
    return True


def _add_edge(adj, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _branch_and_bound(adj, candidate_edges, deadline, rng):
    """
    Branch-and-bound over candidate edges.
    For each candidate edge, decide include or exclude.
    Prune if triangle-free is violated.
    Maximize edge count while maintaining triangle-freeness and 4-chromaticity.
    """
    best_cnt = _edge_count(adj)
    best_adj = [a for a in adj]

    def _search(adj_cur, candidates, idx):
        nonlocal best_cnt, best_adj

        if time.time() > deadline:
            return

        # If no more candidates, check validity
        if idx >= len(candidates):
            cnt = _edge_count(adj_cur)
            if cnt > best_cnt and _is_valid(adj_cur):
                best_cnt = cnt
                best_adj = [a for a in adj_cur]
            return

        u, v = candidates[idx]

        # Branch 1: include edge (u, v)
        if _try_add_edge(adj_cur, u, v):
            _add_edge(adj_cur, u, v)
            # Upper bound: current edges + all remaining candidates
            upper = _edge_count(adj_cur) + (len(candidates) - idx - 1)
            if upper > best_cnt:
                _search(adj_cur, candidates, idx + 1)
            _remove_edge(adj_cur, u, v)

        # Branch 2: exclude edge (u, v)
        upper_excl = _edge_count(adj_cur) + (len(candidates) - idx - 1)
        if upper_excl > best_cnt:
            _search(adj_cur, candidates, idx + 1)

    _search(adj, candidate_edges, 0)
    return best_adj, best_cnt


def _local_search(adj, deadline, rng):
    """
    Aggressive local search: greedy add + edge swaps to escape local optima.
    """
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)

    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)

    # Greedy add all possible edges
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((cur_adj[u] >> v) & 1):
                    missing.append((u, v))
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 10.0:
                break
            if _try_add_edge(cur_adj, u, v):
                _add_edge(cur_adj, u, v)
                cur_cnt += 1
                improved = True

    if cur_cnt > best_cnt and _is_valid(cur_adj):
        best_cnt = cur_cnt
        best_adj = [a for a in cur_adj]

    # Edge swaps
    improved = True
    swap_rounds = 0
    while improved and time.time() < deadline - 10.0 and swap_rounds < 80:
        improved = False
        swap_rounds += 1

        existing = [(u, v) for u in range(N) for v in range(u + 1, N)
                    if (cur_adj[u] >> v) & 1]
        if not existing:
            break

        for _ in range(30):
            if time.time() > deadline - 10.0:
                break
            rem_u, rem_v = rng.choice(existing)
            adj_copy = [a for a in cur_adj]
            _remove_edge(adj_copy, rem_u, rem_v)

            # Greedily fill
            imp = True
            while imp and time.time() < deadline - 10.0:
                imp = False
                for u in range(N):
                    for v in range(u + 1, N):
                        if time.time() > deadline - 10.0:
                            break
                        if not ((adj_copy[u] >> v) & 1) and _try_add_edge(adj_copy, u, v):
                            _add_edge(adj_copy, u, v)
                            imp = True

            cnt_copy = _edge_count(adj_copy)
            if cnt_copy > cur_cnt and _is_valid(adj_copy):
                cur_adj = adj_copy
                cur_cnt = cnt_copy
                improved = True
                if cur_cnt > best_cnt:
                    best_cnt = cur_cnt
                    best_adj = [a for a in cur_adj]

    return best_adj, best_cnt


def _direct_bnb(adj_base, deadline, rng):
    """
    Direct branch-and-bound: start from Grötzsch core, add edges involving
    vertices 11-15 and any missing edges among 0-10, maximizing total edges
    while maintaining triangle-freeness and 4-chromaticity.
    """
    # Candidate edges: all missing edges in the base graph
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj_base[u] >> v) & 1):
                candidates.append((u, v))

    # Order candidates: edges involving high-degree vertices first (more constraining)
    # Also prioritize edges that are "safe" (don't create triangles)
    def _candidate_priority(u, v):
        # Score: prefer edges that don't create triangles, and involve free vertices
        safe = 1 if _try_add_edge(adj_base, u, v) else 0
        free_score = 0
        if u >= 11 or v >= 11:
            free_score = 1
        return (safe, free_score, rng.random())

    candidates.sort(key=lambda e: _candidate_priority(*e), reverse=True)

    adj_cur = [a for a in adj_base]
    best_adj, best_cnt = _branch_and_bound(adj_cur, candidates, deadline, rng)

    # Polish with local search
    if time.time() < deadline - 5.0:
        best_adj, best_cnt = _local_search(best_adj, deadline, rng)

    return best_adj, best_cnt


def _greedy_fill(adj, deadline, rng):
    """Greedy fill: add edges in random order while maintaining triangle-freeness."""
    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((cur_adj[u] >> v) & 1):
                    missing.append((u, v))
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(cur_adj, u, v):
                _add_edge(cur_adj, u, v)
                cur_cnt += 1
                improved = True
    return cur_adj, cur_cnt


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin vs 150s kill

    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    base_cnt = _edge_count(base_adj)  # 20

    if not _is_valid(base_adj):
        out = []
        for u in range(N):
            m = base_adj[u]
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                if v > u:
                    out.append([u, v])
        return out

    best_adj = [a for a in base_adj]
    best_cnt = base_cnt

    rng = random.Random(42)

    # Phase 1: Greedy fill from base
    if time.time() < deadline - 30.0:
        adj_g, cnt_g = _greedy_fill(base_adj, deadline - 30.0, rng)
        if cnt_g > best_cnt and _is_valid(adj_g):
            best_cnt = cnt_g
            best_adj = [a for a in adj_g]

    # Phase 2: Local search from greedy result
    if time.time() < deadline - 20.0:
        adj_ls, cnt_ls = _local_search(best_adj, deadline - 20.0, rng)
        if cnt_ls > best_cnt and _is_valid(adj_ls):
            best_cnt = cnt_ls
            best_adj = [a for a in adj_ls]

    # Phase 3: Branch-and-bound with multiple orderings
    if time.time() < deadline - 15.0:
        for seed_val in range(5):
            if time.time() > deadline - 15.0:
                break
            rng2 = random.Random(1000 + seed_val)
            adj_bnb, cnt_bnb = _direct_bnb(base_adj, deadline - 15.0, rng2)
            if cnt_bnb > best_cnt and _is_valid(adj_bnb):
                best_cnt = cnt_bnb
                best_adj = [a for a in adj_bnb]

    # Phase 4: Additional local search refinements
    if time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            adj_ref, cnt_ref = _local_search(best_adj, deadline - 5.0, rng)
            if cnt_ref > best_cnt and _is_valid(adj_ref):
                best_cnt = cnt_ref
                best_adj = [a for a in adj_ref]
            else:
                break

    # Final verification
    if not _is_valid(best_adj):
        best_adj = [a for a in base_adj]
        best_cnt = base_cnt

    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])

    return out
# EVOLVE-BLOCK-END
