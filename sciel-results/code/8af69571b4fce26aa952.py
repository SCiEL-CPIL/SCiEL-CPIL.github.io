"""Fast C6-free graph maximization on 40 vertices.
Key idea: incremental C6 check that only verifies paths through a candidate edge,
plus 2-opt swaps to escape local optima.
"""
# EVOLVE-BLOCK-START
import time
import random

DEADLINE = 145.0  # safety margin before 150s kill
N = 40


def _check_edge_c6(adj, u, v):
    """Check whether adding edge (u,v) to a C6-free graph creates a C6.
    Only need to check paths of length 4 from u to v (then u-v closes the C6).
    A path u-a-b-c-d-v with all distinct and u,v not adjacent already.
    """
    # neighbors of u excluding v
    nu = adj[u]
    nv = adj[v]
    # a in N(u), d in N(v), a != d, a != v, d != u
    for a in range(N):
        if not (nu >> a) & 1:
            continue
        if a == v:
            continue
        for d in range(N):
            if d == a or d == u:
                continue
            if not (nv >> d) & 1:
                continue
            # now need b in N(a) and c in N(d) with b-c edge, b!=c, all distinct
            # b in N(a) \ {u, d}, c in N(d) \ {a, u}, b != c, b-c adjacent
            na = adj[a]
            nd = adj[d]
            b_mask = na & ~(1 << u) & ~(1 << d)
            while b_mask:
                b = (b_mask & (-b_mask)).bit_length() - 1
                b_mask &= b_mask - 1
                if b == v:
                    continue
                # c in N(d) \ {a, u, b}, and b-c adjacent
                c_mask = nd & adj[b] & ~(1 << a) & ~(1 << u) & ~(1 << b)
                if c_mask:
                    return False
    return True


def _is_c6_free(adj):
    """Full check: for every edge, verify no C6 through it."""
    for u in range(N):
        nu = adj[u]
        while nu:
            v = (nu & (-nu)).bit_length() - 1
            nu &= nu - 1
            if v <= u:
                continue
            if not _check_edge_c6(adj, u, v):
                return False
    return True


def _count_edges(adj):
    return sum(bin(x).count('1') for x in adj) // 2


def _get_edges(adj):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges


def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _random_greedy(seed=None, max_time=3.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    adj = [0] * N
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_edges)
    for u, v in all_edges:
        if time.time() - start > max_time:
            break
        _add_edge(adj, u, v)
        if not _check_edge_c6(adj, u, v):
            _remove_edge(adj, u, v)
    return adj, _count_edges(adj)


def _local_search(adj, max_time=10.0):
    """Try to add non-edges one at a time."""
    start = time.time()
    current = _count_edges(adj)
    while time.time() - start < max_time:
        improved = False
        non_edges = [(i, j) for i in range(N) for j in range(i + 1, N)
                     if not (adj[i] & (1 << j))]
        random.shuffle(non_edges)
        for u, v in non_edges:
            if time.time() - start > max_time:
                return adj, current
            _add_edge(adj, u, v)
            if _check_edge_c6(adj, u, v):
                current += 1
                improved = True
            else:
                _remove_edge(adj, u, v)
        if not improved:
            break
    return adj, current


def _two_opt_swap(adj, max_time=15.0):
    """2-opt: remove one edge, add one non-edge, keep if C6-free and same count (or better)."""
    start = time.time()
    current = _count_edges(adj)
    best_adj = adj[:]
    best_count = current
    while time.time() - start < max_time:
        improved = False
        cur_edges = [(i, j) for i in range(N) for j in range(i + 1, N)
                     if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(N) for j in range(i + 1, N)
                     if not (adj[i] & (1 << j))]
        random.shuffle(cur_edges)
        random.shuffle(non_edges)
        for ru, rv in cur_edges:
            if time.time() - start > max_time:
                break
            _remove_edge(adj, ru, rv)
            for au, av in non_edges:
                if time.time() - start > max_time:
                    break
                if adj[au] & (1 << av):
                    continue
                _add_edge(adj, au, av)
                if _check_edge_c6(adj, au, av):
                    nc = _count_edges(adj)
                    if nc > best_count:
                        best_count = nc
                        best_adj = adj[:]
                        improved = True
                    # keep same-count swap as it may help escape
                else:
                    _remove_edge(adj, au, av)
            _add_edge(adj, ru, rv)
        if not improved:
            break
    if best_count > current:
        return best_adj, best_count
    return adj, current


def _known_good():
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * N
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                adj[base + a] |= (1 << (base + b))
                adj[base + b] |= (1 << (base + a))
    return adj


def construct():
    start_time = time.time()
    best_adj = None
    best_edges = 0

    # Phase 1: Start from 8xK5
    adj = _known_good()
    best_adj = adj[:]
    best_edges = _count_edges(adj)

    # Phase 2: Local search from 8xK5
    if time.time() - start_time < DEADLINE - 30:
        adj_ls, count_ls = _local_search(adj, max_time=min(15.0, DEADLINE - (time.time() - start_time) - 15))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls

    # Phase 3: Multiple random greedy restarts
    for attempt in range(40):
        remaining = DEADLINE - (time.time() - start_time) - 10
        if remaining < 2:
            break
        mt = min(2.0, remaining / 3)
        adj_g, count_g = _random_greedy(seed=100 + attempt, max_time=mt)
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
        if count_g >= best_edges - 2 and time.time() - start_time < DEADLINE - 10:
            adj_l, count_l = _local_search(adj_g, max_time=min(3.0, DEADLINE - (time.time() - start_time) - 5))
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]

    # Phase 4: 2-opt swap search on best
    if time.time() - start_time < DEADLINE - 5:
        adj_sw, count_sw = _two_opt_swap(best_adj, max_time=min(12.0, DEADLINE - (time.time() - start_time) - 2))
        if count_sw > best_edges:
            best_edges = count_sw
            best_adj = adj_sw

    # Phase 5: Final local search
    if time.time() - start_time < DEADLINE - 2:
        adj_f, count_f = _local_search(best_adj, max_time=min(5.0, DEADLINE - (time.time() - start_time) - 1))
        if count_f > best_edges:
            best_edges = count_f
            best_adj = adj_f

    if best_adj is None:
        best_adj = _known_good()

    if not _is_c6_free(best_adj):
        best_adj = _known_good()

    return _get_edges(best_adj)
# EVOLVE-BLOCK-END
