import time
import random

# EVOLVE-BLOCK-START
_N = 15
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill


def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)


def _build_adj_deg(edges):
    """Build adjacency lists and degree array from a set of edges."""
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg


def _is_valid_fast(adj, deg, edges):
    """
    Correctly check if the graph is valid: no path s-u-v-t with deg[s] == deg[t].
    For each edge (u,v), for each neighbor s of u (s != v), for each neighbor t of v (t != u, t != s),
    check deg[s] != deg[t].
    """
    for u, v in edges:
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        for s in neighbors_u:
            if s == v:
                continue
            ds = deg[s]
            for t in neighbors_v:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True


def _count_violations(adj, deg, edges):
    """Count the number of violating paths s-u-v-t where deg[s] == deg[t]."""
    count = 0
    for u, v in edges:
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        for s in neighbors_u:
            if s == v:
                continue
            ds = deg[s]
            for t in neighbors_v:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    count += 1
    return count


def _find_violation_edges(adj, deg, edges):
    """
    Find one violation and return the three candidate edges on the path.
    Returns (candidate_edges, violation_count_after) or (None, None) if valid.
    """
    for u, v in edges:
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        for s in neighbors_u:
            if s == v:
                continue
            ds = deg[s]
            for t in neighbors_v:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return [
                        _norm((s, u)),
                        _norm((u, v)),
                        _norm((v, t)),
                    ]
    return None


def _try_add_edge(g, adj, deg, e):
    """Try to add edge e to graph g. Return (new_g, new_adj, new_deg) if valid, else None."""
    u, v = e
    if u == v:
        return None
    if e in g:
        return None
    new_g = g | {e}
    new_adj, new_deg = _build_adj_deg(new_g)
    if _is_valid_fast(new_adj, new_deg, new_g):
        return new_g, new_adj, new_deg
    return None


def _greedy_remove(g, adj, deg, rng, deadline):
    """
    Phase 1: Remove edges until valid.
    At each step, find a violation, consider the 3 edges on that path,
    pick the one whose removal minimizes total violation count.
    """
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        if _is_valid_fast(adj, deg, g):
            break

        candidates = _find_violation_edges(adj, deg, g)
        if candidates is None:
            break

        rng.shuffle(candidates)

        best_edge = None
        best_vcount = float('inf')

        for e in candidates:
            if e not in g:
                continue
            new_g = g - {e}
            new_adj, new_deg = _build_adj_deg(new_g)
            vcount = _count_violations(new_adj, new_deg, new_g)
            if vcount < best_vcount:
                best_vcount = vcount
                best_edge = e

        if best_edge is None:
            # Fallback: remove first candidate that exists
            for e in candidates:
                if e in g:
                    best_edge = e
                    break
            if best_edge is None:
                break

        g.discard(best_edge)
        adj, deg = _build_adj_deg(g)

    return g, adj, deg


def _readd_edges(g, adj, deg, rng, deadline):
    """
    Phase 2: Try to re-add missing edges.
    Try in random order, and also do a few passes to catch edges that
    become addable after other additions.
    """
    improved = True
    max_passes = 5
    passes = 0
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        missing = [e for e in _ALL_EDGES if e not in g]
        rng.shuffle(missing)
        for e in missing:
            if time.time() > deadline:
                break
            res = _try_add_edge(g, adj, deg, e)
            if res:
                g, adj, deg = res
                improved = True
    return g, adj, deg


def _local_search_swap(g, adj, deg, rng, deadline):
    """
    Phase 3: Try to improve via swap: remove one edge, then greedily add as many as possible.
    Keep the swap only if it results in more edges.
    """
    if len(g) >= _N * (_N - 1) // 2:
        return g, adj, deg

    current_edges = list(g)
    rng.shuffle(current_edges)

    best_g, best_adj, best_deg = g, adj, deg
    best_size = len(g)

    # Try a limited number of random removals
    for e_out in current_edges[:10]:
        if time.time() > deadline:
            break

        g_temp = g - {e_out}
        adj_temp, deg_temp = _build_adj_deg(g_temp)

        # Greedily add edges
        missing_temp = [e for e in _ALL_EDGES if e not in g_temp]
        rng.shuffle(missing_temp)

        for e_in in missing_temp:
            if time.time() > deadline:
                break
            res = _try_add_edge(g_temp, adj_temp, deg_temp, e_in)
            if res:
                g_temp, adj_temp, deg_temp = res

        if len(g_temp) > best_size:
            best_g, best_adj, best_deg = g_temp, adj_temp, deg_temp
            best_size = len(g_temp)

    return best_g, best_adj, best_deg


def _solve_once(rng, deadline):
    """
    Perform one run of the search.
    1. Start with complete graph.
    2. Greedily remove violating edges.
    3. Re-add edges.
    4. Local search swap.
    """
    g = set(_ALL_EDGES)
    adj, deg = _build_adj_deg(g)

    # Phase 1: Remove edges until valid
    g, adj, deg = _greedy_remove(g, adj, deg, rng, deadline)

    # Phase 2: Re-add edges
    g, adj, deg = _readd_edges(g, adj, deg, rng, deadline)

    # Phase 3: Local search swap
    g, adj, deg = _local_search_swap(g, adj, deg, rng, deadline)

    return g


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_g = None
    best_size = -1

    restart_count = 0
    base_seed = 42

    while time.time() < deadline:
        rng = random.Random(base_seed + restart_count)

        g = _solve_once(rng, deadline)

        if g:
            adj, deg = _build_adj_deg(g)
            if _is_valid_fast(adj, deg, g):
                if len(g) > best_size:
                    best_size = len(g)
                    best_g = g

        restart_count += 1

        # Limit restarts to avoid infinite loop if one restart takes too long
        if restart_count > 100:
            break

    if best_g is None:
        # Fallback: Star graph (always valid: no 4-vertex simple path exists)
        best_g = {(0, i) for i in range(1, _N)}

    # Convert to required format
    result = []
    for u, v in best_g:
        result.append([int(u), int(v)])

    return result
# EVOLVE-BLOCK-END
