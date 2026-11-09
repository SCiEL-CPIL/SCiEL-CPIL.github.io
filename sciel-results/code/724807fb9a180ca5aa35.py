"""Greedy clique partitioning with Bron-Kerbosch on residual graph."""
# EVOLVE-BLOCK-START
import time
import itertools

_START = time.time()
_LIMIT = 115.0  # safety margin under 120s wall-clock


def _deadline_ok():
    return (time.time() - _START) < _LIMIT


def _random_edges(n, p, seed):
    """Reproduce verifier's random_graph exactly: iterate u<v in order, rng.random()<p"""
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _adj_mask(n, edges):
    """Build adjacency bitmasks from a list/set of edges."""
    m = [0] * n
    for (u, v) in edges:
        m[u] |= 1 << v
        m[v] |= 1 << u
    return m


def _iter_bits(x):
    """Yield indices of set bits in a bitmask."""
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


def _maximal_cliques_bit(adj_mask, n, min_size=2, cap=100000, deadline_ok=None):
    """
    Enumerate maximal cliques (as bitmasks) via Bron-Kerbosch with pivot.
    Only returns cliques with size >= min_size.
    """
    result = []
    aborted = [False]

    def bronk(R, P, X):
        if aborted[0]:
            return
        if deadline_ok is not None and not deadline_ok():
            aborted[0] = True
            return
        if not P and not X:
            if R.bit_count() >= min_size:
                result.append(R)
                if len(result) >= cap:
                    aborted[0] = True
            return
        if not P:
            return

        # Pivot selection
        ux = P | X
        best_u = -1
        best_c = -1
        for u in _iter_bits(ux):
            c = (P & adj_mask[u]).bit_count()
            if c > best_c:
                best_c = c
                best_u = u

        # Recurse on P \ N(best_u)
        if best_u >= 0:
            P_sub = P & ~adj_mask[best_u]
        else:
            P_sub = P

        for v in _iter_bits(P_sub):
            bit = 1 << v
            bronk(R | bit, P & adj_mask[v], X & adj_mask[v])
            if aborted[0]:
                return
            P &= ~bit
            X |= bit

    bronk(0, (1 << n) - 1, 0)
    return result


def _mask_to_list(mask, n):
    return [v for v in range(n) if (mask >> v) & 1]


def _clique_edges(clique):
    return frozenset(_edge_key(u, v) for u, v in itertools.combinations(clique, 2))


def _solve_instance(n, p, seed):
    """Solve one instance: partition edges into minimum number of cliques."""
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    chosen = []
    remaining = set(all_edges)

    while remaining and _deadline_ok():
        # Build adjacency for remaining edges
        rem_mask = _adj_mask(n, remaining)

        # Find all maximal cliques in the remaining graph
        maxcls = _maximal_cliques_bit(rem_mask, n, min_size=2, cap=100000, deadline_ok=_deadline_ok)

        if not maxcls:
            break

        # Convert to lists and filter to valid cliques (all edges in remaining)
        candidates = []
        for m in maxcls:
            c = _mask_to_list(m, n)
            if len(c) < 2:
                continue
            ec = _clique_edges(c)
            if ec <= remaining:
                candidates.append((c, ec))

        if not candidates:
            break

        # Select clique covering most edges; tie-break by larger size
        best_c, best_ec = max(candidates, key=lambda x: (len(x[1]), len(x[0])))

        chosen.append(best_c)
        remaining -= best_ec

    # Fallback: leftover edges as single-edge cliques
    for e in remaining:
        chosen.append([e[0], e[1]])

    return chosen


def construct():
    """
    Returns [cliques_for_instance1, cliques_for_instance2]
    where each instance is a list of cliques (vertex lists).
    """
    res1 = _solve_instance(32, 0.6, 601)
    if not _deadline_ok():
        # Emergency fallback: all edges as single cliques
        edges1 = _random_edges(32, 0.6, 601)
        res1 = [[u, v] for u, v in edges1]

    res2 = _solve_instance(64, 0.55, 602)
    if not _deadline_ok():
        edges2 = _random_edges(64, 0.55, 602)
        res2 = [[u, v] for u, v in edges2]

    return [res1, res2]
# EVOLVE-BLOCK-END
