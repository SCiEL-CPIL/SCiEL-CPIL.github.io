"""Clique partitioning: batched maximal-clique packing with density-aware selection."""
# EVOLVE-BLOCK-START
import time
import itertools

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock


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


def _maximal_cliques_bit(adj_mask, n, min_size=3, cap=200000, deadline_ok=None):
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
        if not P:
            if not X and R.bit_count() >= min_size:
                result.append(R)
                if len(result) >= cap:
                    aborted[0] = True
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


def _clique_density(clique, adj_mask, n):
    """
    Density-aware score: for each edge in the clique, count common neighbors
    within the clique. Higher = more 'tight' clique.
    """
    if len(clique) < 2:
        return 0
    # For a clique of size k, every pair has (k-2) common neighbors within the clique
    # But we want edges that are in dense regions of the ORIGINAL graph
    # Score = sum over edges of (number of vertices in clique adjacent to both endpoints in original graph)
    # For a clique, all other clique vertices are adjacent to both endpoints, so each edge has (k-2) common neighbors in clique
    # But we also care about the original graph density around these edges
    # Simple: k^2 * (k-2) / k = k*(k-2), but let's use a normalized version
    k = len(clique)
    return k * (k - 2)


def _solve_instance(n, p, seed):
    """
    Partition edges into minimum number of cliques.
    Strategy: Each round, find all maximal cliques (size>=3) in residual graph.
    Greedily select a maximal set of edge-disjoint cliques, prioritizing by
    density score (larger cliques first), then remove them all at once.
    This is a batched approach vs one-clique-per-iteration.
    """
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    chosen = []
    remaining = set(all_edges)

    while remaining and _deadline_ok():
        # Build adjacency for remaining edges
        rem_mask = _adj_mask(n, remaining)

        # Find all maximal cliques of size >= 3 in the remaining graph
        maxcls = _maximal_cliques_bit(rem_mask, n, min_size=3, cap=200000, deadline_ok=_deadline_ok)

        if not maxcls:
            # No cliques of size >= 3; try size 2
            maxcls = _maximal_cliques_bit(rem_mask, n, min_size=2, cap=200000, deadline_ok=_deadline_ok)

        if not maxcls:
            break

        # Convert to lists and compute edge sets
        candidates = []
        for m in maxcls:
            c = _mask_to_list(m, n)
            if len(c) < 2:
                continue
            ec = _clique_edges(c)
            if ec <= remaining:
                density = _clique_density(c, rem_mask, n)
                candidates.append((c, ec, density))

        if not candidates:
            break

        # Sort by density descending, then by edge count descending
        candidates.sort(key=lambda x: (-x[2], -len(x[1])))

        # Greedily select edge-disjoint cliques
        selected_edges = set()
        for c, ec, density in candidates:
            if not _deadline_ok():
                break
            if ec & selected_edges:
                continue
            # All edges must still be in remaining
            if not ec <= remaining:
                continue
            selected_edges |= ec
            chosen.append(c)

        if not selected_edges:
            # Fallback: pick the best single clique
            c, ec, density = candidates[0]
            chosen.append(c)
            selected_edges = ec

        remaining -= selected_edges

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
