# EVOLVE-BLOCK-START
import time
import itertools

_START = time.time()
_LIMIT = 110.0  # safety margin under 120s wall-clock


def _deadline_ok():
    return (time.time() - _START) < _LIMIT


def _random_edges(n, p, seed):
    # Reproduce verifier's random_graph exactly: iterate u<v in order, rng.random()<p
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
    m = [0] * n
    for (u, v) in edges:
        m[u] |= 1 << v
        m[v] |= 1 << u
    return m


def _iter_bits(x):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


def _maximal_cliques_bit(adj_mask, n, min_size=3, cap=50000, deadline_ok=None):
    """Enumerate maximal cliques (as bitmasks) via Bron-Kerbosch with pivot."""
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
        ux = P | X
        best_u = -1
        best_c = -1
        for u in _iter_bits(ux):
            c = (P & adj_mask[u]).bit_count()
            if c > best_c:
                best_c = c
                best_u = u
        for v in _iter_bits(P & ~adj_mask[best_u]):
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
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    adj_mask_full = _adj_mask(n, edges)
    
    # Find maximal cliques of size >= 3
    maxcls = _maximal_cliques_bit(adj_mask_full, n, 3, 50000, _deadline_ok)
    candidates = [_mask_to_list(m, n) for m in maxcls]
    
    # Sort by size descending to minimize count
    candidates.sort(key=lambda c: -len(c))
    
    chosen = []
    covered = set()
    remaining = set(all_edges)
    
    # Greedy selection
    for c in candidates:
        if not _deadline_ok():
            break
        ec = _clique_edges(c)
        new_edges = ec - covered
        if new_edges:
            chosen.append(c)
            covered |= ec
            remaining -= new_edges
            
    # Cover remaining edges with pairs (size 2 cliques)
    for e in remaining:
        if not _deadline_ok():
            break
        chosen.append([e[0], e[1]])
        
    return chosen


def construct():
    res1 = _solve_instance(32, 0.6, 601)
    res2 = _solve_instance(64, 0.55, 602)
    return [res1, res2]
# EVOLVE-BLOCK-END
