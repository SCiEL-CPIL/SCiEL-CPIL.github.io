"""Construct a large C4-free graph with no independent 11-set (R(C4,K11) > n)."""
import time
import itertools

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin well under the 150 s wall-clock kill

# Prime p with p ≡ 3 (mod 4).  The Paley graph on p vertices (edges join
# quadratic residues) is C4-free and has independence number (p-1)/2.
_P = 43
_QR = set()
for a in range(1, _P):
    if pow(a, (_P - 1) // 2, _P) == 1:
        _QR.add(a)


def _paley_edges(p):
    edges = []
    for u in range(p):
        for v in range(u + 1, p):
            if (v - u) % p in _QR or (u - v) % p in _QR:
                edges.append([u, v])
    return edges


def _check_c4(adj, n):
    """Return True if any pair of vertices has two common neighbours."""
    for u in range(n):
        su = adj[u]
        for v in range(u + 1, n):
            if len(su & adj[v]) >= 2:
                return True
    return False


def _check_no_ind11(adj, n):
    """Return True if there is NO independent set of size 11."""
    for comb in itertools.combinations(range(n), 11):
        ok = True
        for i in range(11):
            u = comb[i]
            for j in range(i + 1, 11):
                if comb[j] in adj[u]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False  # found an independent 11-set
    return True


def _valid(adj, n):
    return _check_no_ind11(adj, n) and not _check_c4(adj, n)


def construct():
    # Baseline: the Paley graph on 43 vertices.
    edges = _paley_edges(_P)
    n = _P
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    best = (n, list(edges))
    if _valid(adj, n):
        # n = 43 is the maximum allowed; return immediately.
        return {'n': n, 'edges': list(edges)}

    # Fallback: try to grow a smaller C4-free graph by adding vertices.
    # (Defensive path; the Paley-43 baseline should already be valid.)
    for start_n in range(40, 28, -1):
        if time.time() > _DEADLINE:
            break
        edges = _paley_edges(_P)[:0]  # empty
        # build a random-free deterministic base: first start_n Paley vertices
        base = _paley_edges(_P)
        base = [e for e in base if e[0] < start_n and e[1] < start_n]
        adj = [set() for _ in range(start_n)]
        for u, v in base:
            adj[u].add(v)
            adj[v].add(u)
        cur_edges = list(base)
        if not _check_c4(adj, start_n) and _check_no_ind11(adj, start_n):
            if start_n > best[0]:
                best = (start_n, cur_edges)
        # attempt to add vertices start_n..42 one by one
        for nv in range(start_n, 43):
            if time.time() > _DEADLINE:
                break
            # choose the vertex with the most edges among C4-safe choices
            best_nv = None
            best_cnt = -1
            for cand_edges in _candidate_edges(adj, nv, start_n):
                if time.time() > _DEADLINE:
                    break
                if cand_edges is None:
                    continue
                # test C4
                adj2 = [set(s) for s in adj]
                adj2.append(set())
                for u, v in cand_edges:
                    adj2[u].add(v)
                    adj2[v].add(u)
                if _check_c4(adj2, nv + 1):
                    continue
                # test independence
                if not _check_no_ind11(adj2, nv + 1):
                    continue
                if len(cand_edges) > best_cnt:
                    best_cnt = len(cand_edges)
                    best_nv = cand_edges
            if best_nv is None:
                break
            adj.append(set())
            for u, v in best_nv:
                adj[u].add(v)
                adj[v].add(u)
            cur_edges.extend(best_nv)
            if nv + 1 > best[0]:
                best = (nv + 1, list(cur_edges))
    return {'n': best[0], 'edges': best[1]}


def _candidate_edges(adj, nv, n):
    """Yield C4-safe candidate edge-sets for a new vertex nv."""
    # Try all subsets of existing vertices of size 0..n, but prune by C4.
    # Efficient approach: pick neighbours such that no two share a neighbour.
    # We simply try a few deterministic density levels.
    for density in range(0, n + 1):
        # choose `density` neighbours greedily (C4-safe)
        chosen = []
        for v in range(n):
            if len(chosen) >= density:
                break
            # v is C4-safe with all chosen if it shares no neighbour with any chosen
            safe = True
            for c in chosen:
                if adj[v] & adj[c]:
                    safe = False
                    break
            if safe:
                chosen.append(v)
        if len(chosen) == density:
            yield [[nv, v] for v in chosen]
        else:
            break


# Self-test guard: ensure the module is importable without side effects.
if __name__ == '__main__':
    r = construct()
    print('n =', r['n'], 'edges =', len(r['edges']))
