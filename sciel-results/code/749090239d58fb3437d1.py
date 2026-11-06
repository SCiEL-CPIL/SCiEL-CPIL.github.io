"""Fresh design: maximal-clique partition + local merge improvement."""
# EVOLVE-BLOCK-START
import time
from random import Random

_DEADLINE = 150.0
_SAFETY = 10.0
_start = time.monotonic()


def _remaining():
    return _DEADLINE - _SAFETY - (time.monotonic() - _start)


def _instance(n, p, seed):
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _maximal_cliques(adj):
    n = len(adj)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = {v: i for i, v in enumerate(order)}
    M = []
    for v in order:
        Nv = adj[v]
        cand = [u for u in Nv if pos[u] < pos[v]]
        cand_set = set(cand)
        # Bron-Kerbosch with pivot
        def bk(P, X):
            if not P and not X:
                M.append(v)
                return
            U = P | X
            piv = max(U, key=lambda x: len(adj[x] & U))
            for u in list(P - adj[piv]):
                bk(P & adj[u], X & adj[u])
                P.remove(u)
                X.add(u)
        bk(set(cand), set())
    # convert to vertex lists
    cliques = []
    for v in M:
        Nv = adj[v]
        cand = [u for u in Nv if pos[u] < pos[v]]
        cand_set = set(cand)
        found = []

        def bk2(P, X, acc):
            if not P and not X:
                found.append(list(acc))
                return
            U = P | X
            piv = max(U, key=lambda x: len(adj[x] & U))
            for u in list(P - adj[piv]):
                acc.append(u)
                bk2(P & adj[u], X & adj[u], acc)
                acc.pop()
                P.remove(u)
                X.add(u)

        bk2(set(cand), set(), [v])
        cliques.extend(found)
    return cliques


def _clique_edges(cliq):
    return set((min(a, b), max(a, b)) for i, a in enumerate(cliq) for b in cliq[i + 1:])


def _solve(n, p, seed):
    edges = _instance(n, p, seed)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    cliques = _maximal_cliques(adj)
    # index: edge -> clique id
    edge_to_clq = {}
    for cid, c in enumerate(cliques):
        for e in _clique_edges(c):
            edge_to_clq[e] = cid

    # map vertex -> incident edges
    vert_edges = [set() for _ in range(n)]
    for e in edges:
        vert_edges[e[0]].add(e)
        vert_edges[e[1]].add(e)

    # local improvement: merge several 2-cliques into a larger clique
    improved = True
    while improved and _remaining() > 0.5:
        improved = False
        # candidate cliques from 3-vertex neighborhoods
        for v in range(n):
            if _remaining() < 0.5:
                break
            Nv = list(adj[v])
            for i in range(len(Nv)):
                for j in range(i + 1, len(Nv)):
                    a, b = Nv[i], Nv[j]
                    if b not in adj[a]:
                        continue
                    K = [v, a, b]
                    ke = _clique_edges(K)
                    # must be covered by >=3 distinct 2-cliques
                    cids = set()
                    ok = True
                    for e in ke:
                        cid = edge_to_clq.get(e)
                        if cid is None:
                            ok = False
                            break
                        cids.add(cid)
                    if not ok or len(cids) < 3:
                        continue
                    # all those cliques are 2-cliques
                    if any(len(cliques[c]) != 2 for c in cids):
                        continue
                    # try to extend K greedily
                    grow = True
                    while grow:
                        grow = False
                        for w in range(n):
                            if w in K:
                                continue
                            if all(w in adj[x] for x in K):
                                K.append(w)
                                grow = True
                                break
                    if len(K) < 3:
                        continue
                    ke = _clique_edges(K)
                    cids = set()
                    ok = True
                    for e in ke:
                        cid = edge_to_clq.get(e)
                        if cid is None:
                            ok = False
                            break
                        cids.add(cid)
                    if not ok or len(cids) < 3:
                        continue
                    if any(len(cliques[c]) != 2 for c in cids):
                        continue
                    # perform merge
                    for e in ke:
                        edge_to_clq[e] = len(cliques)
                    for c in cids:
                        pass  # mark removed
                    newid = len(cliques)
                    for c in cids:
                        cliques[c] = None
                    cliques.append(K)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break

    result = [c for c in cliques if c is not None and len(c) >= 2]
    return result


def construct():
    r1 = _solve(32, 0.6, 601)
    if _remaining() < 5.0:
        # not enough time; fall back to trivial edge partition for both
        e1 = _instance(32, 0.6, 601)
        e2 = _instance(64, 0.55, 602)
        return [[list(e) for e in e1], [list(e) for e in e2]]
    r2 = _solve(64, 0.55, 602)
    return [r1, r2]
# EVOLVE-BLOCK-END
