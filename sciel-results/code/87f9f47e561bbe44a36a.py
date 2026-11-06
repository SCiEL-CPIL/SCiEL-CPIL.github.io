import time
import random
import sys

sys.setrecursionlimit(1000000)

_EDGE_SET = set()


def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_once(adj, edge_set, rng):
    remaining = set(edge_set)
    pieces = []
    while remaining:
        cand = list(remaining)
        cand.sort(key=lambda e: (-min(len(adj.get(e[0], ())), len(adj.get(e[1], ()))), rng.random()))
        found = None
        for e in cand[:10]:
            if e not in remaining:
                continue
            u, v = e
            path = [(u, v)]
            cur = v
            cur_set = {u, v}
            while True:
                if len(path) >= 2 and u in adj.get(cur, ()):
                    ce = (min(cur, u), max(cur, u))
                    if ce in edge_set and ce not in (set(x for pe in path for x in pe) and {(min(a,b),max(a,b)) for a,b in path}):
                        if ce not in { (min(pe[0],pe[1]),max(pe[0],pe[1])) for pe in path }:
                            found = list(path)
                            break
                nbrs = [x for x in adj.get(cur, ()) if x not in cur_set and (min(cur, x), max(cur, x)) in edge_set and (min(cur, x), max(cur, x)) not in {(min(pe[0],pe[1]),max(pe[0],pe[1])) for pe in path}]
                if not nbrs:
                    break
                nbrs.sort(key=lambda x: (-len(adj.get(x, ())), rng.random()))
                nxt = nbrs[0]
                path.append((cur, nxt))
                cur = nxt
                cur_set.add(cur)
            if found is not None:
                break
        if found is None:
            e = cand[0]
            pieces.append([e[0], e[1]])
            remaining.discard(e)
        else:
            verts = [found[0][0]]
            for pe in found:
                verts.append(pe[1])
            pieces.append(verts)
            for pe in found:
                remaining.discard((min(pe[0], pe[1]), max(pe[0], pe[1])))
            a, b = verts[0], verts[-1]
            remaining.discard((min(a, b), max(a, b)))
    return pieces


def _two_opt(pieces, adj, edge_set, rng, iters=40):
    improved = True
    guard = 0
    while improved and guard < iters:
        improved = False
        guard += 1
        for i in range(len(pieces)):
            pi = pieces[i]
            if len(pi) < 3:
                continue
            for j in range(i + 1, len(pieces)):
                pj = pieces[j]
                if len(pj) < 3:
                    continue
                si = set(pi)
                common = [v for v in pj if v in si]
                if not common:
                    continue
                v = common[0]
                idx_i = pi.index(v)
                idx_j = pj.index(v)
                pi_a = [pi[(idx_i + k) % len(pi)] for k in range(len(pi))]
                pj_b = [pj[(idx_j - k) % len(pj)] for k in range(len(pj))]
                cand = pi_a + pj_b[1:][::-1]
                ok = True
                seen = set()
                for k in range(len(cand)):
                    a = cand[k]
                    b = cand[(k + 1) % len(cand)]
                    if a == b:
                        ok = False
                        break
                    e = (min(a, b), max(a, b))
                    if e not in edge_set or e in seen:
                        ok = False
                        break
                    seen.add(e)
                if not ok or len(set(cand)) != len(cand) or len(cand) < 3:
                    continue
                pieces[i] = cand
                pieces.pop(j)
                improved = True
                break
            if improved:
                break
    return pieces


def construct():
    deadline = time.time() + 140.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = {}
    for (n, p, seed) in specs:
        if time.time() > deadline - 15:
            break
        es = instance(n, p, seed)
        edge_set = set(es)
        adj = {i: set() for i in range(n)}
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        rng = random.Random(seed * 1000 + 7)
        best = None
        t0 = time.time()
        per_graph_deadline = min(deadline - 15, t0 + 40.0)
        while time.time() < per_graph_deadline:
            pieces = _decompose_once(adj, edge_set, rng)
            if time.time() < per_graph_deadline - 1:
                pieces = _two_opt(pieces, adj, edge_set, rng, iters=30)
            if best is None or len(pieces) < len(best):
                best = pieces
        if best is None:
            best = [[u, v] for (u, v) in es]
        results[(n, p, seed)] = best
    for (n, p, seed) in specs:
        if (n, p, seed) not in results:
            es = instance(n, p, seed)
            results[(n, p, seed)] = [[u, v] for (u, v) in es]
    return [results[(20, 0.4, 11)], results[(25, 0.35, 12)], results[(30, 0.3, 13)]]
