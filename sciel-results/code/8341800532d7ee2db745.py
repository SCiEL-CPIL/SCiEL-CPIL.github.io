"""Fresh design: structured value families + complete-bipartite edge skeleton + simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAXV = 10000

def _eval(vals, edges):
    s = set()
    p = set()
    for u, v in edges:
        s.add(vals[u] + vals[v])
        p.add(vals[u] * vals[v])
    return max(len(s), len(p))

def _bipartite_edges(n1, n2, k):
    n = n1 + n2
    L = list(range(n1))
    R = list(range(n1, n))
    all_e = [(u, v) for u in L for v in R]
    random.shuffle(all_e)
    return set(all_e[:k]), L, R

def _init_bipartite():
    for a in range(2, 20):
        for b in range(2, 20):
            for c in range(2, 20):
                for d in range(2, 20):
                    vals = [a, b, c, d,
                            a + b, a + c, a + d,
                            b + c, b + d, c + d,
                            a * b, a * c, a * d,
                            b * c]
                    vals = sorted(set(vals))
                    if len(vals) < N:
                        continue
                    if max(vals) > MAXV:
                        continue
                    vals = vals[:N]
                    if len(set(vals)) != N:
                        continue
                    edges, L, R = _bipartite_edges(7, 7, M)
                    return vals, edges, L, R
    return None

def _try_improve(vals, edges, L, R, deadline, best):
    n = len(vals)
    cur = _eval(vals, edges)
    best_val = cur
    best_state = (list(vals), set(edges))
    t = 40.0
    cool = 0.9995
    tmin = 0.02
    rng = random.Random(12345)

    def rand_val():
        return rng.randint(1, MAXV)

    while time.time() < deadline:
        # pick a move type
        r = rng.random()
        if r < 0.30:
            # swap two values
            i, j = rng.sample(range(n), 2)
            new = list(vals)
            new[i], new[j] = vals[j], vals[i]
            if len(set(new)) != n or max(new) > MAXV or min(new) < 1:
                continue
            nv = _eval(new, edges)
            vals = new
        elif r < 0.55:
            # change one value
            i = rng.randrange(n)
            new = list(vals)
            new[i] = rand_val()
            if len(set(new)) != n or max(new) > MAXV or min(new) < 1:
                continue
            nv = _eval(new, edges)
            vals = new
        elif r < 0.80:
            # swap two edges
            elist = list(edges)
            if len(elist) < 2:
                continue
            e1, e2 = rng.sample(elist, 2)
            new_edges = set(edges)
            new_edges.discard(e1)
            new_edges.discard(e2)
            # cross endpoints
            u1, v1 = e1
            u2, v2 = e2
            cand = [(u1, v2), (u2, v1)]
            cand = [tuple(sorted(c)) for c in cand]
            cand = [c for c in cand if c[0] != c[1]]
            ok = False
            for c in cand:
                if c not in new_edges:
                    new_edges.add(c)
                    ok = True
                    break
            if not ok:
                continue
            nv = _eval(vals, new_edges)
            edges = new_edges
        else:
            # single edge replace
            e = rng.choice(list(edges))
            new_edges = set(edges)
            new_edges.discard(e)
            # try random new edge
            for _ in range(30):
                u, v = rng.sample(range(n), 2)
                c = (min(u, v), max(u, v))
                if c not in new_edges:
                    new_edges.add(c)
                    break
            else:
                continue
            nv = _eval(vals, new_edges)
            edges = new_edges

        if nv <= best_val:
            best_val = nv
            best_state = (list(vals), set(edges))
            if nv < best[0]:
                best[0] = nv
                best[1] = (list(vals), set(edges))
            t = 40.0
        else:
            if rng.random() < 1.0 / (1.0 + (nv - cur) / max(t, 1e-6)):
                cur = nv
            else:
                # revert
                if r < 0.30:
                    vals = list(best_state[0]) if False else vals  # keep partial; simple revert:
                # simpler: revert to previous by re-deriving is complex; instead accept only if improved
                pass
        cur = _eval(vals, edges)
        t *= cool
        if t < tmin:
            t = tmin
    return best_state, best_val

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin vs 90s kill
    rng = random.Random(7)

    best = (10**9, None)
    # Try several structured initializations
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        init = _init_bipartite()
        if init is None:
            continue
        vals, edges, L, R = init
        cur = _eval(vals, edges)
        if cur < best[0]:
            best = (cur, (list(vals), set(edges)))
        if time.time() > deadline - 20.0:
            break
        # short local search per init
        bstate, bval = _try_improve(vals, edges, L, R, min(deadline, time.time() + 15.0), list(best))
        if bval < best[0]:
            best = (bval, bstate)

    if best[1] is None:
        # fallback
        vals = list(range(1, N + 1))
        edges = set(list(combinations(range(N), 2))[:M])
        best = (_eval(vals, edges), (vals, edges))

    vals, edges = best[1]
    # enforce types
    vals = [int(x) for x in vals]
    edges = [list(e) for e in edges]
    return {'values': vals, 'edges': edges}
# EVOLVE-BLOCK-END
