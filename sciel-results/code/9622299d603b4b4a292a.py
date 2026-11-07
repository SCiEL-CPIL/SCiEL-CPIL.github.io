"""Rewritten: joint value+edge local search for small sums/products on 35 edges."""
import time
import random


def _raw(values, edges):
    ss = set()
    pp = set()
    for i, j in edges:
        ss.add(values[i] + values[j])
        pp.add(values[i] * values[j])
    return max(len(ss), len(pp))


def _select_edges(values, time_budget):
    n = len(values)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, values[i] + values[j], values[i] * values[j]))
    m = len(pairs)
    need = 35

    # Greedy initialization
    sel = []
    selset = set()
    S = set()
    P = set()
    for _ in range(need):
        best = None
        bestkey = None
        for idx in range(m):
            if idx in selset:
                continue
            _, _, s, p = pairs[idx]
            ns = len(S) + (0 if s in S else 1)
            np = len(P) + (0 if p in P else 1)
            key = (max(ns, np), ns + np)
            if bestkey is None or key < bestkey:
                bestkey = key
                best = idx
        sel.append(best)
        selset.add(best)
        S.add(pairs[best][2])
        P.add(pairs[best][3])

    def score_of(sel_list):
        S2 = set()
        P2 = set()
        for idx in sel_list:
            S2.add(pairs[idx][2])
            P2.add(pairs[idx][3])
        return (max(len(S2), len(P2)), len(S2) + len(P2))

    best_score = score_of(sel)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        sel_list = sel[:]
        for a in range(len(sel_list)):
            for b in range(m):
                if b in selset:
                    continue
                trial = sel_list[:]
                trial[a] = b
                sc = score_of(trial)
                if sc < best_score:
                    best_score = sc
                    sel_list = trial
                    selset.discard(sel[a])
                    selset.add(b)
                    improved = True
                    a = len(sel_list)  # restart scan
        sel = sel_list
    return [[pairs[idx][0], pairs[idx][1]] for idx in sel]


def construct():
    t0 = time.time()
    deadline = t0 + 50.0
    rng = random.Random(12345)

    def eval_values(values, tb):
        edges = _select_edges(values, tb)
        return _raw(values, edges), edges

    # Multiple initial families
    inits = [
        [1 << i for i in range(14)],                 # geometric 1..8192
        list(range(1, 15)),                           # arithmetic
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    ]

    best = None  # (raw, edges, values)
    for v in inits:
        if time.time() > deadline - 6:
            break
        r, e = eval_values(v, 0.5)
        if best is None or r < best[0]:
            best = (r, e, v[:])

    cur_v = best[2][:]
    cur_r = best[0]
    cur_e = best[1]

    # Value hill-climb
    while time.time() < deadline - 4:
        v = cur_v[:]
        k = rng.randrange(14)
        newv = rng.randrange(1, 10001)
        if newv in v:
            continue
        v[k] = newv
        r, e = eval_values(v, 0.3)
        if r < cur_r:
            cur_v = v
            cur_r = r
            cur_e = e
            best = (r, e, v[:])

    # Final stronger edge optimization on best values
    if time.time() < deadline - 2:
        e = _select_edges(cur_v, max(0.5, deadline - time.time() - 1.0))
        r = _raw(cur_v, e)
        if r <= cur_r:
            cur_r = r
            cur_e = e
            best = (r, e, cur_v[:])

    return {
        'values': [int(x) for x in best[2]],
        'edges': [[int(i), int(j)] for i, j in best[1]],
    }
