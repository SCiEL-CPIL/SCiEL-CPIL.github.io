import time
import random

N = 15
DEADLINE = 50.0  # safety margin under the 60 s wall-clock limit

def _adjacency(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees(adj):
    return [len(adj[i]) for i in range(N)]

def _is_valid(adj, deg):
    # For every middle edge (u, v) with u < v, for every simple 3-edge path
    # s - u - v - t  (s in adj[u]\{v}, t in adj[v]\{u}, s != t),
    # require deg[s] != deg[t].
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _adjacency(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def _greedy(order, start_edges):
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    # single pass over the shuffled order; add an edge if it keeps validity
    for (u, v) in order:
        key = (min(u, v), max(u, v))
        if key in present:
            continue
        trial = edges + [(u, v)]
        if _valid(trial):
            edges = trial
            present.add(key)
    return edges

def construct():
    t0 = time.time()
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng = random.Random(20240607)

    # A guaranteed-valid seed: the star centred at 0 has no 3-edge path at all.
    star = [(0, v) for v in range(1, N)]
    best = star
    best_len = len(star)

    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand

    # Phase 1: many randomized greedy builds from empty and from the star.
    while time.time() - t0 < DEADLINE:
        order = all_edges[:]
        rng.shuffle(order)
        for start in (star, []):
            if time.time() - t0 > DEADLINE:
                break
            cand = _greedy(order, start)
            consider(cand)
            # a second pass in a fresh order to fill remaining gaps
            if time.time() - t0 < DEADLINE:
                order2 = all_edges[:]
                rng.shuffle(order2)
                cand2 = _greedy(order2, cand)
                consider(cand2)

    # Phase 2: light local search -- drop one edge, re-greedy, keep improvement.
    while time.time() - t0 < DEADLINE:
        base = best
        for idx in range(len(base)):
            if time.time() - t0 > DEADLINE:
                break
            reduced = base[:idx] + base[idx + 1:]
            order = all_edges[:]
            rng.shuffle(order)
            cand = _greedy(order, reduced)
            consider(cand)
            if time.time() - t0 < DEADLINE:
                order2 = all_edges[:]
                rng.shuffle(order2)
                cand2 = _greedy(order2, cand)
                consider(cand2)

    # Exact type conversion: Python int inside a list of lists.
    result = [[int(u), int(v)] for (u, v) in best]
    return result
