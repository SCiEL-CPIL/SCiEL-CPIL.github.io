import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _build_adj_score(edges, n=16):
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    deg = A.sum(axis=1)
    best = 0
    for u in range(n):
        Au = A[u]
        du = deg[u]
        for v in range(u + 1, n):
            if Au[v]:
                common = np.nonzero(Au & A[v])[0]
                dv = deg[v]
                for w in common:
                    if w > v:
                        s = du + dv + deg[w]
                        if s > best:
                            best = s
    return A, best

def _seed_edges(n=16):
    e = [(u, v) for u in range(8) for v in range(8, 16)]
    e.append((0, 1))
    return e

def construct():
    n = 16
    target = 65
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng = np.random.default_rng(2024)

    best_edges = _seed_edges(n)
    best_A, best_score = _build_adj_score(best_edges, n)

    cur_edges = list(best_edges)
    cur_set = set(frozenset(e) for e in cur_edges)
    cur_A = best_A
    cur_score = best_score

    T = 5.0
    T_min = 0.03
    total_steps = 1200
    moves_per_step = 110

    for step in range(total_steps):
        if _elapsed() > _DEADLINE - 6.0:
            break
        for _ in range(moves_per_step):
            if _elapsed() > _DEADLINE - 6.0:
                break
            remove = cur_edges[rng.integers(0, len(cur_edges))]
            while True:
                add = all_edges[rng.integers(0, len(all_edges))]
                if frozenset(add) not in cur_set:
                    break
            new_edges = [e for e in cur_edges if e != remove]
            new_edges.append(add)
            new_A, new_score = _build_adj_score(new_edges, n)

            if new_score < cur_score:
                cur_edges = new_edges
                cur_set = cur_set - {frozenset(remove)} | {frozenset(add)}
                cur_A = new_A
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = list(cur_edges)
                    best_A = cur_A
            else:
                delta = new_score - cur_score
                if delta > 0 and rng.random() < np.exp(-delta / T):
                    cur_edges = new_edges
                    cur_set = cur_set - {frozenset(remove)} | {frozenset(add)}
                    cur_A = new_A
                    cur_score = new_score
        T = max(T_min, T * 0.99)

        if step % 250 == 0 and step > 0 and cur_score > best_score + 1:
            cur_edges = list(best_edges)
            cur_set = set(frozenset(e) for e in cur_edges)
            cur_A = best_A
            cur_score = best_score

    # polish
    if _elapsed() < _DEADLINE - 4.0:
        cur_edges = list(best_edges)
        cur_set = set(frozenset(e) for e in cur_edges)
        cur_A = best_A
        cur_score = best_score
        improved = True
        while improved and _elapsed() < _DEADLINE - 4.0:
            improved = False
            for ri in range(len(cur_edges)):
                if _elapsed() > _DEADLINE - 4.0:
                    break
                remove = cur_edges[ri]
                for add in all_edges:
                    if frozenset(add) in cur_set:
                        continue
                    new_edges = [e for e in cur_edges if e != remove]
                    new_edges.append(add)
                    new_A, new_score = _build_adj_score(new_edges, n)
                    if new_score < cur_score:
                        cur_edges = new_edges
                        cur_set = cur_set - {frozenset(remove)} | {frozenset(add)}
                        cur_A = new_A
                        cur_score = new_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_edges = list(cur_edges)
                            best_A = cur_A
                        break

    return {'n': n, 'edges': [[int(u), int(v)] for u, v in best_edges]}
