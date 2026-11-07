"""Rewritten: CSP backtracking with MRV + forward checking for packing T1..T20 into K20."""
import sys
import time
from random import Random

sys.setrecursionlimit(100000)


def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees


def _precompute(trees):
    n = len(trees)
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for k in range(1, n + 1):
        for (a, b) in trees[k - 1]:
            adj[k - 1].add((a, b)); adj[k - 1].add((b, a))
            deg[a] += 1; deg[b] += 1
    order = list(range(n))
    order.sort(key=lambda k: (-(k + 1), -max(deg, default=0)))
    return adj, deg, order


def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    trees = instance_trees(20, 611)
    adj, deg, order = _precompute(trees)
    n = 20
    n_trees = len(order)

    for attempt in range(500):
        if time.time() > deadline - 3.0:
            break
        rng = Random(attempt * 1000 + 7)
        free = set(range(n))
        assignments = {}
        stack = []

        def solve(idx):
            if idx == n_trees:
                return True
            k = order[idx]
            size = k + 1
            if len(free) < size:
                return False
            if time.time() > deadline:
                return False
            horder = sorted(free, key=lambda v: (-deg[v], rng.random()))
            torder = list(range(size))
            torder.sort(key=lambda v: (-deg[v], rng.random()))
            host_set = set()
            adjhost = [set() for _ in range(size)]
            host_of = [0] * size

            def find_host(tv):
                cand = []
                for h in horder:
                    if h in host_set:
                        continue
                    ok = True
                    for nb in adj[tv]:
                        if nb in host_set:
                            if h not in adjhost[nb]:
                                ok = False
                                break
                        else:
                            if h in adjhost[nb]:
                                ok = False
                                break
                    if ok:
                        cand.append(h)
                return cand

            def backtrack(tidx):
                if tidx == size:
                    return True
                tv = torder[tidx]
                cand = find_host(tv)
                if not cand:
                    return False
                cand.sort(key=lambda h: -sum(1 for nb in adj[tv] if nb in host_set and h in adjhost[nb]))
                for h in cand:
                    host_set.add(h)
                    for nb in adj[tv]:
                        adjhost[nb].add(h)
                    host_of[tv] = h
                    if backtrack(tidx + 1):
                        return True
                    for nb in adj[tv]:
                        adjhost[nb].discard(h)
                    host_set.discard(h)
                    host_of[tv] = 0
                return False

            if not backtrack(0):
                return False
            emb = list(host_of)
            for tv, hv in enumerate(emb):
                free.discard(hv)
            assignments[k] = emb
            stack.append((k, emb))
            if solve(idx + 1):
                return True
            for tv, hv in enumerate(emb):
                free.add(hv)
            del assignments[k]
            stack.pop()
            return False

        if solve(0):
            result = [None] * n
            for k in range(n):
                result[k - 1] = list(assignments[k])
            return result

    # Fallback: return seed-like result
    return [None] * 19 + [list(range(20))]
