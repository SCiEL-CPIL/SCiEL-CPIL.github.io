"""Packing trees T1..T20 into K20 via randomized backtracking."""
import time
from random import Random

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START


def _instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
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


def _build_adj(edges, k):
    adj = [[] for _ in range(k)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _traversal_order(adj, k, root):
    order = [root]
    seen = {root}
    i = 0
    while i < len(order):
        v = order[i]; i += 1
        for nb in adj[v]:
            if nb not in seen:
                seen.add(nb)
                order.append(nb)
    return order


# EVOLVE-BLOCK-START
def construct():
    n = 20
    trees = _instance_trees(n, 611)
    idx_by_k = list(range(len(trees)))
    idx_by_k.sort(key=lambda i: len(trees[i]), reverse=True)

    for seed in range(0, 50000):
        if _elapsed() > _DEADLINE:
            break
        rng = Random(seed)
        # randomized adjacency neighbor order
        trees_rand = []
        for edges in trees:
            k = len(edges) + 1
            if k <= 1:
                trees_rand.append(None)
                continue
            adj = _build_adj(edges, k)
            for v in range(k):
                rng.shuffle(adj[v])
            trees_rand.append(adj)

        used = [[False] * n for _ in range(n)]
        result = [None] * len(trees)
        success = True
        for ti in idx_by_k:
            edges = trees[ti]
            k = len(edges) + 1
            if k == 1:
                result[ti] = None
                continue
            adj = trees_rand[ti]
            order = _traversal_order(adj, k, 0)
            root_host = rng.randrange(n)
            used[root_host, root_host] = True
            host_of = [-1] * k
            host_of[0] = root_host

            def bt(idx):
                if idx == len(order):
                    return True
                t = order[idx]
                for h in range(n):
                    if used[t, h]:
                        continue
                    ok = True
                    for nb in adj[t]:
                        if host_of[nb] != -1:
                            if used[h, host_of[nb]]:
                                ok = False
                                break
                    if not ok:
                        continue
                    host_of[t] = h
                    used[t, h] = True
                    for nb in adj[t]:
                        if host_of[nb] != -1:
                            used[h, host_of[nb]] = True
                            used[host_of[nb], h] = True
                    if bt(idx + 1):
                        return True
                    host_of[t] = -1
                    used[t, h] = False
                    for nb in adj[t]:
                        if host_of[nb] != -1:
                            used[h, host_of[nb]] = False
                            used[host_of[nb], h] = False
                return False

            ok = bt(1)
            used[root_host, root_host] = False
            if not ok:
                success = False
                break
            mapping = [int(host_of[i]) for i in range(k)]
            if len(set(mapping)) != k:
                success = False
                break
            result[ti] = mapping
        if success:
            out = []
            for r in result:
                if r is None:
                    out.append(None)
                else:
                    out.append([int(x) for x in r])
            return out
    # fallback
    return [None] * 19 + [list(range(20))]
# EVOLVE-BLOCK-END
