import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140

    rng = Random(611)
    trees = []
    for k in range(1, 21):
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)

    n = 20
    used_edges = set()
    result = [None] * 20

    for tree_idx in range(20):
        if time.time() - start_time > DEADLINE:
            break
        k = tree_idx + 1

        if k == 1:
            used_verts = set()
            for r in result:
                if r is not None:
                    used_verts.update(r)
            for v in range(n):
                if v not in used_verts:
                    result[tree_idx] = [v]
                    break
        elif k == 2:
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if frozenset((u, v)) not in used_edges:
                        result[tree_idx] = [u, v]
                        used_edges.add(frozenset((u, v)))
                        found = True
                        break
                if found:
                    break
        else:
            embedding = _embed(tree_idx, used_edges, result, start_time, DEADLINE)
            if embedding is not None:
                result[tree_idx] = embedding
                for (u, v) in trees[tree_idx]:
                    used_edges.add(frozenset((embedding[u], embedding[v])))
            else:
                break

    return [list(r) if r is not None else None for r in result]

def _embed(tree_idx, used_edges, existing, start_time, deadline):
    k = tree_idx + 1
    tree = trees[tree_idx]
    adj = [[] for _ in range(k)]
    for (u, v) in tree:
        adj[u].append(v)
        adj[v].append(u)

    used_verts = set()
    for r in existing:
        if r is not None:
            used_verts.update(r)

    avail = [v for v in range(20) if v not in used_verts]
    if len(avail) < k:
        return None

    def dfs(node, parent, mapping, used_host):
        if time.time() - start_time > deadline:
            return None
        for hv in avail:
            if hv in used_host:
                continue
            if parent is not None and frozenset((mapping[parent], hv)) in used_edges:
                continue
            mapping[node] = hv
            used_host.add(hv)
            ok = True
            for child in adj[node]:
                if child == parent:
                    continue
                if dfs(child, node, mapping, used_host) is None:
                    ok = False
                    break
            if ok:
                for (u, v) in tree:
                    if frozenset((mapping[u], mapping[v])) in used_edges:
                        ok = False
                        break
            if ok:
                return mapping.copy()
            del mapping[node]
            used_host.remove(hv)
        return None

    return dfs(0, None, {}, set())
