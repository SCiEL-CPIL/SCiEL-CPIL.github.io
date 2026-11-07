import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 110.0
    trees = instance_trees(20, 611)
    N = 20

    tree_data = {}
    for k in range(1, 21):
        edges = trees[k - 1]
        if not edges:
            tree_data[k] = {'edges': [], 'deg': [0], 'root': 0, 'children': {0: []}}
            continue
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for a, b in edges:
            adj[a].append(b); adj[b].append(a)
            deg[a] += 1; deg[b] += 1
        root = max(range(k), key=lambda v: (deg[v], -v))
        parent = {root: -1}
        children = {v: [] for v in range(k)}
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    children[u].append(v)
                    q.append(v)
        tree_data[k] = {'edges': edges, 'deg': deg, 'root': root, 'children': children}

    ordered_ks = sorted(range(1, 21), key=lambda k: (-len(trees[k - 1]),
                                                       -max(tree_data[k]['deg']) if trees[k - 1] else 0))

    best_value = 0
    best_result = [None] * 20
    rng = Random(12345)

    for attempt in range(500):
        if time.time() > deadline:
            break
        host_order = list(range(N))
        rng.shuffle(host_order)
        used_edges = set()
        result = [None] * 20
        ok = True

        for k in ordered_ks:
            if time.time() > deadline:
                ok = False
                break
            edges = trees[k - 1]
            if not edges:
                result[k - 1] = [host_order[0]]
                continue

            td = tree_data[k]
            root = td['root']
            children = td['children']
            assignment = [None] * k
            used_hosts = set()
            nodes = [0]
            node_limit = 100000
            found = [False]

            def dfs(u, hu):
                if found[0] or time.time() > deadline or nodes[0] > node_limit:
                    return False
                nodes[0] += 1
                assignment[u] = hu
                used_hosts.add(hu)
                for v in children[u]:
                    for hv in host_order:
                        if hv in used_hosts:
                            continue
                        e = frozenset((hu, hv))
                        if e in used_edges:
                            continue
                        if dfs(v, hv):
                            found[0] = True
                            return True
                    if found[0]:
                        break
                used_hosts.discard(hu)
                assignment[u] = None
                return found[0]

            for hr in host_order:
                if found[0]:
                    break
                assignment = [None] * k
                used_hosts = set()
                nodes[0] = 0
                dfs(root, hr)

            if not found[0]:
                ok = False
                break

            for a, b in edges:
                used_edges.add(frozenset((assignment[a], assignment[b])))
            result[k - 1] = list(assignment)

        if ok:
            best_value = 190
            best_result = list(result)
            break
        else:
            value = sum(len(trees[k - 1]) for k in range(1, 21) if result[k - 1] is not None)
            if value > best_value:
                best_value = value
                best_result = list(result)

    final = []
    for k in range(1, 21):
        r = best_result[k - 1]
        final.append(None if r is None else [int(x) for x in r])
    return final
