import time
import random
from random import Random

def instance_trees(n=20, seed=611):
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 20
    trees = instance_trees(n, 611)

    # Precompute adjacency for each tree
    tree_adj = []
    for k in range(1, n + 1):
        edges = trees[k - 1]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)

    # Precompute parent and order for each tree (rooted at 0)
    tree_parent = []
    tree_order = []
    for k in range(1, n + 1):
        if k == 1:
            tree_parent.append([-1])
            tree_order.append([0])
            continue
        adj = tree_adj[k - 1]
        parent = [-1] * k
        parent[0] = -2  # root
        order_list = [0]
        for v in order_list:
            for nb in adj[v]:
                if parent[nb] == -1:
                    parent[nb] = v
                    order_list.append(nb)
        tree_parent.append(parent)
        tree_order.append(order_list)

    # Process trees in order of increasing size (least constrained first)
    # This allows small trees to find embeddings easily, leaving the dense graph
    # for larger trees which have more flexibility
    order = list(range(n))  # indices 0 to 19, i.e., k=1,2,...,20

    best_result = [None] * n
    best_edges = 0

    def try_solution(rng):
        nonlocal best_result, best_edges
        used_edges = set()
        embedding = [None] * n

        for idx in order:
            k = idx + 1  # tree size
            tree_edges = trees[idx]

            if k == 1:
                embedding[idx] = [rng.randint(0, n - 1)]
                continue

            adj = tree_adj[idx]
            parent = tree_parent[idx]
            order_list = tree_order[idx]

            mapping = [-1] * k

            def dfs(pos):
                if time.time() > deadline:
                    return False

                if pos == len(order_list):
                    for u, v in tree_edges:
                        gu, gv = mapping[u], mapping[v]
                        e = (min(gu, gv), max(gu, gv))
                        if e in used_edges:
                            return False
                    return True

                tv = order_list[pos]

                if parent[tv] != -2:
                    p_tv = parent[tv]
                    p_gv = mapping[p_tv]
                    candidates = [v for v in range(n)
                                  if v != p_gv
                                  and (min(p_gv, v), max(p_gv, v)) not in used_edges]
                else:
                    candidates = list(range(n))

                rng.shuffle(candidates)

                for gv in candidates:
                    if gv in mapping:
                        continue

                    ok = True
                    for nb in adj[tv]:
                        if nb == parent[tv]:
                            continue
                        if mapping[nb] != -1:
                            e = (min(gv, mapping[nb]), max(gv, mapping[nb]))
                            if e in used_edges:
                                ok = False
                                break

                    if not ok:
                        continue

                    mapping[tv] = gv
                    if dfs(pos + 1):
                        return True
                    mapping[tv] = -1

                return False

            found = False
            for root_attempt in range(3):
                if time.time() > deadline:
                    return False

                mapping = [-1] * k
                if dfs(0):
                    found = True
                    break

            if not found:
                return False

            for u, v in tree_edges:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                used_edges.add(e)

            embedding[idx] = mapping[:]

        total_edges = len(used_edges)
        if total_edges > best_edges:
            best_edges = total_edges
            best_result = [list(e) if e is not None else None for e in embedding]
        return True

    restart = 0
    while time.time() < deadline:
        restart += 1
        rng = Random(611 + restart * 7919)
        try_solution(rng)
        if best_edges == 190:
            break

    if best_edges > 0:
        return best_result

    return [None] * 19 + [list(range(20))]
