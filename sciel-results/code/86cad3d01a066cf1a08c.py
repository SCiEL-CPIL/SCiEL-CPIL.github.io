import time
import random

def instance_trees(n=20, seed=611):
    from random import Random
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
    deadline = time.time() + 140
    trees = instance_trees(20, 611)
    n_hosts = 20
    
    tree_adj = []
    for tree in trees:
        k = len(tree) + 1
        adj = [[] for _ in range(k)]
        for (u, v) in tree:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    order = sorted(range(20), key=lambda i: len(trees[i]), reverse=True)
    
    def embed_tree(tree_idx, used_edges, rng):
        k = tree_idx + 1
        if k == 1:
            return [0]
        adj = tree_adj[tree_idx]
        
        for attempt in range(200):
            host_order = list(range(n_hosts))
            rng.shuffle(host_order)
            mapping = [-1] * k
            used_hosts = set()
            
            def dfs(v, parent):
                if time.time() > deadline:
                    return False
                for h in host_order:
                    if h in used_hosts:
                        continue
                    mapping[v] = h
                    used_hosts.add(h)
                    ok = True
                    for nb in adj[v]:
                        if nb != parent and mapping[nb] != -1:
                            a, b = h, mapping[nb]
                            if a > b:
                                a, b = b, a
                            if (a, b) in used_edges:
                                ok = False
                                break
                    if ok:
                        for nb in adj[v]:
                            if nb != parent and mapping[nb] == -1:
                                if not dfs(nb, v):
                                    ok = False
                                    break
                        if ok:
                            return True
                    used_hosts.remove(h)
                    mapping[v] = -1
                return False
            
            if dfs(0, -1):
                return list(mapping)
        return None
    
    def search(pos, used_edges, embeddings, rng):
        if time.time() > deadline:
            return False
        if pos == 20:
            return True
        tree_idx = order[pos]
        mapping = embed_tree(tree_idx, used_edges, rng)
        if mapping is not None:
            new_edges = set()
            for (u, v) in trees[tree_idx]:
                a, b = mapping[u], mapping[v]
                if a > b:
                    a, b = b, a
                new_edges.add((a, b))
            embeddings[tree_idx] = mapping
            if search(pos + 1, used_edges | new_edges, embeddings, rng):
                return True
            embeddings[tree_idx] = None
        return False
    
    for trial in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(trial * 7919 + 104729)
        embeddings = [None] * 20
        used_edges = set()
        if search(0, used_edges, embeddings, rng):
            result = []
            for i in range(20):
                if embeddings[i] is None:
                    result.append(None)
                else:
                    result.append([int(v) for v in embeddings[i]])
            return result
    
    return [None] * 19 + [list(range(20))]
