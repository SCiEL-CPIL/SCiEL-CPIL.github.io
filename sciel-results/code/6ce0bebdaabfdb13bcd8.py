import time
import random

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 140
    
    trees = instance_trees(20, 611)
    used_edges = set()
    result = [None] * 20
    
    tree_structures = []
    for k in range(1, 21):
        if k == 1:
            tree_structures.append(([], [0], [-1], {}))
            continue
        edges = trees[k-1]
        adj = {i: [] for i in range(k)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    order.append(v)
        tree_structures.append((edges, order, parent, adj))

    deg_host = [19] * 20
    
    def update_deg_host(u, v):
        e = (min(u, v), max(u, v))
        if e not in used_edges:
            used_edges.add(e)
            deg_host[u] -= 1
            deg_host[v] -= 1

    rng = random.Random(123)

    def solve_tree(k_idx):
        if k_idx == 1:
            return [0]
        
        edges, order, parent, adj = tree_structures[k_idx]
        k = k_idx
        mapping = {}
        used_hosts_in_tree = set()
        
        def dfs(idx):
            if idx == k:
                return True
            
            v = order[idx]
            
            if idx == 0:
                candidates = list(range(20))
            else:
                p = parent[v]
                h_p = mapping[p]
                candidates = []
                for u in range(20):
                    if u in used_hosts_in_tree:
                        continue
                    e = (min(h_p, u), max(h_p, u))
                    if e in used_edges:
                        continue
                    candidates.append(u)
            
            candidates.sort(key=lambda u: deg_host[u], reverse=True)
            
            if len(candidates) > 3:
                rng.shuffle(candidates)
            
            for u in candidates:
                mapping[v] = u
                used_hosts_in_tree.add(u)
                
                if dfs(idx + 1):
                    return True
                    
                del mapping[v]
                used_hosts_in_tree.remove(u)
                
            return False
        
        if dfs(0):
            for u, v in edges:
                h_u = mapping[u]
                h_v = mapping[v]
                update_deg_host(h_u, h_v)
            return [mapping[i] for i in range(k)]
        
        return None

    for k in range(20, 0, -1):
        if time.time() > deadline:
            break
        res = solve_tree(k)
        if res is not None:
            result[k-1] = res
        else:
            pass

    return result
