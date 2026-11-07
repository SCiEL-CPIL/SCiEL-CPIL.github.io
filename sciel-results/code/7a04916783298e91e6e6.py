from random import Random
import time

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

N = 20

def construct():
    start_time = time.time()
    deadline = 140.0
    
    trees = instance_trees(N, 611)
    
    tree_indices = sorted(range(N), key=lambda i: len(trees[i]), reverse=True)
    
    used_edges = set()
    result = [None] * N
    
    for idx in tree_indices:
        if time.time() - start_time > deadline:
            break
        tree = trees[idx]
        k = len(tree) + 1
        
        if k == 1:
            result[idx] = [0]
            continue
        
        if k == 2:
            placed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) not in used_edges:
                        result[idx] = [u, v]
                        used_edges.add((u, v))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                result[idx] = None
            continue
        
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        
        root = next((i for i in range(k) if len(adj[i]) == 1), 0)
        
        parent = [-1] * k
        children = [[] for _ in range(k)]
        visited = [False] * k
        visited[root] = True
        order = [root]
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    children[u].append(v)
                    queue.append(v)
                    order.append(v)
        
        mapping = [-1] * k
        mapping[root] = 0
        host_used = [False] * N
        host_used[0] = True
        
        def backtrack(pos):
            if time.time() - start_time > deadline:
                return False
            if pos == len(order):
                return True
            
            tv = order[pos]
            ph = mapping[parent[tv]]
            
            cands = []
            for hv in range(N):
                if hv == ph or host_used[hv]:
                    continue
                e = (min(ph, hv), max(ph, hv))
                if e not in used_edges:
                    cands.append(hv)
            
            if not cands:
                return False
            
            for hv in cands:
                e = (min(ph, hv), max(ph, hv))
                mapping[tv] = hv
                host_used[hv] = True
                used_edges.add(e)
                
                if backtrack(pos + 1):
                    return True
                
                used_edges.discard(e)
                host_used[hv] = False
                mapping[tv] = -1
            
            return False
        
        if backtrack(0):
            result[idx] = [int(mapping[i]) for i in range(k)]
        else:
            result[idx] = None
            break
    
    return result
