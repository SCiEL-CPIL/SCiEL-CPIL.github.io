import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    trees = instance_trees(20, 611)
    n = 20
    
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    def embed_tree(tree, used_edges, rng, max_attempts=500):
        """Try to embed tree into unused edges. Returns list of host vertices or None."""
        k = len(tree) + 1
        if k == 1:
            return [0]  # Any single vertex works
        if k == 2:
            for _ in range(max_attempts):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                e = (min(i, j), max(i, j))
                if e not in used_edges:
                    return [i, j]
            return None
        
        # Build adjacency
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS order from vertex 0
        visited = [False] * k
        order = [0]
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    queue.append(v)
        
        # Backtracking
        host = [None] * k
        used = set(used_edges)
        
        def backtrack(pos):
            if pos == k:
                return True
            tv = order[pos]
            # Find parent
            parent_tv = None
            for u, v in tree:
                if u == tv and host[v] is not None:
                    parent_tv = v
                    break
                if v == tv and host[u] is not None:
                    parent_tv = u
                    break
            
            candidates = list(range(n))
            rng.shuffle(candidates)
            
            for hv in candidates:
                if any(host[i] == hv for i in range(pos)):
                    continue
                host[tv] = hv
                if parent_tv is not None:
                    e = (min(hv, host[parent_tv]), max(hv, host[parent_tv]))
                    if e in used:
                        host[tv] = None
                        continue
                    used.add(e)
                
                if backtrack(pos + 1):
                    return True
                
                if parent_tv is not None:
                    used.discard((min(hv, host[parent_tv]), max(hv, host[parent_tv])))
                host[tv] = None
            
            return False
        
        if backtrack(0):
            return [int(host[i]) for i in range(k)]
        return None
    
    best_result = [None] * n
    best_score = 0
    
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = Random(seed)
        used_edges = set()
        result = [None] * n
        
        # Embed in order: largest first
        order = sorted(range(n), key=lambda i: len(trees[i]), reverse=True)
        
        success = True
        for idx in order:
            if time.time() > deadline:
                success = False
                break
            tree = trees[idx]
            emb = embed_tree(tree, used_edges, rng, max_attempts=200)
            if emb is None:
                success = False
                break
            result[idx] = emb
            for u, v in tree:
                e = (min(emb[u], emb[v]), max(emb[u], emb[v]))
                used_edges.add(e)
        
        if success:
            score = sum(len(trees[i]) for i in range(n))
            if score > best_score:
                best_score = score
                best_result = result
            if best_score == 190:
                break
    
    final_result = []
    for i in range(n):
        if best_result[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_result[i]])
    
    return final_result
