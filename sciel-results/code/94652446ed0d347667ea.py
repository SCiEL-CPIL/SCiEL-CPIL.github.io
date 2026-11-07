from random import Random
import time

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
    deadline = start_time + 140.0
    
    trees = instance_trees(20, 611)
    N = 20
    
    # Build adjacency for each tree
    tree_adj = []
    for edges in trees:
        k = len(edges) + 1 if edges else 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order: largest first
    order = sorted(range(20), key=lambda i: len(trees[i]), reverse=True)
    
    def score(result):
        return sum(len(trees[i]) for i in range(20) if result[i] is not None)
    
    def embed_tree_random(idx, used_edges, rng, max_tries=50000):
        """Try to embed tree idx using random permutations."""
        k = idx + 1
        edges = trees[idx]
        
        if k == 1:
            return [0]
        
        for _ in range(max_tries):
            if time.time() > deadline:
                return None
            
            perm = list(range(N))
            rng.shuffle(perm)
            
            valid = True
            for u, v in edges:
                hu, hv = perm[u], perm[v]
                if hu == hv:
                    valid = False
                    break
                e = (min(hu, hv), max(hu, hv))
                if e in used_edges:
                    valid = False
                    break
            
            if valid:
                return [perm[v] for v in range(k)]
        
        return None
    
    # Main packing with restarts
    best = [None] * 20
    best_s = 0
    
    rng = Random(999)
    
    for restart in range(1000):
        if time.time() > deadline - 3.0:
            break
        
        used_edges = set()
        result = [None] * 20
        success = True
        
        for idx in order:
            if time.time() > deadline - 1.0:
                success = False
                break
            k = idx + 1
            if k == 1:
                result[idx] = [0]
                continue
            
            host = embed_tree_random(idx, used_edges, rng, max_tries=50000)
            if host is None:
                success = False
                break
            
            result[idx] = host
            for u, v in trees[idx]:
                hu, hv = host[u], host[v]
                used_edges.add((min(hu, hv), max(hu, hv)))
        
        s = score(result)
        if s > best_s:
            best_s = s
            best = result[:]
        
        if best_s == 190:
            break
    
    # Final type conversion
    final = []
    for i in range(20):
        if best[i] is None:
            final.append(None)
        else:
            final.append([int(x) for x in best[i]])
    
    return final
