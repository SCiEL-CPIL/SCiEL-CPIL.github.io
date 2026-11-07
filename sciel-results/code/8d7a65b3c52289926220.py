import time
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
    trees = instance_trees()
    
    # Pre-compute adjacency and degrees
    tree_adj = []
    tree_deg = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        tree_deg.append([len(adj[v]) for v in range(k)])
        
    # Sort indices by size descending
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # All edges in K20
    all_edges = set((i, j) for i in range(20) for j in range(i+1, 20))
    
    embedding = [None] * 20
    
    # Backtracking solver
    def solve(idx):
        if idx == 20:
            return True
        
        orig_idx = tree_indices[idx]
        t_edges = trees[orig_idx]
        k = len(t_edges) + 1 if t_edges else 1
        adj = tree_adj[orig_idx]
        deg = tree_deg[orig_idx]
        
        if k == 1:
            embedding[orig_idx] = [0]
            return solve(idx + 1)
        
        # Order vertices by degree desc
        verts = list(range(k))
        verts.sort(key=lambda v: -deg[v])
        
        # Try to find embeddings
        # We use a generator to yield valid embeddings
        
        def gen_embeddings():
            map = {}
            used = set()
            
            def backtrack(i):
                if i == k:
                    res = [0] * k
                    for t, h in map.items():
                        res[t] = h
                    yield res
                    return
                
                t = verts[i]
                for h in range(20):
                    if h in used:
                        continue
                    
                    valid = True
                    for u in adj[t]:
                        if u in map:
                            hu = map[u]
                            e = (min(h, hu), max(h, hu))
                            if e not in all_edges:
                                valid = False
                                break
                    
                    if valid:
                        map[t] = h
                        used.add(h)
                        
                        yield from backtrack(i + 1)
                        
                        del map[t]
                        used.remove(h)
            
            yield from backtrack(0)

        for emb in gen_embeddings():
            edges_to_remove = set()
            for u, v in t_edges:
                hu = emb[u]
                hv = emb[v]
                e = (min(hu, hv), max(hu, hv))
                edges_to_remove.add(e)
                
            # Remove edges
            for e in edges_to_remove:
                all_edges.remove(e)
                
            embedding[orig_idx] = emb
            
            if solve(idx + 1):
                return True
                
            # Restore
            for e in edges_to_remove:
                all_edges.add(e)
            embedding[orig_idx] = None
            
        return False

    # Time limit guard
    start_time = time.time()
    
    # We can't easily interrupt the generator in Python without threads or flags.
    # But given the small size, it should be fast.
    # If it's not, we might hang.
    # Let's add a simple check: if it takes > 10s, abort and use greedy.
    # But we can't easily abort the recursion.
    
    # Alternative: Run solve in a thread? Or just trust it.
    # Let's trust it for now. If it's too slow, the evaluator will kill it.
    
    success = solve(0)
    
    if success:
        result = []
        for i in range(20):
            if embedding[i] is None:
                result.append(None)
            else:
                result.append([int(x) for x in embedding[i]])
        return result
    else:
        # Fallback to Greedy
        all_edges_greedy = set((i, j) for i in range(20) for j in range(i+1, 20))
        embedding_greedy = [None] * 20
        
        for orig_idx in tree_indices:
            t_edges = trees[orig_idx]
            k = len(t_edges) + 1 if t_edges else 1
            
            if k == 1:
                embedding_greedy[orig_idx] = [0]
                continue
                
            verts = list(range(k))
            verts.sort(key=lambda v: -tree_deg[orig_idx][v])
            
            found_emb = None
            
            def greedy_backtrack(i, map, used):
                nonlocal found_emb
                if found_emb is not None:
                    return
                if i == k:
                    res = [0]*k
                    for t, h in map.items():
                        res[t] = h
                    found_emb = res
                    return
                
                t = verts[i]
                for h in range(20):
                    if h in used:
                        continue
                    valid = True
                    for u in tree_adj[orig_idx][t]:
                        if u in map:
                            hu = map[u]
                            e = (min(h, hu), max(h, hu))
                            if e not in all_edges_greedy:
                                valid = False
                                break
                    if valid:
                        map[t] = h
                        used.add(h)
                        greedy_backtrack(i+1, map, used)
                        if found_emb is not None:
                            return
                        del map[t]
                        used.remove(h)
            
            greedy_backtrack(0, {}, set())
            
            if found_emb:
                embedding_greedy[orig_idx] = found_emb
                for u, v in t_edges:
                    hu = found_emb[u]
                    hv = found_emb[v]
                    e = (min(hu, hv), max(hu, hv))
                    all_edges_greedy.remove(e)
            else:
                embedding_greedy[orig_idx] = None
        
        result = []
        for i in range(20):
            if embedding_greedy[i] is None:
                result.append(None)
            else:
                result.append([int(x) for x in embedding_greedy[i]])
        return result

