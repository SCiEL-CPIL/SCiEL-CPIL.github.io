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
    start_time = time.time()
    deadline = start_time + 115
    
    trees = instance_trees(20, 611)
    
    all_edges = set()
    for i in range(20):
        for j in range(i + 1, 20):
            all_edges.add((i, j))
            
    # Pre-compute adjacency for K20 for speed?
    # Not necessary, set lookup is O(1).
    
    def find_embedding(tree_edges, available_edges, k, rng):
        if k == 1:
            # Pick any available vertex? 
            # T1 uses 1 vertex. It doesn't use edges.
            # We just need to pick a vertex.
            # To be consistent, pick the first available one?
            # Actually, any vertex is fine.
            return [0]
            
        adj = {i: [] for i in range(k)}
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        parent = [-1] * k
        order = [0]
        parent[0] = -2
        for u in order:
            for v in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    order.append(v)
                    
        host_map = [None] * k
        
        def backtrack(idx):
            if idx == k:
                return True
                
            u = order[idx]
            candidates = []
            
            if parent[u] != -2:
                p = parent[u]
                host_p = host_map[p]
                for v in range(20):
                    if v == host_p:
                        continue
                    e1 = (host_p, v) if host_p < v else (v, host_p)
                    if e1 in available_edges:
                        used = False
                        for i in range(idx):
                            if host_map[i] == v:
                                used = True
                                break
                        if not used:
                            candidates.append(v)
            else:
                candidates = list(range(20))
                
            if not candidates:
                return False
                
            rng.shuffle(candidates)
            
            for v in candidates:
                host_map[u] = v
                if backtrack(idx + 1):
                    return True
                host_map[u] = None
                
            return False
            
        if backtrack(0):
            return host_map
        return None

    def try_packing(seed_val):
        rng = Random(seed_val)
        available_edges = set(all_edges)
        embeddings = [None] * 20
        edges_embedded = 0
        
        for k in range(20, 0, -1):
            tree = trees[k-1]
            emb = find_embedding(tree, available_edges, k, rng)
            if emb is None:
                break
                
            valid = True
            for u, v in tree:
                hu = emb[u]
                hv = emb[v]
                e = (hu, hv) if hu < hv else (hv, hu)
                if e not in available_edges:
                    valid = False
                    break
                available_edges.remove(e)
                
            if valid:
                embeddings[k-1] = emb
                edges_embedded += len(tree)
            else:
                break
                
        return embeddings, edges_embedded

    best_emb = [None] * 20
    best_score = 0
    
    # Try many seeds
    for seed in range(100000):
        if time.time() > deadline:
            break
        emb, score = try_packing(seed)
        if score > best_score:
            best_score = score
            best_emb = emb
            if best_score == 190:
                # Convert to required types
                return [list(e) if e is not None else None for e in best_emb]
                
    # Return best found
    return [list(e) if e is not None else None for e in best_emb]

