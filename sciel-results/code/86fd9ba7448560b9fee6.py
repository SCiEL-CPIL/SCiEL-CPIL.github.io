import time
import random
import sys

sys.setrecursionlimit(100000)

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
    deadline = start_time + 115
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20
    
    tree_adj = []
    tree_edges = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        tree_adj.append(adj_list)
        tree_edges.append(edges)
    
    # Precompute edge index for fast lookup
    edge_to_idx = {}
    idx = 0
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            edge_to_idx[(u, v)] = idx
            idx += 1
    n_edges_total = 190
    
    order = list(range(n_trees - 1, -1, -1))
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def try_packing(seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(seed)
        
        used = [False] * n_edges_total
        mappings = [None] * n_trees
        
        def edge_idx(u, v):
            if u > v:
                u, v = v, u
            return edge_to_idx[(u, v)]
            
        def is_avail(u, v):
            if u == v:
                return False
            return not used[edge_idx(u, v)]
            
        def mark(u, v):
            used[edge_idx(u, v)] = True
            
        def unmark(u, v):
            used[edge_idx(u, v)] = False

        def pack_tree(t_idx):
            k = t_idx + 1
            if k == 1:
                mappings[t_idx] = [0]
                return True
            
            adj_t = tree_adj[t_idx]
            edges_t = tree_edges[t_idx]
            mapping = [-1] * k
            used_v = [False] * n_vertices
            
            # Find leaf to start
            start = 0
            for v in range(k):
                if len(adj_t[v]) == 1:
                    start = v
                    break
            
            # BFS order
            visited = [False] * k
            order_v = [start]
            visited[start] = True
            qi = 0
            while qi < len(order_v):
                cv = order_v[qi]
                qi += 1
                for nb in adj_t[cv]:
                    if not visited[nb]:
                        visited[nb] = True
                        order_v.append(nb)
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                if pos == k:
                    for u, v in edges_t:
                        mark(mapping[u], mapping[v])
                    mappings[t_idx] = mapping[:]
                    return True
                
                tv = order_v[pos]
                cands = []
                for hv in range(n_vertices):
                    if used_v[hv]:
                        continue
                    ok = True
                    for nb in adj_t[tv]:
                        if mapping[nb] != -1:
                            if not is_avail(hv, mapping[nb]):
                                ok = False
                                break
                    if ok:
                        cands.append(hv)
                
                if not cands:
                    return False
                
                local_rng.shuffle(cands)
                for hv in cands:
                    mapping[tv] = hv
                    used_v[hv] = True
                    if backtrack(pos + 1):
                        return True
                    used_v[hv] = False
                    mapping[tv] = -1
                return False
            
            return backtrack(0)

        for t_idx in order:
            if time.time() > deadline:
                break
            if not pack_tree(t_idx):
                break
        
        score = sum(len(tree_edges[i]) for i in range(n_trees) if mappings[i] is not None)
        if score > best_score:
            best_score = score
            best_mappings = [m[: if m is not None else None for m in mappings]
        
        return score == 190

    seed = 0
    while time.time() < deadline:
        if try_packing(seed):
            break
        seed += 1
    
    return [m if m is None else [int(v) for v in m] for m in best_mappings]
