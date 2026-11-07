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
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency for each tree
    tree_adj = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Sort trees by size ascending (smallest first) for packing order
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]))
    
    best_solution = None
    best_value = 0
    
    # Precompute tree degrees for heuristic ordering
    tree_degrees = []
    for adj in tree_adj:
        tree_degrees.append([len(a) for a in adj])
    
    # Try multiple random restarts with different random seeds
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 678)
        
        available = set(all_edges)
        current_solution = [None] * n
        success = True
        packed_edges = 0
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                # T1: Single vertex, no edges
                v = rng.randint(0, n - 1)
                current_solution[t_idx] = [v]
                continue
            
            # Build adjacency for this tree
            adj = tree_adj[t_idx]
            deg = tree_degrees[t_idx]
            
            # Pick a high-degree node as root for better embedding
            max_deg = max(deg)
            candidates_root = [i for i in range(k) if deg[i] == max_deg]
            root = rng.choice(candidates_root)
            
            # Try to embed using backtracking with heuristics
            mapping = [-1] * k
            used_vertices = set()
            
            def embed_dfs(node, parent):
                nonlocal available
                if time.time() > deadline:
                    return False
                
                if parent != -1:
                    parent_gv = mapping[parent]
                    # Find available neighbors of parent_gv
                    candidates = []
                    for v in range(n):
                        if v in used_vertices:
                            continue
                        e = (min(parent_gv, v), max(parent_gv, v))
                        if e in available:
                            candidates.append(v)
                else:
                    candidates = list(range(n))
                
                if not candidates:
                    return False
                
                # Heuristic: prefer vertices with more available edges (more flexibility)
                # But also consider the degree of the tree node
                node_deg = deg[node]
                
                # For high-degree tree nodes, prefer host vertices with more remaining edges
                if node_deg >= 3:
                    candidates.sort(key=lambda v: -sum(1 for w in range(n) if w != v and (min(v, w), max(v, w)) in available))
                else:
                    rng.shuffle(candidates)
                
                for gv in candidates:
                    if gv in used_vertices:
                        continue
                    
                    mapping[node] = gv
                    used_vertices.add(gv)
                    ok = True
                    
                    # Order children: highest degree first for better pruning
                    children = [nb for nb in adj[node] if nb != parent]
                    children.sort(key=lambda x: -deg[x])
                    
                    for neighbor in children:
                        if not embed_dfs(neighbor, node):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    used_vertices.remove(gv)
                    mapping[node] = -1
                
                return False
            
            if not embed_dfs(root, -1):
                # Failed to embed this tree
                success = False
                break
            
            # Update available edges
            for u, v in trees[t_idx]:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                if e in available:
                    available.remove(e)
                    packed_edges += 1
                
            current_solution[t_idx] = mapping[:]
        
        if success:
            # Found a complete packing
            return current_solution
    
    # If no complete solution found, return a fallback
    # Try to pack as many trees as possible
    result = [None] * n
    available = set(all_edges)
    
    # Try in order, skip failures
    for t_idx in order:
        k = t_idx + 1
        if k == 1:
            result[t_idx] = [0]
            continue
        
        adj = tree_adj[t_idx]
        deg = tree_degrees[t_idx]
        max_deg = max(deg)
        candidates_root = [i for i in range(k) if deg[i] == max_deg]
        root = candidates_root[0]
        
        mapping = [-1] * k
        used_vertices = set()
        
        def embed_dfs_fb(node, parent):
            if parent != -1:
                parent_gv = mapping[parent]
                candidates = [v for v in range(n) if v not in used_vertices and (min(parent_gv, v), max(parent_gv, v)) in available]
            else:
                candidates = list(range(n))
            
            if not candidates:
                return False
            
            for gv in candidates:
                if gv in used_vertices:
                    continue
                mapping[node] = gv
                used_vertices.add(gv)
                ok = True
                children = [nb for nb in adj[node] if nb != parent]
                for neighbor in children:
                    if not embed_dfs_fb(neighbor, node):
                        ok = False
                        break
                if ok:
                    return True
                used_vertices.remove(gv)
                mapping[node] = -1
            return False
        
        if embed_dfs_fb(root, -1):
            for u, v in trees[t_idx]:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                if e in available:
                    available.remove(e)
            result[t_idx] = mapping[:]
    
    return result

# Ensure Random is available
from random import Random
