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
    
    # Precompute adjacency and degree for each tree
    tree_adj = []
    tree_deg = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Smallest-first packing order (measured improvement)
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]), reverse=False)
    
    # Precompute all_edges as a set for fast lookup
    # Also precompute neighbor lists for each vertex in K20
    k20_neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            k20_neighbors[i].add(j)
            k20_neighbors[j].add(i)
    
    for restart in range(5000):
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
            deg = tree_deg[t_idx]
            
            # Pick a random root - prefer high-degree nodes
            if rng.random() < 0.7:
                # Pick a high-degree node as root
                max_deg = max(deg)
                high_deg_nodes = [i for i in range(k) if deg[i] == max_deg]
                root = rng.choice(high_deg_nodes)
            else:
                root = rng.randint(0, k - 1)
            
            # Try to embed using backtracking with forward checking
            mapping = [-1] * k
            used_vertices = [False] * n
            
            def embed_dfs(node, parent):
                nonlocal available, used_vertices
                if time.time() > deadline:
                    return False
                
                if parent != -1:
                    parent_gv = mapping[parent]
                    # Find available neighbors of parent_gv using precomputed neighbors
                    candidates = []
                    for v in k20_neighbors[parent_gv]:
                        if v != parent_gv and not used_vertices[v]:
                            e = (min(parent_gv, v), max(parent_gv, v))
                            if e in available:
                                candidates.append(v)
                else:
                    candidates = [v for v in range(n) if not used_vertices[v]]
                
                if not candidates:
                    return False
                
                # Score candidates: prefer vertices with FEWER available edges
                # (most-constrained-first heuristic)
                def candidate_score(v):
                    count = 0
                    for u in k20_neighbors[v]:
                        if used_vertices[u]:
                            continue
                        e = (min(v, u), max(v, u))
                        if e in available:
                            count += 1
                    return count  # Lower score = fewer options = more constrained
                
                candidates.sort(key=candidate_score)
                rng.shuffle(candidates)  # Add some randomness to break ties
                
                for gv in candidates:
                    mapping[node] = gv
                    used_vertices[gv] = True
                    ok = True
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if not embed_dfs(neighbor, node):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    mapping[node] = -1
                    used_vertices[gv] = False
                
                return False
            
            if not embed_dfs(root, -1):
                # Failed to embed this tree
                success = False
                break
            
            # Update available edges
            for u, v in trees[t_idx]:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                available.remove(e)
                packed_edges += 1
                
            current_solution[t_idx] = mapping[:]
        
        if success:
            # Found a complete packing
            return current_solution
    
    # If no complete solution found, return a fallback
    result = [None] * n
    result[0] = [0]
    result[1] = [0, 1]
    
    return result
