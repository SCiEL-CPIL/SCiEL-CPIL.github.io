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
    tree_size = []
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
        tree_size.append(k)
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Sort trees by size descending for packing order
    # Larger trees are more constrained, so embed them first
    order = list(range(n))
    order.sort(key=lambda x: tree_size[x], reverse=True)
    
    # Precompute for each tree: which vertices have high degree (good roots)
    tree_roots = []
    for t_idx in range(n):
        k = tree_size[t_idx]
        if k <= 1:
            tree_roots.append([0])
            continue
        deg = tree_deg[t_idx]
        max_deg = max(deg)
        high_deg_nodes = [i for i in range(k) if deg[i] == max_deg]
        tree_roots.append(high_deg_nodes)
    
    best_solution = None
    best_value = 0
    
    # Try multiple random restarts with different random seeds
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
            
            # Pick a root - prefer high-degree nodes
            roots = tree_roots[t_idx]
            root = rng.choice(roots)
            
            # Try to embed using backtracking with forward checking
            mapping = [-1] * k
            used_vertices = [False] * n
            failed_states = set()  # Cache of (node, parent_gv) that led to failure
            
            # Precompute adjacency list for available edges for faster lookup
            avail_adj = [set() for _ in range(n)]
            for (a, b) in available:
                avail_adj[a].add(b)
                avail_adj[b].add(a)
            
            def embed_dfs(node, parent):
                nonlocal available, used_vertices, avail_adj
                if time.time() > deadline:
                    return False
                
                if parent != -1:
                    parent_gv = mapping[parent]
                    # Check if this state was already tried and failed
                    state_key = (node, parent_gv)
                    if state_key in failed_states:
                        return False
                    
                    # Find available neighbors of parent_gv
                    candidates = []
                    for v in avail_adj[parent_gv]:
                        if not used_vertices[v]:
                            candidates.append(v)
                else:
                    candidates = [v for v in range(n) if not used_vertices[v]]
                
                if not candidates:
                    if parent != -1:
                        failed_states.add((node, mapping[parent]))
                    return False
                
                # Sort candidates: prefer vertices with fewer available edges
                # (to constrain the search space early)
                # Use precomputed avail_adj for O(1) count
                candidates.sort(key=lambda v: len(avail_adj[v]))
                
                for gv in candidates:
                    mapping[node] = gv
                    used_vertices[gv] = True
                    
                    # Update avail_adj for this assignment
                    # Save old state for restoration
                    old_avail = {v: set(avail_adj[v]) for v in range(n)}
                    
                    ok = True
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if not embed_dfs(neighbor, node):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    # Restore avail_adj
                    for v in range(n):
                        avail_adj[v] = old_avail[v]
                    
                    mapping[node] = -1
                    used_vertices[gv] = False
                    # Remember this failed state
                    if parent != -1:
                        failed_states.add((node, gv))
                
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
