import time
import sys
from random import Random

sys.setrecursionlimit(100000)

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
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency and degree for each tree
    tree_adj = []
    tree_deg = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in trees[i]:
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
    
    # Sort trees by size descending for packing order
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]), reverse=True)
    
    best_solution = None
    best_value = 0
    
    # Try multiple random restarts with different random seeds
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 7919 + 13)
        
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
            
            # Try to embed using backtracking with forward checking
            mapping = [-1] * k
            used_vertices = [False] * n
            
            def embed_dfs():
                nonlocal available, used_vertices
                if time.time() > deadline:
                    return False
                
                # MRV heuristic: pick unassigned node with most assigned neighbors
                best_node = -1
                max_assigned = -1
                for v in range(k):
                    if mapping[v] != -1:
                        continue
                    cnt = 0
                    for nb in adj[v]:
                        if mapping[nb] != -1:
                            cnt += 1
                    if cnt > max_assigned:
                        max_assigned = cnt
                        best_node = v
                
                if best_node == -1:
                    # All assigned
                    return True
                
                # Find candidates
                candidates = []
                for h in range(n):
                    if used_vertices[h]:
                        continue
                    # Check edge constraints with assigned neighbors
                    valid = True
                    for nb in adj[best_node]:
                        if mapping[nb] != -1:
                            e = (min(h, mapping[nb]), max(h, mapping[nb]))
                            if e not in available:
                                valid = False
                                break
                    if valid:
                        candidates.append(h)
                
                if not candidates:
                    return False
                
                # Score candidates: prefer vertices that leave more options for others
                # (i.e., vertices with fewer available edges to unassigned tree nodes' potential neighbors)
                def candidate_score(v):
                    # Count how many available edges incident to v go to vertices
                    # that could be assigned to unassigned tree nodes
                    count = 0
                    for u in range(n):
                        if u == v or used_vertices[u]:
                            continue
                        e = (min(v, u), max(v, u))
                        if e in available:
                            count += 1
                    return -count  # Prefer fewer remaining options (more constrained)
                
                candidates.sort(key=candidate_score)
                # Add some randomness to break ties
                rng.shuffle(candidates)
                
                for gv in candidates:
                    mapping[best_node] = gv
                    used_vertices[gv] = True
                    if embed_dfs():
                        return True
                    mapping[best_node] = -1
                    used_vertices[gv] = False
                
                return False
            
            if not embed_dfs():
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
