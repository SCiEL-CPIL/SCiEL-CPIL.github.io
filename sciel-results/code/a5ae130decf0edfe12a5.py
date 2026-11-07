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
    deadline = start_time + 145.0
    
    trees = instance_trees(20, 611)
    num_trees = 20
    
    # Precompute tree data
    tree_data = []
    for i in range(num_trees):
        k = i + 1
        if k == 1:
            tree_data.append({'edges': [], 'adj': [[]], 'size': 1, 'deg': [0]})
        elif k == 2:
            tree_data.append({'edges': [(0, 1)], 'adj': [[1], [0]], 'size': 2, 'deg': [1, 1]})
        else:
            edges = trees[i]
            adj = [[] for _ in range(k)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            deg = [len(adj[v]) for v in range(k)]
            tree_data.append({'edges': edges, 'adj': adj, 'size': k, 'deg': deg})
    
    best_score = 0
    best_mappings = [None] * 20
    
    def try_packing(order, rng, time_limit):
        """Try to pack all trees in the given order. Returns (mappings, used_edges) or (None, None)."""
        used_edges = set()
        mappings = [None] * 20
        
        for tree_idx in order:
            if time.time() > time_limit:
                return None, None
            
            k = tree_data[tree_idx]['size']
            if k == 1:
                # For k=1, any host vertex works
                mappings[tree_idx] = [0]
                continue
            
            if k == 2:
                # Find any unused edge
                found = False
                for u in range(20):
                    for v in range(u + 1, 20):
                        if (u, v) not in used_edges:
                            mappings[tree_idx] = [u, v]
                            used_edges.add((u, v))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return None, None
                continue
            
            # For larger trees: use greedy + limited backtracking
            edges = tree_data[tree_idx]['edges']
            adj = tree_data[tree_idx]['adj']
            deg = tree_data[tree_idx]['deg']
            
            # Order tree vertices by degree (high to low) for embedding
            # But we need to maintain the tree structure
            # Use a BFS/DFS order starting from highest degree vertex
            start_v = max(range(k), key=lambda v: deg[v])
            visited = set([start_v])
            queue = [start_v]
            embed_order = []
            while queue:
                u = queue.pop(0)
                embed_order.append(u)
                for w in adj[u]:
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
            
            # Greedy assignment with backtracking
            mapping = {}
            def assign_vertex(v, host):
                mapping[v] = host
                # Check all edges from v to already-assigned neighbors
                for u in adj[v]:
                    if u in mapping:
                        e_key = (min(host, mapping[u]), max(host, mapping[u]))
                        if e_key in used_edges:
                            return False
                return True
            
            def backtrack(idx):
                if idx == len(embed_order):
                    # All assigned, verify all edges
                    for u, v in edges:
                        hu, hv = mapping[u], mapping[v]
                        e_key = (min(hu, hv), max(hu, hv))
                        if e_key in used_edges:
                            return False
                    return True
                
                v = embed_order[idx]
                # Try all host vertices, ordered by fewest conflicts
                candidates = []
                for h in range(20):
                    if h in mapping.values():
                        continue
                    # Count conflicts
                    conflicts = 0
                    for u in adj[v]:
                        if u in mapping:
                            e_key = (min(h, mapping[u]), max(h, mapping[u]))
                            if e_key in used_edges:
                                conflicts += 1
                    candidates.append((conflicts, h))
                
                candidates.sort()
                
                for _, h in candidates:
                    if assign_vertex(v, h):
                        if backtrack(idx + 1):
                            return True
                    del mapping[v]
                
                return False
            
            # Try greedy first (no backtracking)
            mapping = {}
            success = True
            for v in embed_order:
                best_h = None
                min_conflicts = 999
                for h in range(20):
                    if h in mapping.values():
                        continue
                    conflicts = 0
                    for u in adj[v]:
                        if u in mapping:
                            e_key = (min(h, mapping[u]), max(h, mapping[u]))
                            if e_key in used_edges:
                                conflicts += 1
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_h = h
                    if conflicts == 0:
                        break
                
                if best_h is None:
                    success = False
                    break
                
                mapping[v] = best_h
            
            # Verify
            if success:
                valid = True
                for u, v in edges:
                    e_key = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    if e_key in used_edges:
                        valid = False
                        break
            
            if not valid:
                # Try backtracking
                mapping = {}
                if not backtrack(0):
                    return None, None
            
            mappings[tree_idx] = [mapping[v] for v in range(k)]
            for u, v in edges:
                used_edges.add((min(mapping[u], mapping[v]), max(mapping[u], mapping[v])))
        
        return mappings, used_edges
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Largest to smallest
    strategies.append(list(range(19, -1, -1)))
    
    # Strategy 2: Smallest to largest
    strategies.append(list(range(20)))
    
    # Strategy 3: Random orderings
    rng_init = Random(611)
    for _ in range(5):
        order = list(range(20))
        rng_init.shuffle(order)
        strategies.append(order)
    
    # Try each strategy
    for si, order in enumerate(strategies):
        if time.time() > deadline:
            break
        
        rng = Random(si * 1000 + 611)
        mappings, used_edges = try_packing(order, rng, deadline)
        
        if mappings is not None:
            score = 0
            for i in range(20):
                if mappings[i] is not None:
                    score += len(tree_data[i]['edges'])
            
            if score > best_score:
                best_score = score
                best_mappings = [m[:] if m else None for m in mappings]
            
            if best_score == 190:
                break
    
    # If still not perfect, try more random restarts
    if best_score < 190:
        for restart in range(200):
            if time.time() > deadline:
                break
            
            rng = Random(restart * 7919 + 611)
            order = list(range(20))
            rng.shuffle(order)
            
            mappings, used_edges = try_packing(order, rng, deadline)
            
            if mappings is not None:
                score = 0
                for i in range(20):
                    if mappings[i] is not None:
                        score += len(tree_data[i]['edges'])
                
                if score > best_score:
                    best_score = score
                    best_mappings = [m[:] if m else None for m in mappings]
                
                if best_score == 190:
                    break
    
    # Build final result
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
    
    return final_result
