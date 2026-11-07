def construct():
    import time
    import random
    
    # Deadline: 120s
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    # All possible edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
            
    # Order of packing: largest first
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]), reverse=True)
    
    # Precompute adjacency for all trees
    tree_adjs = []
    for t_idx in range(n):
        k = t_idx + 1
        if k <= 2:
            tree_adjs.append([])
            continue
        edges = trees[t_idx]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)

    # Define strategies
    # We will try a sequence of deterministic and seeded random strategies
    strategies = []
    strategies.append(('det_asc', None))
    strategies.append(('det_desc', None))
    for s in range(1, 21):
        strategies.append(('rand', s))

    for strat_type, strat_seed in strategies:
        if time.time() > deadline:
            break
            
        # Set up RNG if random
        if strat_type == 'rand':
            rng = random.Random(strat_seed)
        else:
            rng = None

        available = set(all_edges)
        current_solution = [None] * n
        success = True
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                if strat_type == 'rand':
                    v = rng.randint(0, n-1)
                else:
                    v = 0
                current_solution[t_idx] = [v]
                continue
                
            edges = trees[t_idx]
            adj = tree_adjs[t_idx]
            
            if strat_type == 'rand':
                root = rng.randint(0, k-1)
            else:
                root = 0

            mapping = [-1] * k
            
            def local_dfs(node, parent):
                nonlocal success
                if time.time() > deadline:
                    return False
                
                if parent != -1:
                    parent_gv = mapping[parent]
                    candidates = [v for v in range(n) if (min(parent_gv, v), max(parent_gv, v)) in available]
                else:
                    candidates = list(range(n))
                
                if strat_type == 'det_asc':
                    candidates.sort()
                elif strat_type == 'det_desc':
                    candidates.sort(reverse=True)
                elif strat_type == 'rand':
                    rng.shuffle(candidates)
                
                for gv in candidates:
                    if gv in mapping:
                        continue
                    
                    mapping[node] = gv
                    ok = True
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if not local_dfs(neighbor, node):
                            ok = False
                            break
                    if ok:
                        return True
                    mapping[node] = -1
                return False
            
            if not local_dfs(root, -1):
                success = False
                break
            
            for u, v in edges:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                if e in available:
                    available.remove(e)
                else:
                    success = False
                    break
                    
            if not success:
                break
                
            current_solution[t_idx] = mapping[:]
            
        if success:
            return current_solution

    # Fallback
    return [None] * 19 + [list(range(20))]
