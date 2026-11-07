import time
import random
import sys

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
    deadline = start_time + 145  # 5s margin before 150s limit
    
    trees = instance_trees(20, 611)
    
    # Pre-compute adjacency and degrees for each tree
    tree_adj = []
    tree_degrees = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_degrees.append(deg)
    
    # Generate multiple orderings to try
    orderings = []
    # Descending size (largest first)
    desc_order = list(range(20))
    desc_order.sort(key=lambda x: -(x + 1))
    orderings.append(desc_order)
    # Ascending size (smallest first)
    asc_order = list(range(20))
    asc_order.sort(key=lambda x: x + 1)
    orderings.append(asc_order)
    # Several random shuffles
    for s in range(10):
        r = random.Random(1000 + s)
        shuffled = list(range(20))
        r.shuffle(shuffled)
        orderings.append(shuffled)
    
    # State
    used_edges = set()
    mappings = [None] * 20
    
    best_score = 0
    best_mappings = [None] * 20
    
    rng = random.Random(12345)
    
    def update_best():
        nonlocal best_score, best_mappings
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += len(trees[i])
        if score > best_score:
            best_score = score
            for i in range(20):
                best_mappings[i] = mappings[i][:] if mappings[i] else None
    
    def embed_tree(tree_idx, mapping, used_hosts_in_tree):
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Find unmapped vertex with highest constraint:
        # 1. Most mapped neighbors
        # 2. Then highest degree in the tree
        best_vertex = -1
        best_mapped_neighbors = -1
        best_deg = -1
        for i in range(k):
            if mapping[i] != -1:
                continue
            mapped_neighbors = 0
            for neighbor in adj_t[i]:
                if mapping[neighbor] != -1:
                    mapped_neighbors += 1
            if (mapped_neighbors > best_mapped_neighbors or 
                (mapped_neighbors == best_mapped_neighbors and tree_degrees[tree_idx][i] > best_deg)):
                best_vertex = i
                best_mapped_neighbors = mapped_neighbors
                best_deg = tree_degrees[tree_idx][i]
                
        if best_vertex == -1:
            # All mapped - verify edge disjointness
            for u in range(k):
                hu = mapping[u]
                for v in adj_t[u]:
                    if u < v:
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
            # Commit
            for u in range(k):
                hu = mapping[u]
                for v in adj_t[u]:
                    if u < v:
                        hv = mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
            mappings[tree_idx] = mapping[:]
            return True

        u = best_vertex
        valid_candidates = []
        for h in range(20):
            if h in used_hosts_in_tree:
                continue
            valid = True
            for neighbor in adj_t[u]:
                if mapping[neighbor] != -1:
                    hn = mapping[neighbor]
                    e = (min(h, hn), max(h, hn))
                    if e in used_edges:
                        valid = False
                        break
            if valid:
                valid_candidates.append(h)
                
        if not valid_candidates:
            return False
            
        rng.shuffle(valid_candidates)
        
        for h in valid_candidates:
            mapping[u] = h
            used_hosts_in_tree.add(h)
            if embed_tree(tree_idx, mapping, used_hosts_in_tree):
                return True
            used_hosts_in_tree.remove(h)
            mapping[u] = -1
            
        return False

    def solve(order):
        nonlocal best_score, best_mappings
        if len(order) == 0:
            update_best()
            return True
        
        if time.time() > deadline:
            return False
            
        t_idx = order[0]
        k = t_idx + 1
        
        mapping = [-1] * k
        used_hosts = set()
        
        if embed_tree(t_idx, mapping, used_hosts):
            if solve(order[1:]):
                return True
        
        # If we couldn't embed this tree, try skipping it (for partial solutions)
        # But we need to be careful: only skip if we're in "best partial" mode
        # For now, just return False to backtrack
        return False

    # Run solver with multiple restarts and orderings
    attempts = 0
    for ord_idx in range(len(orderings)):
        if time.time() > deadline:
            break
        order = orderings[ord_idx]
        
        # Multiple restarts per ordering
        max_restarts_per_order = 50
        for restart in range(max_restarts_per_order):
            if time.time() > deadline:
                break
            
            # Reset state
            used_edges = set()
            mappings = [None] * 20
            rng = random.Random(12345 + attempts)
            
            success = solve(order)
            attempts += 1
            
            if success and best_score == 190:
                break
                
            # Track partial progress
            update_best()
            
            if best_score == 190:
                break
        
        if best_score == 190:
            break

    # Final polish: if we have a partial solution, try to embed remaining trees
    if best_score < 190 and time.time() < deadline:
        # Try to add more trees to the best partial solution
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Start from best known partial solution
            used_edges = set()
            mappings = [None] * 20
            for i in range(20):
                if best_mappings[i] is not None:
                    mappings[i] = best_mappings[i][:]
                    # Add its edges
                    k = i + 1
                    for u in range(k):
                        hu = mappings[i][u]
                        for v in tree_adj[i][u]:
                            if u < v:
                                hv = mappings[i][v]
                                used_edges.add((min(hu, hv), max(hu, hv)))
            
            # Try to embed each unembedded tree
            rng = random.Random(99999 + attempt)
            improved = False
            for t_idx in range(20):
                if mappings[t_idx] is not None:
                    continue
                if time.time() > deadline:
                    break
                k = t_idx + 1
                mapping = [-1] * k
                used_hosts = set()
                # Pre-fill hosts already used by other trees? No, vertices can be shared.
                # But edges must be disjoint.
                if embed_tree(t_idx, mapping, used_hosts):
                    improved = True
                    update_best()
                    if best_score == 190:
                        break

    # Ensure return type is correct
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    return result
