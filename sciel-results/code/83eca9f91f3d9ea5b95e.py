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
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20
    
    # Precompute tree adjacencies and degrees
    tree_adj = []
    tree_edges = []
    tree_deg = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        tree_adj.append(adj_list)
        tree_edges.append(edges)
        tree_deg.append([len(adj_list[v]) for v in range(k)])
    
    # Sort trees by size descending, then by max degree descending
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda x: (-(x + 1), -max(tree_deg[x]) if tree_deg[x] else 0))
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def try_packing(attempt_seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * n_trees
        
        # Host graph adjacency (K20 is complete, so all edges exist initially)
        # We track used edges to avoid conflicts.
        
        def pack_tree(tree_order_idx):
            t_idx = tree_indices[tree_order_idx]
            k = t_idx + 1
            adj_t = tree_adj[t_idx]
            edges_t = tree_edges[t_idx]
            
            if k == 1:
                # Find any available vertex
                for hv in range(n_vertices):
                    if not any((min(hv, u), max(hv, u)) in used_edges for u in range(n_vertices) if u != hv):
                        # For k=1, no edges are added, so any vertex works
                        mappings[t_idx] = [hv]
                        return True
                mappings[t_idx] = [0] # Fallback
                return True
            
            # Build a static ordering of tree vertices to guide search
            # We will use dynamic variable ordering (MCV) during backtracking
            
            # Initial mapping state
            mapping = [-1] * k
            used_verts = set()
            
            # Counters for time check
            nodes_visited = 0
            max_nodes = 500000 # Limit per tree to avoid hanging one tree
            
            def backtrack(pos, depth):
                nonlocal nodes_visited
                nodes_visited += 1
                if nodes_visited > max_nodes:
                    return False
                if time.time() > deadline:
                    return False
                
                if pos == k:
                    # Verify and commit
                    for u, v in edges_t:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
                    for u, v in edges_t:
                        hu = mapping[u]
                        hv = mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    mappings[t_idx] = mapping[:]
                    return True
                
                # Find the next tree vertex to map
                # MCV: Pick unmapped vertex with fewest valid host candidates
                best_tv = -1
                best_candidates = None
                min_cand_count = float('inf')
                
                for tv in range(k):
                    if mapping[tv] != -1:
                        continue
                    
                    candidates = []
                    for hv in range(n_vertices):
                        if hv in used_verts:
                            continue
                        # Check edges to already mapped neighbors
                        valid = True
                        for nb in adj_t[tv]:
                            if mapping[nb] != -1:
                                hn = mapping[nb]
                                e = (min(hv, hn), max(hv, hn))
                                if e in used_edges:
                                    valid = False
                                    break
                        if valid:
                            candidates.append(hv)
                    
                    if not candidates:
                        return False # Dead end
                    
                    if len(candidates) < min_cand_count:
                        min_cand_count = len(candidates)
                        best_tv = tv
                        best_candidates = candidates
                        if min_cand_count == 1:
                            break # Can't get better than 1
                
                if best_tv == -1:
                    return False # Should not happen if pos < k
                
                # LCV: Order candidates by how many options they leave for neighbors
                def count_options(hv):
                    count = 0
                    for nb in adj_t[best_tv]:
                        if mapping[nb] == -1:
                            for hnb in range(n_vertices):
                                if hnb not in used_verts and hnb != hv:
                                    e = (min(hv, hnb), max(hv, hnb))
                                    if e not in used_edges:
                                        count += 1
                    return count
                
                # Sort candidates: Fewer options first (More constrained)
                # Actually, LCV says pick the value that leaves MOST options for others.
                # But in CSP, MCV picks variable with fewest values.
                # For values, LCV is often better: pick value that prunes least.
                # However, for packing, "conflict" is the main issue.
                # Let's try sorting by count_options DESC (keep most options)
                # Or ASC (fail fast)?
                # Let's stick with a simple heuristic: try random or sorted by degree of host?
                # Since K20 is symmetric, host degree is same.
                # Let's try sorting by count_options DESC.
                
                # Add randomness to escape local minima
                if local_rng.random() < 0.5:
                    local_rng.shuffle(best_candidates)
                else:
                    best_candidates.sort(key=lambda hv: -count_options(hv))
                
                for hv in best_candidates:
                    mapping[best_tv] = hv
                    used_verts.add(hv)
                    if backtrack(pos + 1, depth + 1):
                        return True
                    used_verts.remove(hv)
                    mapping[best_tv] = -1
                
                return False
            
            # Try different starting vertices for the root?
            # With MCV, we don't strictly need a fixed root order, 
            # but the first variable picked is the one with fewest candidates.
            # In an empty graph, all vertices have 20 candidates.
            # So it picks tv=0 (or whichever comes first in loop).
            # We can bias the initial choice.
            
            # To increase diversity, we can try different random seeds for the shuffle logic inside
            # or simply restart the whole tree packing with a different seed.
            
            return backtrack(0, 0)
        
        # Main loop
        success = True
        for i in range(n_trees):
            if time.time() > deadline:
                success = False
                break
            if pack_tree(i):
                pass
            else:
                success = False
                break
        
        # Calculate score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])
        
        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
        
        return success
    
    # Run multiple attempts with different seeds
    attempt = 0
    while time.time() < deadline:
        try_packing(attempt)
        attempt += 1
        if best_score == 190:
            break
    
    return best_mappings
