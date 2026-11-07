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
    # We pack larger trees first as they are more constrained
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda x: (-(x + 1), -max(tree_deg[x]) if tree_deg[x] else 0))
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def try_packing(attempt_seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * n_trees
        
        # For the first tree, we can fix vertex 0 to host 0 to break symmetry
        # This reduces search space by factor of 20 for the largest tree
        # But we must be careful: if the first tree is T1 (k=1), it's trivial.
        # If T20 is first, fixing v0->0 is valid.
        
        def pack_tree(t_idx, time_limit=2.0):
            k = t_idx + 1
            adj_t = tree_adj[t_idx]
            edges_t = tree_edges[t_idx]
            start_tree_time = time.time()
            
            if k == 1:
                mappings[t_idx] = [0]
                return True
            
            # State for backtracking
            # mapping: list of size k, mapping[tree_v] = host_v or -1
            # used_verts: set of host vertices used by this tree
            # mapped_tv: set of tree vertices already mapped
            
            def backtrack(mapping, used_verts, mapped_tv):
                if time.time() > deadline or time.time() - start_tree_time > time_limit:
                    return False
                
                # Find the next tree vertex to map
                # Heuristic: MRV (Minimum Remaining Values)
                # Count available host vertices for each unmapped tree vertex
                best_v = -1
                best_avail = None
                best_count = 1000
                best_candidates = []
                
                for tv in range(k):
                    if tv in mapped_tv:
                        continue
                    
                    avail = []
                    for hv in range(n_vertices):
                        if hv in used_verts:
                            continue
                        # Check conflicts with mapped neighbors
                        conflict = False
                        for nb in adj_t[tv]:
                            if nb in mapped_tv:
                                hn = mapping[nb]
                                e = (min(hv, hn), max(hv, hn))
                                if e in used_edges:
                                    conflict = True
                                    break
                        if not conflict:
                            avail.append(hv)
                    
                    if len(avail) < best_count:
                        best_count = len(avail)
                        best_v = tv
                        best_candidates = avail
                        if best_count == 0:
                            return False
                        if best_count == 1:
                            break # Can't get better than 1
                
                if best_v == -1:
                    # All mapped
                    # Commit
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
                
                # Order candidates for best_v
                # Heuristic: Prefer host vertices that are less constrained? 
                # Or more central? 
                # Let's try random shuffle with bias towards high degree in remaining graph
                # Or just random for diversity
                local_rng.shuffle(best_candidates)
                
                for hv in best_candidates:
                    mapping[best_v] = hv
                    used_verts.add(hv)
                    mapped_tv.add(best_v)
                    
                    if backtrack(mapping, used_verts, mapped_tv):
                        return True
                    
                    mapped_tv.remove(best_v)
                    used_verts.remove(hv)
                    mapping[best_v] = -1
                
                return False
            
            # Initial setup
            # For the very first tree packed in this attempt, we can fix v0 to 0
            # To identify if it's the first, we can check if used_edges is empty
            # But `used_edges` is modified in place. 
            # Let's pass a flag or check if mappings are all None.
            # Simpler: just try all 20 options for v0, but for the first tree, 
            # we can fix v0=0 to save time.
            
            # Let's identify if this is the first tree to be packed
            # We can check if all mappings are None
            is_first = all(m is None for m in mappings)
            
            if is_first:
                # Fix tree vertex 0 to host 0
                mapping = [-1] * k
                mapping[0] = 0
                used_verts = {0}
                mapped_tv = {0}
                if backtrack(mapping, used_verts, mapped_tv):
                    return True
                return False
            else:
                # Try a limited number of host vertices for tree vertex 0
                # to reduce branching factor at the root
                host_candidates = list(range(n_vertices))
                local_rng.shuffle(host_candidates)
                # Limit to 5 candidates to speed up search
                host_candidates = host_candidates[:5]
                
                for h0 in host_candidates:
                    mapping = [-1] * k
                    mapping[0] = h0
                    used_verts = {h0}
                    mapped_tv = {0}
                    if backtrack(mapping, used_verts, mapped_tv):
                        return True
                return False
        
        # Main loop
        # Try to pack trees in sorted order
        # If a tree fails, we skip it and try the next one
        # This allows packing of other trees that might fit in the remaining space
        
        for i in range(n_trees):
            if time.time() > deadline:
                break
            t_idx = tree_indices[i]
            pack_tree(t_idx)
        
        # Calculate score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])
        
        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
        
        return best_score == 190
    
    # Run multiple attempts
    attempt = 0
    while time.time() < deadline:
        try_packing(attempt)
        attempt += 1
        if best_score == 190:
            break
    
    return best_mappings
