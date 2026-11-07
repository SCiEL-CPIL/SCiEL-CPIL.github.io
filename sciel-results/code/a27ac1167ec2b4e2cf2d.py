import time
import random

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
    deadline = start_time + 140  # 10s margin before 150s limit
    
    trees = instance_trees(20, 611)
    n_trees = 20
    
    # Pre-compute adjacency and degrees for each tree
    tree_adj = []
    tree_degrees = []
    for i in range(n_trees):
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
        
    # Sort trees by size descending (largest first) to constrain the search earlier
    # We need to map results back to original indices
    order = list(range(n_trees))
    order.sort(key=lambda x: -(x + 1)) # k = x+1
    
    # State
    used_edges = set()
    # mappings[i] will be list of host vertices for tree i, or None
    mappings = [None] * n_trees
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    # Use a deterministic seed for reproducibility of the search path if needed, 
    # but since we are aiming for a perfect packing, we want a robust solver.
    rng = random.Random(42)
    
    def get_remaining_capacity(host):
        """Count how many unused edges are incident to 'host' in K20."""
        count = 0
        for other in range(20):
            if other == host: continue
            e = (min(host, other), max(host, other))
            if e not in used_edges:
                count += 1
        return count

    def check_edges_validity(tree_idx, mapping):
        """Check if all edges of tree tree_idx are in available_edges."""
        k = tree_idx + 1
        for u in range(k):
            hu = mapping[u]
            for v in tree_adj[tree_idx][u]:
                if u < v:
                    hv = mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        return False
        return True

    def embed_tree(tree_idx, mapping, used_hosts_in_tree):
        """
        Recursively embed tree tree_idx.
        mapping: list of length k, -1 if unmapped.
        used_hosts_in_tree: set of host vertices already used by this tree.
        Returns True if success.
        """
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Find unmapped vertex with highest constraint (most mapped neighbors)
        # Tie breaker: higher degree in the tree
        best_vertex = -1
        max_mapped_neighbors = -1
        max_degree = -1
        for i in range(k):
            if mapping[i] != -1:
                continue
            mapped_neighbors = 0
            for neighbor in adj_t[i]:
                if mapping[neighbor] != -1:
                    mapped_neighbors += 1
            # Tie breaker: higher degree
            if mapped_neighbors > max_mapped_neighbors or (mapped_neighbors == max_mapped_neighbors and tree_degrees[tree_idx][i] > max_degree):
                max_mapped_neighbors = mapped_neighbors
                max_degree = tree_degrees[tree_idx][i]
                best_vertex = i
                
        if best_vertex == -1:
            # All mapped, check validity
            if check_edges_validity(tree_idx, mapping):
                # Commit
                for u in range(k):
                    hu = mapping[u]
                    for v in adj_t[u]:
                        if u < v:
                            hv = mapping[v]
                            used_edges.add((min(hu, hv), max(hu, hv)))
                mappings[tree_idx] = mapping[:]
                return True
            return False

        u = best_vertex
        
        # Candidates for host vertex
        candidates = list(range(20))
        # Filter candidates:
        # 1. Not in used_hosts_in_tree
        # 2. Edge to any mapped neighbor must be available
        
        valid_candidates = []
        for h in candidates:
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
                # Heuristic: Prefer hosts with lower remaining capacity? 
                # Or higher? Usually, we want to save flexible vertices for later.
                # However, for the current vertex, we just need *a* valid host.
                # Sorting by remaining capacity can help prune.
                valid_candidates.append((get_remaining_capacity(h), h))
                
        # Sort by remaining capacity ascending (use scarce resources first)
        valid_candidates.sort(key=lambda x: x[0])
        
        # Randomize tie-breaks slightly to avoid local minima if needed,
        # but for this specific instance, deterministic often works well.
        # Let's add a small random shuffle to the sorted list to vary the path.
        # rng.shuffle(valid_candidates) # Might break the heuristic benefit. 
        # Let's stick to heuristic first.
        
        for _, h in valid_candidates:
            mapping[u] = h
            used_hosts_in_tree.add(h)
            if embed_tree(tree_idx, mapping, used_hosts_in_tree):
                return True
            used_hosts_in_tree.remove(h)
            mapping[u] = -1
            
        return False

    def solve(idx_in_order):
        nonlocal best_score, best_mappings
        if idx_in_order == n_trees:
            # All packed
            score = 190
            if score > best_score:
                best_score = score
                for i in range(n_trees):
                    best_mappings[i] = mappings[i][:] if mappings[i] else None
            return True
        
        if time.time() > deadline:
            return False
            
        t_idx = order[idx_in_order]
        k = t_idx + 1
        
        mapping = [-1] * k
        used_hosts = set()
        
        if embed_tree(t_idx, mapping, used_hosts):
            return solve(idx_in_order + 1)
            
        return False

    # Run solver
    # Given the small size, a single deterministic pass with good heuristics should suffice.
    # If it fails, we can try random restarts.
    
    success = solve(0)
    
    if not success:
        # Fallback: Multiple random restarts
        attempts = 0
        while time.time() < deadline and not success:
            # Reset state
            used_edges = set()
            mappings = [None] * n_trees
            rng.seed(attempts) # Change search path
            success = solve(0)
            attempts += 1
            
            # Calculate score of current partial packing
            score = 0
            for i in range(n_trees):
                if mappings[i] is not None:
                    score += len(trees[i])
            if score > best_score:
                best_score = score
                for i in range(n_trees):
                    best_mappings[i] = mappings[i][:] if mappings[i] else None

    # Final score calculation
    final_score = 0
    for i in range(n_trees):
        if best_mappings[i] is not None:
            final_score += len(trees[i])
            
    return best_mappings

