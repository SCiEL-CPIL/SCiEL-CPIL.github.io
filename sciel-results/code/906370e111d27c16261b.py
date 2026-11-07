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
    deadline = start_time + 140

    trees = instance_trees(20, 611)
    
    # Pre-compute adjacency and degrees for each tree
    tree_adj = []
    tree_degrees = []
    tree_edges = []
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
        tree_edges.append(edges)
        
    # Sort trees by size descending (largest first)
    order = list(range(20))
    order.sort(key=lambda x: -(x + 1))
    
    # State
    used_edges = set()
    mappings = [None] * 20
    host_remaining_degree = [19] * 20
    
    best_score = 0
    best_mappings = [None] * 20
    
    rng = random.Random(12345)
    
    def check_edges_validity(tree_idx, mapping):
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        for u in range(k):
            hu = mapping[u]
            for v in adj_t[u]:
                if u < v:
                    hv = mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        return False
        return True

    def embed_tree(tree_idx, mapping, used_hosts_in_tree, depth=0):
        if time.time() > deadline:
            return False
            
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Find unmapped vertex with highest constraint
        best_vertex = -1
        max_mapped_neighbors = -1
        max_deg = -1
        for i in range(k):
            if mapping[i] != -1:
                continue
            mapped_neighbors = 0
            for neighbor in adj_t[i]:
                if mapping[neighbor] != -1:
                    mapped_neighbors += 1
            if best_vertex == -1 or mapped_neighbors > max_mapped_neighbors or (mapped_neighbors == max_mapped_neighbors and tree_degrees[tree_idx][i] > max_deg):
                max_mapped_neighbors = mapped_neighbors
                max_deg = tree_degrees[tree_idx][i]
                best_vertex = i
                
        if best_vertex == -1:
            # All mapped
            if check_edges_validity(tree_idx, mapping):
                for u in range(k):
                    hu = mapping[u]
                    for v in adj_t[u]:
                        if u < v:
                            hv = mapping[v]
                            used_edges.add((min(hu, hv), max(hu, hv)))
                            host_remaining_degree[hu] -= 1
                            host_remaining_degree[hv] -= 1
                mappings[tree_idx] = mapping[:]
                return True
            return False

        u = best_vertex
        valid_candidates = []
        for h in range(20):
            if h in used_hosts_in_tree:
                continue
            if host_remaining_degree[h] <= 0:
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
                
        rng.shuffle(valid_candidates)
        
        for h in valid_candidates:
            mapping[u] = h
            used_hosts_in_tree.add(h)
            
            # Tentatively decrement degree
            # Note: We only decrement degree when the edge is finalized in check_edges_validity?
            # No, we should decrement immediately for pruning?
            # But we don't know the neighbors' hosts yet.
            # The degree constraint is on the HOST vertex.
            # Each time we map a vertex u to h, we are "reserving" degree for h.
            # But u has multiple neighbors. We don't know how many edges h will be incident to until the whole tree is mapped.
            # Actually, the degree of h in the final packing is the sum of degrees of all tree vertices mapped to h.
            # So we can't just decrement by 1 per vertex mapping.
            # We must check the final degree.
            # However, we can prune if the *potential* degree exceeds 19.
            # For now, let's stick to edge disjointness and check degree at the end?
            # Or better: When we map u to h, we add deg_t[u] to the "used degree" of h?
            # Yes! The degree of h in the host graph used by this tree is sum of deg_t[u] for all u mapped to h.
            # We can maintain `host_used_degree` and check if it exceeds 19.
            
            # Let's implement `host_used_degree`
            pass # Placeholder, will refactor below
            
            if embed_tree(tree_idx, mapping, used_hosts_in_tree, depth + 1):
                return True
            used_hosts_in_tree.remove(h)
            mapping[u] = -1
            
        return False

    def solve(idx_in_order):
        nonlocal best_score, best_mappings
        if idx_in_order == 20:
            score = 190
            if score > best_score:
                best_score = score
                for i in range(20):
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

    # Run solver with multiple restarts
    attempts = 0
    while time.time() < deadline:
        # Reset state
        used_edges = set()
        mappings = [None] * 20
        host_remaining_degree = [19] * 20 # Reset
        rng = random.Random(12345 + attempts)
        
        success = solve(0)
        attempts += 1
        
        if success:
            break
            
        # Calculate score of current partial packing
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += len(tree_edges[i])
                
        if score > best_score:
            best_score = score
            for i in range(20):
                best_mappings[i] = mappings[i][:] if mappings[i] else None
                
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
