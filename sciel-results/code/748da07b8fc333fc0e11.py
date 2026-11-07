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
    deadline = start_time + 140  # 10s margin before 150s limit
    
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
        
    # Sort trees by size descending (largest first)
    order = list(range(20))
    order.sort(key=lambda x: -(x + 1))
    
    # State
    used_edges = set()
    mappings = [None] * 20
    
    best_score = 0
    best_mappings = [None] * 20
    
    rng = random.Random(12345)
    
    def check_edges_validity(tree_idx, mapping):
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

    def has_valid_host_for_vertex(tree_idx, v, mapping, used_hosts_in_tree):
        """Check if vertex v has at least one valid host."""
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        for h in range(20):
            if h in used_hosts_in_tree:
                continue
            valid = True
            for neighbor in adj_t[v]:
                if mapping[neighbor] != -1:
                    hn = mapping[neighbor]
                    e = (min(h, hn), max(h, hn))
                    if e in used_edges:
                        valid = False
                        break
            if valid:
                return True
        return False

    def embed_tree(tree_idx, mapping, used_hosts_in_tree):
        if time.time() > deadline:
            return False
            
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Find unmapped vertex with highest constraint (most mapped neighbors, then highest degree)
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
            deg_i = tree_degrees[tree_idx][i]
            if best_vertex == -1:
                best_vertex = i
                max_mapped_neighbors = mapped_neighbors
                max_degree = deg_i
            elif mapped_neighbors > max_mapped_neighbors:
                best_vertex = i
                max_mapped_neighbors = mapped_neighbors
                max_degree = deg_i
            elif mapped_neighbors == max_mapped_neighbors and deg_i > max_degree:
                best_vertex = i
                max_degree = deg_i
                
        if best_vertex == -1:
            # All mapped
            if check_edges_validity(tree_idx, mapping):
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
                
        # LCV: Sort candidates by how many options they leave for neighbors
        if len(valid_candidates) > 1:
            # For each candidate host, count how many valid hosts remain for each unmapped neighbor
            def count_remaining_options(h):
                total_options = 0
                for neighbor in adj_t[u]:
                    if mapping[neighbor] == -1:
                        # Count how many hosts are valid for this neighbor given u->h
                        valid_count = 0
                        for hn in range(20):
                            if hn in used_hosts_in_tree:
                                continue
                            # Check if hn is valid for neighbor given u->h
                            valid_for_neighbor = True
                            for n_neighbor in adj_t[neighbor]:
                                if mapping[n_neighbor] != -1:
                                    hnn = mapping[n_neighbor]
                                    e = (min(hn, hnn), max(hn, hnn))
                                    if e in used_edges:
                                        valid_for_neighbor = False
                                        break
                            if valid_for_neighbor:
                                valid_count += 1
                        total_options += valid_count
                return total_options
            
            # Sort by decreasing remaining options (LCV: least constraining value)
            valid_candidates.sort(key=lambda h: -count_remaining_options(h))
            # Add some randomness to break ties
            if len(valid_candidates) > 5:
                # Shuffle the top few to add diversity
                top_k = min(5, len(valid_candidates))
                valid_candidates[:top_k] = [valid_candidates[i] for i in rng.sample(range(top_k), top_k)]
        else:
            rng.shuffle(valid_candidates)
        
        for h in valid_candidates:
            mapping[u] = h
            used_hosts_in_tree.add(h)
            
            # Forward Checking: Ensure all unmapped neighbors of u still have valid hosts
            fc_failed = False
            for neighbor in adj_t[u]:
                if mapping[neighbor] == -1:
                    if not has_valid_host_for_vertex(tree_idx, neighbor, mapping, used_hosts_in_tree):
                        fc_failed = True
                        break
            
            if not fc_failed:
                if embed_tree(tree_idx, mapping, used_hosts_in_tree):
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
        rng = random.Random(12345 + attempts)
        
        success = solve(0)
        attempts += 1
        
        if success:
            break
            
        # Calculate score of current partial packing
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += len(trees[i])
                
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
