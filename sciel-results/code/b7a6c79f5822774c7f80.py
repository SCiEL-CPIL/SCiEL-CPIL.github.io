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
    deadline = start_time + 115  # Safety margin before 120s limit
    
    trees = instance_trees(20, 611)
    
    # Pre-compute tree adjacency
    tree_adj = []
    tree_edges = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        tree_edges.append(edges)
    
    # Total edges to pack
    total_edges = sum(len(t) for t in trees)  # Should be 190
    
    # K20 edges
    all_k20_edges = set()
    for i in range(20):
        for j in range(i + 1, 20):
            all_k20_edges.add((i, j))
    
    # Strategy: Backtracking with edge-level constraint propagation
    # Process trees in order of decreasing complexity (more edges, higher max degree)
    
    # Compute tree complexity metrics
    def tree_complexity(idx):
        k = idx + 1
        adj = tree_adj[idx]
        max_deg = max(len(adj[v]) for v in range(k)) if k > 1 else 0
        num_edges = len(tree_edges[idx])
        return (num_edges, max_deg, k)
    
    order = list(range(20))
    order.sort(key=lambda x: tree_complexity(x), reverse=True)
    
    # State
    used_edges = set()
    mappings = [None] * 20
    best_mappings = [None] * 20
    best_edge_count = 0
    
    def time_check():
        return time.time() > deadline
    
    def get_available_hosts_for_vertex(tree_idx, v, mapping, used_hosts):
        """Get all host vertices that v can be mapped to, given current partial mapping."""
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        available = []
        for h in range(20):
            if h in used_hosts:
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
                available.append(h)
        return available
    
    def embed_tree(tree_idx, mapping, used_hosts):
        """Embed tree tree_idx using backtracking with MRV."""
        if time_check():
            return False
        
        k = tree_idx + 1
        if k <= 1:
            if k == 1:
                # Single vertex, any unused host works
                for h in range(20):
                    if h not in used_hosts:
                        mapping[0] = h
                        mappings[tree_idx] = mapping[:]
                        return True
                return False
            else:
                # T2: single edge (0,1)
                for h0 in range(20):
                    if h0 in used_hosts:
                        continue
                    for h1 in range(20):
                        if h1 in used_hosts or h1 == h0:
                            continue
                        e = (min(h0, h1), max(h0, h1))
                        if e not in used_edges:
                            mapping[0] = h0
                            mapping[1] = h1
                            mappings[tree_idx] = mapping[:]
                            return True
                return False
        
        adj_t = tree_adj[tree_idx]
        
        # Find unmapped vertex with MRV
        best_v = -1
        min_opts = 21
        best_opts = []
        
        for i in range(k):
            if mapping[i] != -1:
                continue
            opts = get_available_hosts_for_vertex(tree_idx, i, mapping, used_hosts)
            if len(opts) < min_opts:
                min_opts = len(opts)
                best_v = i
                best_opts = opts
                if min_opts == 0:
                    return False
                if min_opts == 1:
                    break
        
        if best_v == -1:
            # All mapped - verify and commit
            for u in range(k):
                hu = mapping[u]
                for v in adj_t[u]:
                    if u < v:
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
                        used_edges.add(e)
            mappings[tree_idx] = mapping[:]
            return True
        
        # Try each option for best_v
        for h in best_opts:
            mapping[best_v] = h
            used_hosts.add(h)
            
            # Forward checking
            fc_ok = True
            for neighbor in adj_t[best_v]:
                if mapping[neighbor] == -1:
                    if not get_available_hosts_for_vertex(tree_idx, neighbor, mapping, used_hosts):
                        fc_ok = False
                        break
            
            if fc_ok and embed_tree(tree_idx, mapping, used_hosts):
                return True
            
            used_hosts.remove(h)
            mapping[best_v] = -1
        
        return False
    
    def solve(order, idx):
        nonlocal best_edge_count, best_mappings
        
        if idx == 20:
            # All trees embedded
            edge_count = len(used_edges)
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_mappings = [m[:] if m else None for m in mappings]
            return True
        
        if time_check():
            return False
        
        t_idx = order[idx]
        k = t_idx + 1
        mapping = [-1] * k
        used_hosts = set()
        
        if embed_tree(t_idx, mapping, used_hosts):
            if solve(order, idx + 1):
                return True
        
        # Undo
        if mappings[t_idx] is not None:
            # Remove edges from used_edges
            m = mappings[t_idx]
            adj_t = tree_adj[t_idx]
            for u in range(k):
                for v in adj_t[u]:
                    if u < v:
                        e = (min(m[u], m[v]), max(m[u], m[v]))
                        used_edges.discard(e)
            mappings[t_idx] = None
        
        return False
    
    # Try the sorted order
    success = solve(order, 0)
    
    if not success and not time_check():
        # Try a few random restarts with different orderings
        rng = random.Random(42)
        for attempt in range(10):
            if time_check():
                break
            # Reset state
            used_edges.clear()
            mappings = [None] * 20
            
            # Randomize order slightly
            rand_order = order[:]
            rng.shuffle(rand_order)
            # Keep large trees first but shuffle within similar sizes
            solve(rand_order, 0)
    
    # Build result
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    
    return result
