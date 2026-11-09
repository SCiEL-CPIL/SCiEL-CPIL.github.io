import time
from random import Random

def _gen_trees(n=20, seed=611):
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    N = 20
    trees = _gen_trees(N, 611)
    
    # Precompute tree properties
    tree_info = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        tree_info.append({
            'k': k,
            'edges': edges,
            'deg': deg,
            'adj': adj,
            'nedges': len(edges)
        })
        
    all_edges = set()
    for a in range(N):
        for b in range(a + 1, N):
            all_edges.add((a, b))
            
    # Order of embedding: Try largest trees first
    order = sorted(range(N), key=lambda i: -tree_info[i]['k'])
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_result = [None] * N
    best_value = 0
    
    def value_of(result):
        return sum(tree_info[i]['nedges'] for i in range(N) if result[i] is not None)
        
    def try_order(order):
        nonlocal best_result, best_value
        used_edges = set()
        host_deg_used = [0] * N
        result = [None] * N
        node_count = 0
        max_nodes = 2000000
        
        def dfs_tree(idx_in_order):
            nonlocal node_count
            if node_count > max_nodes:
                return False
            if time.time() > deadline:
                return False
            if idx_in_order == len(order):
                return True
            node_count += 1
            tree_idx = order[idx_in_order]
            info = tree_info[tree_idx]
            k = info['k']
            degs = info['deg']
            adj = info['adj']
            
            # Find embedding for this tree
            mapping = [-1] * k
            host_to_tree = [-1] * N
            
            # Sort tree vertices by degree descending
            perm = sorted(range(k), key=lambda i: -degs[i])
            
            def dfs_vertex(pos):
                nonlocal node_count
                if node_count > max_nodes:
                    return False
                if time.time() > deadline:
                    return False
                if pos == k:
                    return True
                node_count += 1
                v = perm[pos]
                
                # Candidates for host
                candidates = []
                for h in range(N):
                    if host_to_tree[h] != -1:
                        continue
                    if host_deg_used[h] + degs[v] > 19:
                        continue
                    # Check edges to placed neighbors
                    valid = True
                    for u in adj[v]:
                        if mapping[u] != -1:
                            hu = mapping[u]
                            e = (min(h, hu), max(h, hu))
                            if e in used_edges:
                                valid = False
                                break
                    if valid:
                        candidates.append(h)
                        
                if not candidates:
                    return False
                    
                # Order candidates: prefer hosts with higher used degree (tighter fit?)
                # Or just arbitrary. Let's try sorting by host_deg_used desc.
                candidates.sort(key=lambda h: -host_deg_used[h])
                
                for h in candidates:
                    mapping[v] = h
                    host_to_tree[h] = v
                    host_deg_used[h] += degs[v]
                    
                    # Forward check: ensure next vertices have candidates
                    ok = True
                    next_pos = pos + 1
                    if next_pos < k:
                        nv = perm[next_pos]
                        has_cand = False
                        for nh in range(N):
                            if host_to_tree[nh] != -1:
                                continue
                            if host_deg_used[nh] + degs[nv] > 19:
                                continue
                            valid_nc = True
                            for nu in adj[nv]:
                                if mapping[nu] != -1:
                                    hnu = mapping[nu]
                                    e = (min(nh, hnu), max(nh, hnu))
                                    if e in used_edges:
                                        valid_nc = False
                                        break
                            if valid_nc:
                                has_cand = True
                                break
                        if not has_cand:
                            ok = False
                            
                    if ok and dfs_vertex(pos + 1):
                        return True
                        
                    host_deg_used[h] -= degs[v]
                    host_to_tree[h] = -1
                    mapping[v] = -1
                return False
                
            if dfs_vertex(0):
                # Update global state
                for v in range(k):
                    h = mapping[v]
                for u, v in info['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
                result[tree_idx] = [int(mapping[i]) for i in range(k)]
                
                if dfs_tree(idx_in_order + 1):
                    return True
                else:
                    # Backtrack: undo updates
                    for u, v in info['edges']:
                        hu = mapping[u]
                        hv = mapping[v]
                        used_edges.remove((min(hu, hv), max(hu, hv)))
                    result[tree_idx] = None
                    return False
            return False
            
        if dfs_tree(0):
            val = value_of(result)
            if val > best_value:
                best_value = val
                best_result = [x[:] if x is not None else None for x in result]
            return True
        return False
        
    # Try the main order
    try_order(order)
    
    # If failed or partial, try randomized restarts
    if best_value < 190:
        rng = Random(123)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            # Perturb order
            o = order[:]
            i = rng.randrange(len(o))
            j = rng.randrange(len(o))
            o[i], o[j] = o[j], o[i]
            try_order(o)
            if best_value >= 190:
                break
                
    if best_result[0] is None:
        best_result[0] = [0]
        
    return best_result
