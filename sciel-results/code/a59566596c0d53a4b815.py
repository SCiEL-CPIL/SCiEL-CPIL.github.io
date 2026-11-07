import time
from random import Random

def _instance_trees(n=20, seed=611):
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
    deadline = start_time + 110
    
    trees = _instance_trees(20, 611)
    
    # Pre-compute tree properties
    # We will embed in order: 20, 19, ..., 1
    # So we reverse the list for processing, but map back to original indices
    
    # Order: [19, 18, ..., 0] (indices for T20 to T1)
    order = list(range(19, -1, -1))
    
    best_result = [None] * 20
    best_value = 0
    
    def try_packing(rng_seed):
        rng = Random(rng_seed)
        used_edges = set()
        assignments = [None] * 20
        total_edges = 0
        
        for idx in order:
            k = idx + 1
            edges = trees[idx]
            
            if not edges:
                # T1
                assignments[idx] = [0]
                continue
            
            # Try to embed T_k
            success = False
            for attempt in range(50):
                if time.time() > deadline:
                    return None, 0
                
                # Greedy embedding with randomization
                # Build adjacency
                adj = [[] for _ in range(k)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                    
                # BFS order
                root = rng.randint(0, k - 1)
                visited = {root}
                queue = [root]
                bfs_order = []
                parent = {root: -1}
                while queue:
                    cur = queue.pop(0)
                    bfs_order.append(cur)
                    for nb in adj[cur]:
                        if nb not in visited:
                            visited.add(nb)
                            parent[nb] = cur
                            queue.append(nb)
                            
                host_map = {}
                used_hosts = set()
                
                # Assign root
                root_host = rng.randint(0, 19)
                host_map[root] = root_host
                used_hosts.add(root_host)
                
                valid = True
                for tv in bfs_order:
                    if tv == root:
                        continue
                    p_tv = parent[tv]
                    p_host = host_map[p_tv]
                    
                    # Find available host for tv
                    candidates = [h for h in range(20) if h not in used_hosts and 
                                 (p_host, h) not in used_edges and (h, p_host) not in used_edges]
                    if not candidates:
                        valid = False
                        break
                    # Choose one
                    h = rng.choice(candidates)
                    host_map[tv] = h
                    used_hosts.add(h)
                
                if not valid:
                    continue
                
                # Verify all edges
                ok = True
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    if (hu, hv) in used_edges or (hv, hu) in used_edges:
                        ok = False
                        break
                
                if ok:
                    assignments[idx] = [host_map[i] for i in range(k)]
                    for u, v in edges:
                        hu, hv = host_map[u], host_map[v]
                        used_edges.add((hu, hv))
                        used_edges.add((hv, hu))
                    total_edges += len(edges)
                    success = True
                    break
            
            if not success:
                return None, 0
        
        return assignments, total_edges

    # Try multiple seeds
    for seed in range(1000):
        if time.time() > deadline:
            break
        res, val = try_packing(seed)
        if res is not None:
            if val > best_value:
                best_value = val
                best_result = res
                if val == 190:
                    break

    # Format output
    final = []
    for i in range(20):
        if best_result[i] is None:
            final.append(None)
        else:
            final.append([int(x) for x in best_result[i]])
            
    return final
