import time
import random
from itertools import combinations

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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    trees = instance_trees(20, 611)
    
    # Precompute all edges in K20
    all_edges = set()
    for i in range(20):
        for j in range(i + 1, 20):
            all_edges.add((i, j))
    
    # Precompute adjacency for each tree
    tree_info = []
    for k in range(1, 21):
        if k == 1:
            tree_info.append({'edges': [], 'adj': {}, 'order': [0], 'parent': [-1], 'deg': [0]})
            continue
        edges = trees[k-1]
        adj = {i: [] for i in range(k)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from root 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    order.append(v)
        
        # Compute degree in tree
        deg = [len(adj[i]) for i in range(k)]
        
        tree_info.append({'edges': edges, 'adj': adj, 'order': order, 'parent': parent, 'deg': deg})
    
    def try_embed(order_idx_list, used_edges, result):
        """Try to embed trees in the given order. Returns True if all embedded."""
        for idx in order_idx_list:
            k = idx + 1
            if time.time() > deadline:
                return False
            
            if k == 1:
                result[k-1] = [0]
                continue
                
            info = tree_info[k-1]
            edges = info['edges']
            order = info['order']
            parent = info['parent']
            deg = info['deg']
            
            # Backtracking with forward checking
            mapping = {}
            assigned_hosts = set()
            # Track remaining degree capacity for each host
            host_rem_deg = [19] * 20  # Each host can have at most 19 edges total
            
            # Subtract already used edges from host_rem_deg
            for (a, b) in used_edges:
                host_rem_deg[a] -= 1
                host_rem_deg[b] -= 1
            
            def dfs(idx_pos):
                if idx_pos == k:
                    return True
                
                v = order[idx_pos]
                
                if idx_pos == 0:
                    # Root: any host with sufficient remaining degree
                    candidates = []
                    for u in range(20):
                        if u not in assigned_hosts and host_rem_deg[u] >= deg[v]:
                            candidates.append(u)
                else:
                    p = parent[v]
                    h_p = mapping[p]
                    candidates = []
                    for u in range(20):
                        if u in assigned_hosts:
                            continue
                        if u == h_p:
                            continue
                        e = (min(h_p, u), max(h_p, u))
                        if e in used_edges:
                            continue
                        # Forward checking: host u must have enough remaining degree
                        # for vertex v's degree (excluding the edge to parent)
                        if host_rem_deg[u] < deg[v]:
                            continue
                        candidates.append(u)
                
                if not candidates:
                    return False
                
                # Heuristic: sort by remaining degree (prefer hosts with more capacity)
                candidates.sort(key=lambda u: -host_rem_deg[u])
                
                for u in candidates:
                    mapping[v] = u
                    assigned_hosts.add(u)
                    
                    # Update host_rem_deg
                    if idx_pos > 0:
                        p = parent[v]
                        h_p = mapping[p]
                        host_rem_deg[u] -= 1
                        host_rem_deg[h_p] -= 1
                    
                    if dfs(idx_pos + 1):
                        return True
                    
                    # Backtrack
                    if idx_pos > 0:
                        p = parent[v]
                        h_p = mapping[p]
                        host_rem_deg[u] += 1
                        host_rem_deg[h_p] += 1
                    
                    del mapping[v]
                    assigned_hosts.remove(u)
                
                return False
            
            if dfs(0):
                # Update used_edges
                for u, v in edges:
                    h_u = mapping[u]
                    h_v = mapping[v]
                    e = (min(h_u, h_v), max(h_u, h_v))
                    used_edges.add(e)
                
                res_list = [mapping[i] for i in range(k)]
                result[k-1] = res_list
                return True
            else:
                return False
    
    # Strategy: Try different orderings with randomized restarts
    best_result = [None] * 20
    best_edge_count = -1
    
    # Base order: largest first
    base_order = list(range(19, 0, -1))  # indices 19,18,...,0 (T20 down to T1)
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    # First try the base order
    result = [None] * 20
    used_edges = set()
    if try_embed(base_order, used_edges, result):
        edge_count = sum(len(trees[i]) for i in range(20) if result[i] is not None)
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_result = result[:]
    
    # Try randomized orderings
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a random ordering biased toward larger trees
        order = base_order[:]
        rng.shuffle(order)
        
        result = [None] * 20
        used_edges = set()
        if try_embed(order, used_edges, result):
            edge_count = sum(len(trees[i]) for i in range(20) if result[i] is not None)
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_result = result[:]
    
    # If we didn't get all 190 edges, try a more exhaustive approach
    if best_edge_count < 190:
        # Try embedding trees one by one with multiple attempts per tree
        result = [None] * 20
        used_edges = set()
        
        for k in range(20, 0, -1):
            if time.time() > deadline - 5:
                break
            
            if k == 1:
                result[k-1] = [0]
                continue
            
            info = tree_info[k-1]
            edges = info['edges']
            order = info['order']
            parent = info['parent']
            deg = info['deg']
            
            # Try multiple random root assignments
            found = False
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                
                mapping = {}
                assigned_hosts = set()
                host_rem_deg = [19] * 20
                
                for (a, b) in used_edges:
                    host_rem_deg[a] -= 1
                    host_rem_deg[b] -= 1
                
                def dfs2(idx_pos):
                    if idx_pos == k:
                        return True
                    
                    v = order[idx_pos]
                    
                    if idx_pos == 0:
                        candidates = []
                        for u in range(20):
                            if u not in assigned_hosts and host_rem_deg[u] >= deg[v]:
                                candidates.append(u)
                        rng.shuffle(candidates)
                    else:
                        p = parent[v]
                        h_p = mapping[p]
                        candidates = []
                        for u in range(20):
                            if u in assigned_hosts:
                                continue
                            if u == h_p:
                                continue
                            e = (min(h_p, u), max(h_p, u))
                            if e in used_edges:
                                continue
                            if host_rem_deg[u] < deg[v]:
                                continue
                            candidates.append(u)
                        rng.shuffle(candidates)
                    
                    if not candidates:
                        return False
                    
                    for u in candidates:
                        mapping[v] = u
                        assigned_hosts.add(u)
                        
                        if idx_pos > 0:
                            p = parent[v]
                            h_p = mapping[p]
                            host_rem_deg[u] -= 1
                            host_rem_deg[h_p] -= 1
                        
                        if dfs2(idx_pos + 1):
                            return True
                        
                        if idx_pos > 0:
                            p = parent[v]
                            h_p = mapping[p]
                            host_rem_deg[u] += 1
                            host_rem_deg[h_p] += 1
                        
                        del mapping[v]
                        assigned_hosts.remove(u)
                    
                    return False
                
                if dfs2(0):
                    for u, v in edges:
                        h_u = mapping[u]
                        h_v = mapping[v]
                        e = (min(h_u, h_v), max(h_u, h_v))
                        used_edges.add(e)
                    
                    result[k-1] = [mapping[i] for i in range(k)]
                    found = True
                    break
            
            if not found:
                break
        
        edge_count = sum(len(trees[i]) for i in range(20) if result[i] is not None)
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_result = result[:]
    
    return best_result
