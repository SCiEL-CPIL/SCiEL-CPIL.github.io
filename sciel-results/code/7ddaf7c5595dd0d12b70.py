import time
import random
from collections import deque

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
    deadline = start_time + 118  # 2s margin before 120s kill
    
    trees = instance_trees(20, 611)
    
    # Precompute tree properties
    tree_info = []
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
        tree_info.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'num_edges': len(edges)
        })
    
    # Precompute DFS order for each tree (parent always before child)
    # This ensures when we place a vertex, its parent is already placed
    def get_dfs_order(adj, k, root=0):
        visited = [False] * k
        visited[root] = True
        order = []
        parent = [-1] * k
        stack = [root]
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    stack.append(v)
        return order, parent
    
    dfs_data = []
    for i in range(20):
        k = tree_info[i]['k']
        if k == 1:
            dfs_data.append(([0], [-1]))
        else:
            order, parent = get_dfs_order(tree_info[i]['adj'], k)
            dfs_data.append((order, parent))
    
    # Sort trees by size descending - pack larger trees first
    base_order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    
    def try_pack(attempt_seed, order, max_backtrack_depth=50000):
        """Try to pack all trees using backtracking with given order and seed"""
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        nodes_backtracked = 0
        
        def find_mapping(t_idx):
            """Find embedding of tree t_idx using backtracking on DFS order"""
            nonlocal nodes_backtracked
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            deg_t = info['deg']
            dfs_order, parent = dfs_data[t_idx]
            
            if k == 1:
                return [0]
            
            # mapping[tree_vertex] = host_vertex
            mapping = [-1] * k
            used_hosts = set()
            
            # We process in DFS order; for each vertex, its parent is already mapped
            # (except root)
            
            def backtrack(pos):
                nonlocal nodes_backtracked
                if time.time() > deadline:
                    return False
                if nodes_backtracked > max_backtrack_depth:
                    return False
                
                if pos == k:
                    # All vertices mapped - commit edges
                    for u, v in info['edges']:
                        hu = mapping[u]
                        hv = mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    return True
                
                tree_v = dfs_order[pos]
                nodes_backtracked += 1
                
                # Get candidate hosts
                if pos == 0:
                    # Root: any unused host
                    candidates = [hv for hv in range(20) if hv not in used_hosts]
                else:
                    # Must connect to parent via unused edge
                    p_tree = parent[tree_v]
                    p_host = mapping[p_tree]
                    candidates = [hv for hv in range(20) 
                                   if hv not in used_hosts 
                                   and (min(p_host, hv), max(p_host, hv)) not in used_edges]
                
                if not candidates:
                    return False
                
                # Heuristic: prefer hosts with more remaining unused edges
                # (more flexible for future placements)
                def host_score(hv):
                    count = 0
                    for o in range(20):
                        if o != hv and (min(hv, o), max(hv, o)) not in used_edges:
                            count += 1
                    return count
                
                candidates.sort(key=host_score, reverse=True)
                
                # Try candidates (shuffle top few for variety)
                top_n = min(len(candidates), 8)
                trial_list = candidates[:top_n]
                local_rng.shuffle(trial_list)
                
                for hv in trial_list:
                    mapping[tree_v] = hv
                    used_hosts.add(hv)
                    # Add edge to parent
                    if pos > 0:
                        p_tree = parent[tree_v]
                        p_host = mapping[p_tree]
                        used_edges.add((min(p_host, hv), max(p_host, hv)))
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Undo
                    used_hosts.discard(hv)
                    if pos > 0:
                        p_tree = parent[tree_v]
                        p_host = mapping[p_tree]
                        used_edges.discard((min(p_host, hv), max(p_host, hv)))
                    mapping[tree_v] = -1
                
                return False
            
            if backtrack(0):
                return mapping[:]
            else:
                # Failed - clean up any partially added edges
                # (they were added during backtracking but not all committed)
                # Actually, edges are only committed at the end (pos==k),
                # but during backtracking we add/remove individual edges.
                # If we return False, no edges should have been permanently added
                # because we only add at pos==k success.
                # Wait - I'm adding edges during the forward pass and removing on backtrack.
                # If we fail completely, all edges should be removed. Let me verify.
                # Actually this is a problem: I'm adding edges as we go, not just at the end.
                # Let me restructure: only track edges in a local set, commit at end.
                return None
        
        # Redo with proper edge tracking
        def find_mapping_v2(t_idx):
            nonlocal nodes_backtracked
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            deg_t = info['deg']
            dfs_order, parent = dfs_data[t_idx]
            
            if k == 1:
                return [0]
            
            mapping = [-1] * k
            used_hosts = set()
            local_used_edges = set()
            
            def backtrack(pos):
                nonlocal nodes_backtracked
                if time.time() > deadline:
                    return False
                if nodes_backtracked > max_backtrack_depth:
                    return False
                
                if pos == k:
                    # Commit all edges to global used_edges
                    for u, v in info['edges']:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            # Shouldn't happen if we checked during backtracking
                            return False
                        used_edges.add(e)
                    return True
                
                tree_v = dfs_order[pos]
                nodes_backtracked += 1
                
                if pos == 0:
                    candidates = [hv for hv in range(20) if hv not in used_hosts]
                else:
                    p_tree = parent[tree_v]
                    p_host = mapping[p_tree]
                    candidates = [hv for hv in range(20)
                                   if hv not in used_hosts
                                   and (min(p_host, hv), max(p_host, hv)) not in used_edges
                                   and (min(p_host, hv), max(p_host, hv)) not in local_used_edges]
                
                if not candidates:
                    return False
                
                def host_score(hv):
                    count = 0
                    for o in range(20):
                        if o != hv and (min(hv, o), max(hv, o)) not in used_edges:
                            count += 1
                    return count
                
                candidates.sort(key=host_score, reverse=True)
                
                top_n = min(len(candidates), 10)
                trial_list = candidates[:top_n]
                local_rng.shuffle(trial_list)
                
                for hv in trial_list:
                    mapping[tree_v] = hv
                    used_hosts.add(hv)
                    if pos > 0:
                        p_tree = parent[tree_v]
                        p_host = mapping[p_tree]
                        local_used_edges.add((min(p_host, hv), max(p_host, hv)))
                    
                    if backtrack(pos + 1):
                        return True
                    
                    used_hosts.discard(hv)
                    if pos > 0:
                        p_tree = parent[tree_v]
                        p_host = mapping[p_tree]
                        local_used_edges.discard((min(p_host, hv), max(p_host, hv)))
                    mapping[tree_v] = -1
                
                return False
            
            if backtrack(0):
                return mapping[:]
            else:
                return None
        
        for t_idx in order:
            if time.time() > deadline:
                break
            info = tree_info[t_idx]
            k = info['k']
            
            if k == 1:
                mappings[t_idx] = [0]
                continue
            
            result = find_mapping_v2(t_idx)
            if result is not None:
                mappings[t_idx] = result
            # If None, skip this tree
        
        score = sum(tree_info[i]['num_edges'] for i in range(20) if mappings[i] is not None)
        return mappings, score
    
    best_mappings = [None] * 20
    best_score = 0
    
    # Try multiple orders and seeds
    orders = [base_order]
    # Add some shuffled variants
    for s in range(10):
        o = base_order[:]
        random.Random(s * 1000).shuffle(o)
        orders.append(o)
    
    attempt = 0
    while time.time() < deadline:
        order = orders[attempt % len(orders)]
        mappings, score = try_pack(attempt * 7919 + 1, order)
        if mappings is not None and score > best_score:
            best_score = score
            best_mappings = mappings[:]
            if best_score == 190:
                break
        attempt += 1
    
    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    return result

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    # Verify
    trees = instance_trees(20, 611)
    used_edges = set()
    total_edges = 0
    valid = True
    for i in range(20):
        if result[i] is None:
            continue
        mapping = result[i]
        k = i + 1
        if len(mapping) != k:
            print(f"Tree {i}: wrong length {len(mapping)} vs {k}")
            valid = False
            break
        if len(set(mapping)) != k:
            print(f"Tree {i}: duplicate host vertices")
            valid = False
            break
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v > 19:
                print(f"Tree {i}: invalid vertex {v}")
                valid = False
                break
        if not valid:
            break
        for u, v in trees[i]:
            hu = mapping[u]
            hv = mapping[v]
            e = (min(hu, hv), max(hu, hv))
            if e in used_edges:
                print(f"Tree {i}: edge conflict {e}")
                valid = False
                break
            used_edges.add(e)
            total_edges += 1
        if not valid:
            break
    
    print(f"Valid: {valid}")
    print(f"Total edges: {total_edges}")
    if valid:
        print(f"Score: {total_edges}")
    else:
        print("INVALID RESULT")
