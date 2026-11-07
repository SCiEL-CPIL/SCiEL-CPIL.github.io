"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 115
    trees = instance_trees(20, 611)
    
    # Build adjacency and degree for each tree
    tree_adj = []
    tree_max_deg = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        tree_max_deg.append(max(len(adj[i]) for i in range(k)) if k > 1 else 0)
        
    best_result = [None] * 20
    best_edges = 0
    
    # Sort tree indices by complexity (max degree) descending, then size descending
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: (-tree_max_deg[i], -(i+1)))
    
    # Precompute all edges as bitmasks for fast edge-disjointness checks
    edge_index = {}
    edge_list = []
    for a in range(20):
        for b in range(a + 1, 20):
            edge_index[(a, b)] = len(edge_list)
            edge_list.append((a, b))
    num_edges = len(edge_list)  # 190
    
    # For each tree, precompute the bitmask of edges that would be used
    # if we assign a specific host to each vertex. Actually, we need to compute
    # this dynamically during backtracking. But we can precompute for each tree
    # the list of (u, v) pairs.
    
    # Helper to convert a set of used edges to a bitmask
    # We'll use a list of booleans or a set of edge indices
    
    def try_pack(order, used_edge_set, rng):
        result = [None] * 20
        used_edges = set(used_edge_set)
        
        for k_idx in order:
            k = k_idx + 1
            if k == 1:
                result[k_idx] = [0]
                continue
                
            if k == 2:
                found = False
                for a in range(20):
                    for b in range(a + 1, 20):
                        edge = (a, b)
                        if edge not in used_edges:
                            used_edges.add(edge)
                            result[k_idx] = [a, b]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    continue
                continue
            
            adj = tree_adj[k_idx]
            k_val = k
            mapping = [-1] * k_val
            host_used = [False] * 20
            
            deg = [len(adj[i]) for i in range(k_val)]
            vertex_order = sorted(range(k_val), key=lambda x: -deg[x])
            
            def has_available_host(u):
                for h in range(20):
                    if host_used[h]:
                        continue
                    ok = True
                    for v in adj[u]:
                        if mapping[v] != -1:
                            hv = mapping[v]
                            edge = (min(h, hv), max(h, hv))
                            if edge in used_edges:
                                ok = False
                                break
                    if ok:
                        return True
                return False

            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(vertex_order):
                    return True
                
                tv = vertex_order[idx]
                candidates = [h for h in range(20) if not host_used[h]]
                
                # Value ordering: prefer hosts with fewer used edges
                used_deg = [0]*20
                for (a,b) in used_edges:
                    used_deg[a] += 1
                    used_deg[b] += 1
                candidates.sort(key=lambda h: used_deg[h])
                
                for hv in candidates:
                    ok = True
                    for nb in adj[tv]:
                        if mapping[nb] != -1:
                            hnb = mapping[nb]
                            edge = (min(hv, hnb), max(hv, hnb))
                            if edge in used_edges:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    mapping[tv] = hv
                    host_used[hv] = True
                    
                    fc_ok = True
                    for nb in adj[tv]:
                        if mapping[nb] == -1:
                            if not has_available_host(nb):
                                fc_ok = False
                                break
                                
                    if fc_ok:
                        if backtrack(idx + 1):
                            return True
                    
                    mapping[tv] = -1
                    host_used[hv] = False
                
                return False
            
            if backtrack(0):
                result[k_idx] = [mapping[i] for i in range(k_val)]
                for (u, v) in trees[k_idx]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
            
        return result, used_edges

    # Strategy 1: Pack in order of complexity with random restarts
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        rng = Random(attempt * 7919 + 12345)
        
        groups = {}
        for i in tree_indices:
            md = tree_max_deg[i]
            if md not in groups:
                groups[md] = []
            groups[md].append(i)
            
        sorted_keys = sorted(groups.keys(), reverse=True)
        current_order = []
        for key in sorted_keys:
            grp = groups[key]
            rng.shuffle(grp)
            current_order.extend(grp)
            
        res, used = try_pack(current_order, set(), rng)
        
        total_edges = sum(len(trees[k]) for k in range(20) if res[k] is not None)
        
        if total_edges > best_edges:
            best_edges = total_edges
            best_result = res[:]
            if best_edges == 190:
                break
                
    # Strategy 2: Random order of trees
    if best_edges < 190:
        for attempt in range(1000):
            if time.time() > deadline:
                break
            rng = Random(attempt * 104729 + 67890)
            current_order = list(range(20))
            rng.shuffle(current_order)
            
            res, used = try_pack(current_order, set(), rng)
            
            total_edges = sum(len(trees[k]) for k in range(20) if res[k] is not None)
            
            if total_edges > best_edges:
                best_edges = total_edges
                best_result = res[:]
                if best_edges == 190:
                    break

    final_result = []
    for k in range(20):
        if best_result[k] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_result[k]])
            
    return final_result
# EVOLVE-BLOCK-END
