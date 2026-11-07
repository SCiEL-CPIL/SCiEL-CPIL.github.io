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
    deadline = time.time() + 115  # safety margin before 120s limit
    trees = instance_trees(20, 611)
    
    # Build adjacency for each tree
    tree_adj = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    best_result = [None] * 20
    best_edges = 0
    
    # Strategy: Fix T20 to identity, then pack remaining trees
    # T20 identity: vertex i -> host i, uses edges (i,j) for all tree edges in T20
    t20_identity = list(range(20))
    used_edges_t20 = set()
    for (u, v) in trees[19]:  # trees[19] is T20
        used_edges_t20.add((min(u, v), max(u, v)))
    
    # Try multiple random seeds for packing the rest
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        rng = Random(attempt * 7919 + 12345)
        
        # Start with T20 fixed
        used_edges = set(used_edges_t20)
        result = [None] * 20
        result[19] = t20_identity[:]
        
        success = True
        
        # Pack trees from T19 down to T1
        for k in range(19, 0, -1):
            if time.time() > deadline:
                success = False
                break
            
            if k == 1:
                # T1 is a single vertex, any host works
                result[0] = [rng.randint(0, 19)]
                continue
            
            if k == 2:
                # T2 is a single edge
                found = False
                for _ in range(100):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a == b:
                        continue
                    edge = (min(a, b), max(a, b))
                    if edge not in used_edges:
                        used_edges.add(edge)
                        result[1] = [a, b]
                        found = True
                        break
                if not found:
                    # Systematic fallback
                    for a in range(20):
                        for b in range(a + 1, 20):
                            edge = (a, b)
                            if edge not in used_edges:
                                used_edges.add(edge)
                                result[1] = [a, b]
                                found = True
                                break
                        if found:
                            break
                if not found:
                    success = False
                    break
                continue
            
            # For k >= 3: backtracking embedding
            adj = tree_adj[k - 1]
            k_val = k
            mapping = [-1] * k_val
            host_used = [False] * 20
            
            # Order vertices by degree (highest first)
            deg = [len(adj[i]) for i in range(k_val)]
            vertex_order = sorted(range(k_val), key=lambda x: -deg[x])
            
            # Precompute for each tree vertex, which neighbors are
            neighbor_sets = [set(adj[i]) for i in range(k_val)]
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(vertex_order):
                    return True
                
                tv = vertex_order[idx]
                
                # Get assigned neighbors
                assigned_neighbors = []
                for nb in adj[tv]:
                    if mapping[nb] != -1:
                        assigned_neighbors.append(nb)
                
                # Try host vertices, shuffled for randomness
                candidates = list(range(20))
                rng.shuffle(candidates)
                
                for hv in candidates:
                    if host_used[hv]:
                        continue
                    
                    # Check edges to assigned neighbors
                    ok = True
                    for nb in assigned_neighbors:
                        hnb = mapping[nb]
                        edge = (min(hv, hnb), max(hv, hnb))
                        if edge in used_edges:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Additional pruning: check if this assignment leaves
                    # enough available edges for unassigned neighbors
                    # (lightweight check)
                    unassigned_neighbors = []
                    for nb in adj[tv]:
                        if mapping[nb] == -1:
                            unassigned_neighbors.append(nb)
                    
                    # Tentatively assign
                    mapping[tv] = hv
                    host_used[hv] = True
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    mapping[tv] = -1
                    host_used[hv] = False
                
                return False
            
            if backtrack(0):
                result[k - 1] = [mapping[i] for i in range(k_val)]
                for (u, v) in trees[k - 1]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
            else:
                success = False
                break
        
        if success:
            total_edges = sum(len(trees[k - 1]) for k in range(1, 21) if result[k - 1] is not None)
            if total_edges > best_edges:
                best_edges = total_edges
                best_result = result[:]
                if best_edges == 190:
                    break
    
    # If we didn't get a perfect packing, try without fixing T20
    if best_edges < 190:
        order = list(range(1, 21))
        order.reverse()
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            rng = Random(attempt * 104729 + 67890)
            used_edges = set()
            result = [None] * 20
            success = True
            
            for k in order:
                if time.time() > deadline:
                    success = False
                    break
                
                if k == 1:
                    result[0] = [rng.randint(0, 19)]
                    continue
                
                if k == 2:
                    found = False
                    for _ in range(100):
                        a = rng.randint(0, 19)
                        b = rng.randint(0, 19)
                        if a == b:
                            continue
                        edge = (min(a, b), max(a, b))
                        if edge not in used_edges:
                            used_edges.add(edge)
                            result[1] = [a, b]
                            found = True
                            break
                    if not found:
                        for a in range(20):
                            for b in range(a + 1, 20):
                                edge = (a, b)
                                if edge not in used_edges:
                                    used_edges.add(edge)
                                    result[1] = [a, b]
                                    found = True
                                    break
                            if found:
                                break
                    if not found:
                        success = False
                        break
                    continue
                
                adj = tree_adj[k - 1]
                k_val = k
                mapping = [-1] * k_val
                host_used = [False] * 20
                
                deg = [len(adj[i]) for i in range(k_val)]
                vertex_order = sorted(range(k_val), key=lambda x: -deg[x])
                
                def backtrack(idx):
                    if time.time() > deadline:
                        return False
                    if idx == len(vertex_order):
                        return True
                    
                    tv = vertex_order[idx]
                    assigned_neighbors = [nb for nb in adj[tv] if mapping[nb] != -1]
                    
                    candidates = list(range(20))
                    rng.shuffle(candidates)
                    
                    for hv in candidates:
                        if host_used[hv]:
                            continue
                        
                        ok = True
                        for nb in assigned_neighbors:
                            hnb = mapping[nb]
                            edge = (min(hv, hnb), max(hv, hnb))
                            if edge in used_edges:
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        mapping[tv] = hv
                        host_used[hv] = True
                        
                        if backtrack(idx + 1):
                            return True
                        
                        mapping[tv] = -1
                        host_used[hv] = False
                    
                    return False
                
                if backtrack(0):
                    result[k - 1] = [mapping[i] for i in range(k_val)]
                    for (u, v) in trees[k - 1]:
                        hu, hv = mapping[u], mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                else:
                    success = False
                    break
            
            if success:
                total_edges = sum(len(trees[k - 1]) for k in range(1, 21) if result[k - 1] is not None)
                if total_edges > best_edges:
                    best_edges = total_edges
                    best_result = result[:]
                    if best_edges == 190:
                        break
    
    # Ensure all values are Python ints in lists
    final_result = []
    for k in range(20):
        if best_result[k] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_result[k]])
    
    return final_result
# EVOLVE-BLOCK-END
