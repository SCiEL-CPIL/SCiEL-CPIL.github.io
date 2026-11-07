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
    deadline = time.time() + 140
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
    # This helps place rigid trees first
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: (-tree_max_deg[i], -(i+1)))
    
    # Precompute all edges
    all_edges = set()
    for a in range(20):
        for b in range(a + 1, 20):
            all_edges.add((a, b))
            
    # Helper to check if a tree embedding is valid given used_edges
    # We will use backtracking inside the loop
    
    def try_pack(order, used_edges_init, rng):
        used_edges = set(used_edges_init)
        result = [None] * 20
        # Pre-fill if any were fixed? No, we pass the order of trees to pack.
        # If a tree is not in order, it remains None? 
        # The task requires returning 20 entries. If we don't pack a tree, it's None.
        # But we want to maximize edges. So we should pack as many as possible.
        # The order defines which trees we *attempt* to pack.
        # If we pack them all, we have a full solution.
        
        # We need to map tree index k (0-based) to its position in result.
        # result[k] is the embedding for tree T_{k+1}.
        
        # We pack trees in the given 'order'.
        # If a tree fails, we return the partial result?
        # The current code breaks on failure. 
        # To maximize edges, we should try to pack as many as possible.
        # However, if we fail to pack T_k, we might still pack T_{k-1}.
        # So we should iterate through the order, and if one fails, skip it?
        # But skipping a large tree might make it easier to pack smaller ones?
        # Actually, if we fail to pack T_k, it's because of edge conflicts.
        # Skipping it frees up edges. So yes, skip on failure.
        
        for k_idx in order:
            k = k_idx + 1 # 1-based size
            if k == 1:
                # T1 is a single vertex. Any free vertex works.
                # Actually, T1 has 0 edges. It doesn't consume edges.
                # But it must be a valid vertex.
                # Any vertex 0..19 is fine.
                result[k_idx] = [0] # Just pick 0, doesn't matter for edge count
                continue
                
            if k == 2:
                # T2 is a single edge.
                found = False
                # Try to find any available edge
                # Optimization: iterate over all edges, pick one
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
                    continue # Skip this tree
                continue
            
            # Backtracking for k >= 3
            adj = tree_adj[k_idx]
            k_val = k
            mapping = [-1] * k_val
            host_used = [False] * 20
            
            # Order vertices by degree descending
            deg = [len(adj[i]) for i in range(k_val)]
            vertex_order = sorted(range(k_val), key=lambda x: -deg[x])
            
            # Precompute neighbors for quick access
            # We need to check forward checking:
            # For an unassigned vertex u, is there any host h such that:
            # 1. h is not host_used
            # 2. For all assigned neighbors v of u, (h, mapping[v]) is not in used_edges
            
            def has_available_host(u):
                # Check if there is any host for u
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
                
                # Candidates: all free hosts
                # Heuristic: Sort by "freedom"
                # Freedom = number of free neighbors in K20
                # We calculate this on the fly? 
                # Or just shuffle?
                # Let's try sorting by number of available edges to *assigned* neighbors? No, those are constrained.
                # Let's try sorting by degree in the *remaining* graph?
                
                candidates = [h for h in range(20) if not host_used[h]]
                
                # Value Ordering Heuristic:
                # Prefer hosts that have fewer available edges to other *free* hosts?
                # This is "Least Constraining Value" for the *future*?
                # Actually, if we pick a central node (high degree), we consume many edges.
                # If we pick a peripheral node (low degree), we consume fewer edges.
                # So prefer LOW degree in free graph?
                
                # Calculate free degree for each candidate
                # free_deg[h] = number of h' in candidates such that (h, h') not in used_edges
                # This is O(N^2) per step. N=20. 400 ops. Acceptable.
                
                # To speed up, we can just use a static heuristic or random if it's too slow.
                # Let's try random first, but with a bias?
                # Or just sort by static degree in K20? All are 19.
                
                # Let's try: Sort by number of *used* edges incident to h.
                # Prefer hosts with FEWER used edges (more free space).
                used_deg = [0]*20
                for (a,b) in used_edges:
                    used_deg[a] += 1
                    used_deg[b] += 1
                    
                candidates.sort(key=lambda h: used_deg[h])
                
                for hv in candidates:
                    # Check edges to assigned neighbors
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
                    
                    # Forward Checking
                    # Assign tentatively
                    mapping[tv] = hv
                    host_used[hv] = True
                    
                    # Check if any unassigned neighbor of tv is now blocked
                    fc_ok = True
                    for nb in adj[tv]:
                        if mapping[nb] == -1:
                            if not has_available_host(nb):
                                fc_ok = False
                                break
                                
                    if fc_ok:
                        if backtrack(idx + 1):
                            return True
                    else:
                        # Undo and continue
                        pass # We undo below anyway
                    
                    # Undo
                    mapping[tv] = -1
                    host_used[hv] = False
                
                return False
            
            if backtrack(0):
                result[k_idx] = [mapping[i] for i in range(k_val)]
                for (u, v) in trees[k_idx]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
            else:
                # Failed to pack this tree. Skip it.
                pass
                
        return result, used_edges

    # Strategy 1: Pack in order of complexity (max degree desc)
    # This is a single pass. If it fails, we try restarts with different orderings?
    # Or random restarts with the same order?
    
    # Let's do random restarts with the sorted order.
    # The randomness comes from the tie-breaking in sorting? 
    # Or we can shuffle the order of trees with same complexity.
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        rng = Random(attempt * 7919 + 12345)
        
        # Create a slightly randomized order based on complexity
        # Group by max degree, shuffle within groups
        groups = {}
        for i in tree_indices:
            md = tree_max_deg[i]
            if md not in groups:
                groups[md] = []
            groups[md].append(i)
            
        # Sort groups by max degree descending
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
                
    # If we didn't get 190, try a different strategy: 
    # Random order of trees?
    if best_edges < 190:
        for attempt in range(500):
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
