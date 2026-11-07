import time
import random
import sys

sys.setrecursionlimit(100000)

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
    deadline = start_time + 115
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20
    
    # Precompute tree data
    tree_data = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        max_deg = max(deg) if k > 1 else 0
        tree_data.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'max_deg': max_deg
        })
        
    # Sort trees: highest max_deg first, then largest k
    # This helps with pruning
    order = sorted(range(n_trees), key=lambda x: (-tree_data[x]['max_deg'], -tree_data[x]['k']))
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    # Global state
    used_edges = set()
    remaining_deg = [19] * n_vertices
    mappings = [None] * n_trees
    
    def is_edge_free(u, v):
        if u == v: return False
        e = (min(u, v), max(u, v))
        return e not in used_edges

    def pack_tree(t_idx):
        if time.time() > deadline:
            return False
        
        if t_idx == n_trees:
            return True
            
        td = tree_data[order[t_idx]]
        k = td['k']
        adj = td['adj']
        deg = td['deg']
        edges = td['edges']
        
        if k == 1:
            # Assign to any vertex with remaining_deg > 0? 
            # Actually, T1 has 0 edges. It doesn't consume capacity.
            # But it must be mapped to a distinct vertex? 
            # "list of k distinct int host vertices"
            # For k=1, just pick one.
            # To be safe and consistent, pick 0 if available? 
            # Since it consumes no edges, it doesn't matter for edge disjointness.
            # But we must return a list of 1 vertex.
            mappings[order[t_idx]] = [0]
            return pack_tree(t_idx + 1)
            
        mapping = [-1] * k
        used_verts = set()
        
        # Determine vertex ordering for backtracking
        # Start with max degree vertex
        # Then MRV
        # We will build the order dynamically or pre-compute a static order?
        # Dynamic MRV is better.
        
        # Helper to get candidates for a tree vertex tv
        def get_candidates(tv):
            cands = []
            d_tv = deg[tv]
            # Count mapped neighbors
            mapped_neighbors = [nb for nb in adj[tv] if mapping[nb] != -1]
            num_mapped = len(mapped_neighbors)
            
            # Capacity required: d_tv - num_mapped
            # Because edges to mapped neighbors are already checked/committed?
            # Wait, if we commit edges as we go, the capacity is already reduced.
            # So remaining_deg[h] is the capacity for *new* edges.
            # The new edges for tv are those to unmapped neighbors.
            # So we need remaining_deg[h] >= d_tv - num_mapped.
            
            req_cap = d_tv - num_mapped
            
            for h in range(n_vertices):
                if h in used_verts:
                    continue
                if remaining_deg[h] < req_cap:
                    continue
                
                # Check edges to mapped neighbors
                valid = True
                for nb in mapped_neighbors:
                    hn = mapping[nb]
                    if not is_edge_free(h, hn):
                        valid = False
                        break
                if valid:
                    cands.append(h)
            return cands

        def backtrack(pos):
            if time.time() > deadline:
                return False
                
            if pos == k:
                # All vertices mapped.
                # Edges were committed during the process?
                # No, we only checked. We need to commit them now?
                # Or commit as we assign?
                # Committing as we assign is better for pruning.
                # Let's assume we commit edges when a vertex is assigned.
                # But we haven't committed them in the code above.
                # Let's modify: when we assign h to tv, we add edges to used_edges
                # and update remaining_deg.
                
                # If we commit as we go, we must undo on backtrack.
                # This is complex.
                # Alternative: Commit only when the whole tree is placed.
                # This reduces pruning power during the tree placement.
                # Given k <= 20, and strong degree pruning, maybe it's okay?
                # But edge conflicts might be detected late.
                
                # Let's stick to "Commit as we go" for better pruning.
                # We need to track which edges were added for this tree.
                pass
            
            # Find MRV vertex
            best_tv = -1
            best_cands = None
            min_c = float('inf')
            
            for tv in range(k):
                if mapping[tv] != -1:
                    continue
                cands = get_candidates(tv)
                if not cands:
                    return False
                if len(cands) < min_c:
                    min_c = len(cands)
                    best_tv = tv
                    best_cands = cands
                    if min_c == 1:
                        break
                        
            if best_tv == -1:
                # All mapped?
                if all(mapping[i] != -1 for i in range(k)):
                    # Commit edges
                    added_edges = []
                    for u, v in edges:
                        eu, ev = mapping[u], mapping[v]
                        e = (min(eu, ev), max(eu, ev))
                        used_edges.add(e)
                        added_edges.append(e)
                        remaining_deg[eu] -= 1
                        remaining_deg[ev] -= 1
                        
                    mappings[order[t_idx]] = mapping[:]
                    
                    if pack_tree(t_idx + 1):
                        return True
                    
                    # Undo
                    for e in added_edges:
                        used_edges.remove(e)
                        remaining_deg[e[0]] += 1
                        remaining_deg[e[1]] += 1
                        
                    mappings[order[t_idx]] = None
                    return False
                else:
                    return False

            # Randomize
            random.shuffle(best_cands)
            
            for h in best_cands:
                mapping[best_tv] = h
                used_verts.add(h)
                
                # Tentatively commit edges to mapped neighbors?
                # If we do, we must undo.
                # Let's NOT commit edges to mapped neighbors immediately.
                # We will commit all edges at the end of the tree.
                # This is safer and simpler to undo.
                
                if backtrack(pos + 1):
                    return True
                    
                mapping[best_tv] = -1
                used_verts.remove(h)
                
            return False
        
        return backtrack(0)

    # Symmetry breaking: Fix first tree's max degree vertex to host 0
    # Find the first tree in order
    first_t_idx = order[0]
    first_td = tree_data[first_t_idx]
    if first_td['k'] > 1:
        # Find max degree vertex in this tree
        max_deg_v = first_td['deg'].index(max(first_td['deg']))
        # We will force mapping[max_deg_v] = 0
        # We can do this by pre-filling mapping[0] = 0 in the backtrack for the first tree
        # But the backtrack function is generic.
        # We can modify the initial call or the candidate generation for the first tree.
        pass

    # Run search
    # Try a few restarts if needed, but with symmetry breaking, one run might suffice.
    # If not, we can try different random seeds for shuffling.
    
    success = pack_tree(0)
    
    if not success:
        # Try a few more times with different random seeds for shuffling
        for seed in range(10):
            random.seed(seed)
            # Reset state
            used_edges.clear()
            remaining_deg = [19] * n_vertices
            mappings = [None] * n_trees
            if pack_tree(0):
                success = True
                break
            if time.time() > deadline - 5:
                break

    # Calculate score
    score = 0
    for i in range(n_trees):
        if mappings[i] is not None:
            score += len(trees[i])
            
    if score > best_score:
        best_score = score
        best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]

    final_mappings = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            final_mappings.append(None)
        else:
            final_mappings.append([int(v) for v in best_mappings[i]])
            
    return final_mappings
