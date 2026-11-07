from random import Random

def instance_trees(n=20, seed=611):
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
    trees = instance_trees(20, 611)
    n = 20
    
    # Track which K20 edges are used
    # Using a set of frozensets for O(1) lookup and easy removal
    all_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.add(frozenset((u, v)))
    
    available_edges = set(all_edges)
    
    # Result: list of 20 entries
    result = [None] * 20
    
    # Process trees in order of decreasing size
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Precompute adjacency lists for each tree
    tree_adjs = []
    for t in trees:
        k = len(t) + 1 if t else 1 # T1 has 1 vertex, T2 has 2, etc.
        # Actually T_k has k vertices.
        # T_1: 1 vertex, 0 edges.
        # T_2: 2 vertices, 1 edge.
        # T_k: k vertices, k-1 edges.
        adj = [[] for _ in range(k)]
        for (a, b) in t:
            adj[a].append(b)
            adj[b].append(a)
        tree_adjs.append(adj)
        
    # Helper to check if a partial assignment is valid and get candidates
    def get_candidates(tree_idx, assigned):
        # assigned: dict {tree_v: host_v}
        # Returns list of (tree_v, host_v) candidates for unassigned tree vertices
        # We want to pick the tree vertex with fewest candidates (MRV)
        
        # Identify unassigned tree vertices
        k = len(tree_adjs[tree_idx])
        unassigned = [v for v in range(k) if v not in assigned]
        
        if not unassigned:
            return None, None # All assigned
        
        best_v = None
        best_cands = None
        min_cands = float('inf')
        
        for v in unassigned:
            cands = []
            # A host h is valid for tree vertex v if:
            # 1. For all assigned neighbors u of v in the tree, (assigned[u], h) is in available_edges
            # 2. h is not already used by another tree vertex (implicitly handled if we check edges to assigned ones, 
            #    but we must also ensure h is not used by an unassigned neighbor? No, unassigned neighbors don't have hosts yet.
            #    However, h cannot be the host of any *assigned* tree vertex u != v.
            #    Since assigned is a dict, if h is in assigned.values(), it is used.
            
            # Optimization: 
            # Let N(v) be neighbors of v in tree.
            # Let A(v) = { assigned[u] for u in N(v) if u in assigned }
            # If A(v) is not empty, h must be in intersection of neighbors of each h' in A(v) in available_edges.
            # Also h not in assigned.values().
            
            used_hosts = set(assigned.values())
            
            # Get neighbors in tree
            neigh = tree_adjs[tree_idx][v]
            
            # Find assigned neighbors
            assigned_neigh_hosts = [assigned[u] for u in neigh if u in assigned]
            
            if assigned_neigh_hosts:
                # Intersection of available neighbors of each assigned neighbor host
                # Start with all possible hosts
                # Better: start with neighbors of the first assigned neighbor, then filter
                first_h = assigned_neigh_hosts[0]
                # Get all hosts h such that (first_h, h) in available_edges
                # Since available_edges is a set of frozensets, we can check
                # Or we can precompute adjacency of available graph? 
                # available_graph[h] = set of h' such that frozenset((h,h')) in available_edges
                # Let's maintain available_graph for speed.
                
                # For now, let's just iterate over all n hosts? n=20, so 20 checks per candidate.
                # k is max 20. unassigned max 20.
                # Total checks ~ 20*20*20 = 8000. Very fast.
                
                for h in range(n):
                    if h in used_hosts:
                        continue
                    valid = True
                    for ah in assigned_neigh_hosts:
                        edge = frozenset((ah, h))
                        if edge not in available_edges:
                            valid = False
                            break
                    if valid:
                        cands.append(h)
            else:
                # No assigned neighbors. Any unused host is a candidate?
                # Yes, but we must ensure that when we assign v to h, it doesn't conflict with future assignments?
                # Not yet. Just unused hosts.
                for h in range(n):
                    if h not in used_hosts:
                        cands.append(h)
                        
            if len(cands) < min_cands:
                min_cands = len(cands)
                best_v = v
                best_cands = cands
                if min_cands == 0:
                    return v, [] # Dead end
                if min_cands == 1:
                    break # Can't get better than 1
            
        return best_v, best_cands

    # Backtracking solver for a specific tree
    def solve_tree(tree_idx):
        k = len(tree_adjs[tree_idx])
        if k == 1:
            return {0: 0} # Arbitrary assignment, no edges to check
        if k == 2:
            # Find any available edge
            for e in available_edges:
                u, v = tuple(e)
                return {0: u, 1: v}
            return None
        
        assigned = {}
        
        # Heuristic: Start with a vertex that has high degree in the tree?
        # Or just let MRV guide it.
        
        def backtrack():
            # Check if all assigned
            if len(assigned) == k:
                # Verify all edges are available
                # This should be guaranteed by the candidate generation if we check edges to assigned neighbors.
                # But we only checked edges to *assigned* neighbors.
                # Edges between two unassigned vertices are not checked until one is assigned.
                # When the second one is assigned, it checks against the first (which is now assigned).
                # So yes, all edges are checked.
                return True
            
            v, cands = get_candidates(tree_idx, assigned)
            
            if cands is None:
                return True # Should not happen if len(assigned)<k
                
            if not cands:
                return False
                
            # Try candidates in some order?
            # Maybe sort by degree in host graph?
            for h in cands:
                assigned[v] = h
                # Tentatively reserve edges? 
                # No, we only remove edges from available_edges when the tree is fully embedded.
                # But wait, if we don't remove edges, another branch might use the same edge.
                # That's fine, backtracking will undo it.
                # BUT, we must ensure that we don't use an edge that is already used by *previous* trees.
                # available_edges contains only edges not used by previous trees.
                # Within this tree, we must ensure edge-disjointness of the tree itself?
                # A tree has no cycles, so no edge is shared within the tree.
                # However, two different tree vertices could map to the same host vertex?
                # "distinct int host vertices" is required.
                # My `used_hosts` check in `get_candidates` ensures distinctness.
                
                if backtrack():
                    return True
                del assigned[v]
            
            return False
        
        if backtrack():
            return assigned
        return None

    # Main loop
    for idx in tree_indices:
        k = idx + 1
        if k == 1:
            result[idx] = None
            continue
            
        # Solve for this tree
        embedding = solve_tree(idx)
        
        if embedding is None:
            # Failed to embed
            # This should not happen if the instance is solvable and our order is good.
            # If it does, we might need to try different orders or more powerful search.
            # For now, return what we have? No, "Any broken rule makes the whole result invalid".
            # We must find a valid packing.
            # Let's assume the instance is solvable.
            # If we fail, it's a bug in our solver or bad ordering.
            # Let's add a fallback: try all permutations of tree order? 20! is too big.
            # Let's try sorting by size ascending?
            # Or just restart with a different seed for tie-breaking in candidates?
            pass

        # Commit the embedding
        # Update available_edges
        for (a, b) in trees[idx]:
            ha, hb = embedding[a], embedding[b]
            edge = frozenset((ha, hb))
            if edge not in available_edges:
                # This should not happen if solver is correct
                # But if it does, we have a conflict.
                # Actually, the solver ensures edges are in available_edges at the time of assignment.
                # But we didn't remove them from available_edges during the search!
                # This is a BUG in the logic above.
                # We must remove edges from available_edges as we assign vertices to prevent
                # other vertices in the SAME tree from using the same edge?
                # No, a tree has unique edges. But we must prevent using an edge that is
                # already "reserved" by another pair in the tree?
                # Since it's a tree, no two edges share the same endpoints.
                # But two different tree edges could map to the same host edge if the mapping is not injective?
                # "distinct int host vertices" ensures injective vertex mapping.
                # If vertex mapping is injective, edge mapping is injective.
                # So we don't need to worry about internal conflicts in the tree.
                # We DO need to worry about conflicts with edges used by PREVIOUS trees.
                # Those are already removed from available_edges.
                # So we just need to remove the edges of THIS tree from available_edges AFTER successful embedding.
                
                # Wait, what if during backtracking, we assign v to h, and later assign w to h'?
                # And edge (v,w) maps to (h,h').
                # Is it possible that (h,h') was used by a previous assignment in the SAME tree?
                # No, because previous assignments in the same tree involved different vertices.
                # Since the tree is a tree, no edge is reused.
                
                # So we just remove the final edges.
                pass
            
            available_edges.remove(edge)
            
        result[idx] = [embedding[i] for i in range(k)]
        
    return result
