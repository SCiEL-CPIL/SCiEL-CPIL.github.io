import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n = 16
    random.seed(42)
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Q3 structure: 8 vertices, 12 edges
    # Vertices labeled 0..7, edges form a cube
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (4, 5), (4, 7),
        (3, 7),
        (6, 7)
    ]
    q3_adj = [0] * 8
    for u, v in q3_edges:
        q3_adj[u] |= (1 << v)
        q3_adj[v] |= (1 << u)
    
    # Precompute all 8-subsets
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    # For each 8-subset, precompute the list of (subset_idx, q3_vertex) pairs
    # that could potentially map, based on degree constraints
    # But we'll do a faster check: use bitmask representation
    
    def contains_q3(adj):
        """Check if graph with adjacency masks adj contains Q3 as subgraph."""
        # Try all 8-subsets
        for subset in all_8_subsets:
            # Build induced subgraph adjacency
            sub_adj = [0] * 8
            deg = [0] * 8
            for i in range(8):
                u = subset[i]
                for j in range(8):
                    if i != j:
                        v = subset[j]
                        if (adj[u] >> v) & 1:
                            sub_adj[i] |= (1 << j)
                            deg[i] += 1
            
            # All Q3 vertices have degree 3, so all must have deg >= 3
            if any(d < 3 for d in deg):
                continue
            
            # Try to find isomorphism using backtracking
            # Order Q3 vertices by degree in the remaining unmapped part
            # For speed, use a fixed order but with pruning
            
            mapping = [-1] * 8  # q3_vertex -> sub_idx
            used_mask = 0
            
            # Precompute for each q3 vertex, its neighbors
            q3_neigh = q3_adj
            
            def backtrack(qv):
                if qv == 8:
                    return True
                
                qn = q3_neigh[qv]
                # Candidates: unused sub vertices
                cand = ~used_mask & 0xFF
                
                while cand:
                    lsb = cand & -cand
                    sv = lsb.bit_length() - 1
                    cand ^= lsb
                    
                    # Check compatibility with already mapped neighbors
                    ok = True
                    for qu in range(8):
                        if (qn >> qu) & 1:
                            if mapping[qu] != -1:
                                if not ((sub_adj[sv] >> mapping[qu]) & 1):
                                    ok = False
                                    break
                    
                    if not ok:
                        continue
                    
                    # Additional pruning: check if sv has enough available neighbors
                    # for unmapped neighbors of qv
                    unmapped_neigh_count = 0
                    for qu in range(8):
                        if (qn >> qu) & 1 and mapping[qu] == -1:
                            unmapped_neigh_count += 1
                    
                    # sv must be adjacent to at least unmapped_neigh_count unused vertices
                    # that are also candidates for those unmapped q3 vertices
                    # This is complex; skip for now
                    
                    mapping[qv] = sv
                    if backtrack(qv + 1):
                        return True
                    mapping[qv] = -1
                    
                return False
            
            if backtrack(0):
                return True
        
        return False
    
    # Greedy initialization: start with a good base
    # Use union of cliques strategy: K7 + K7 + K2 gives 43 edges, no Q3
    adj = [0] * n
    edges_set = set()
    
    # Clique 0: vertices 0-6
    for i in range(7):
        for j in range(i+1, 7):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            edges_set.add((i, j))
    
    # Clique 1: vertices 7-13
    for i in range(7, 14):
        for j in range(i+1, 14):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            edges_set.add((i, j))
    
    # Edge between 14 and 15
    adj[14] |= (1 << 15)
    adj[15] |= (1 << 14)
    edges_set.add((14, 15))
    
    # Verify no Q3
    if contains_q3(adj):
        # Should not happen, but fallback
        print("Warning: initial graph contains Q3!")
    
    current_edges = len(edges_set)
    best_adj = adj[:]
    best_edges = edges_set.copy()
    best_count = current_edges
    
    # Local search: try to add edges, if creates Q3, remove some edges
    # Simulated annealing style
    
    # Precompute for each edge, the 8-subsets containing it
    # This speeds up checking if adding an edge creates Q3
    subsets_containing_edge = {e: [] for e in all_edges}
    for s in all_8_subsets:
        for i in range(8):
            for j in range(i+1, 8):
                u, v = s[i], s[j]
                if u > v:
                    u, v = v, u
                subsets_containing_edge[(u, v)].append(s)
    
    def contains_q3_with_new_edge(adj, u, v):
        """Check if adding edge (u,v) to adj creates Q3. Only check subsets containing u,v."""
        if u > v:
            u, v = v, u
        for subset in subsets_containing_edge[(u, v)]:
            # Check if this subset contains Q3 in the graph with edge (u,v) added
            sub_adj = [0] * 8
            deg = [0] * 8
            for i in range(8):
                ui = subset[i]
                for j in range(8):
                    if i != j:
                        vj = subset[j]
                        # Check if edge exists in adj, or if it's the new edge (u,v)
                        if ui == u and vj == v:
                            sub_adj[i] |= (1 << j)
                            deg[i] += 1
                        elif ui == v and vj == u:
                            sub_adj[i] |= (1 << j)
                            deg[i] += 1
                        elif (adj[ui] >> vj) & 1:
                            sub_adj[i] |= (1 << j)
                            deg[i] += 1
            
            if any(d < 3 for d in deg):
                continue
            
            mapping = [-1] * 8
            used_mask = 0
            q3_neigh = q3_adj
            
            def backtrack(qv):
                if qv == 8:
                    return True
                
                qn = q3_neigh[qv]
                cand = ~used_mask & 0xFF
                
                while cand:
                    lsb = cand & -cand
                    sv = lsb.bit_length() - 1
                    cand ^= lsb
                    
                    ok = True
                    for qu in range(8):
                        if (qn >> qu) & 1:
                            if mapping[qu] != -1:
                                if not ((sub_adj[sv] >> mapping[qu]) & 1):
                                    ok = False
                                    break
                    
                    if not ok:
                        continue
                    
                    mapping[qv] = sv
                    if backtrack(qv + 1):
                        return True
                    mapping[qv] = -1
                    
                return False
            
            if backtrack(0):
                return True
        
        return False
    
    # Local search
    temperature = 0.5
    cooling = 0.999
    min_temp = 0.01
    max_iter = 100000
    iter_count = 0
    
    # List of missing edges
    missing_edges = [e for e in all_edges if e not in edges_set]
    
    while time.time() < deadline - 5 and temperature > min_temp:
        if iter_count >= max_iter:
            break
        iter_count += 1
        
        if time.time() > deadline - 5:
            break
        
        # Choose a random missing edge to try adding
        if not missing_edges:
            break
        
        idx = random.randint(0, len(missing_edges) - 1)
        u, v = missing_edges[idx]
        
        # Try adding edge (u, v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges_set.add((u, v))
        
        if contains_q3_with_new_edge(adj, u, v):
            # Creates Q3. Remove this edge and possibly another to escape
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            edges_set.remove((u, v))
            
            # With some probability, remove a random existing edge to explore
            if random.random() < 0.3 and edges_set:
                existing = list(edges_set)
                rem_u, rem_v = random.choice(existing)
                adj[rem_u] ^= (1 << rem_v)
                adj[rem_v] ^= (1 << rem_u)
                edges_set.remove((rem_u, rem_v))
                missing_edges.append((rem_u, rem_v))
        else:
            # Successfully added edge
            missing_edges.pop(idx)
            
            if len(edges_set) > best_count:
                best_adj = adj[:]
                best_edges = edges_set.copy()
                best_count = len(edges_set)
        
        temperature *= cooling
    
    # Final greedy pass to add any remaining edges
    for u, v in all_edges:
        if time.time() > deadline - 2:
            break
        if (u, v) in edges_set:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges_set.add((u, v))
        
        if contains_q3_with_new_edge(adj, u, v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            edges_set.remove((u, v))
        else:
            if len(edges_set) > best_count:
                best_adj = adj[:]
                best_edges = edges_set.copy()
                best_count = len(edges_set)
    
    # Return the best graph found
    result = [list(e) for e in best_edges]
    return result
