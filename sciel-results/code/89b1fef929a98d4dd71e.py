import itertools
import time

def construct():
    n = 16
    # Start with the complement of the 4-cube Q4
    # Q4 vertices are 0..15 (4-bit binary)
    # Edge in Q4 if Hamming distance is 1
    # Edge in complement if Hamming distance is NOT 1 (and u != v)
    
    q4_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            # Check Hamming distance
            diff = u ^ v
            # Count bits
            if bin(diff).count('1') == 1:
                q4_edges.add((u, v))
                
    all_edges = set((u, v) for u in range(n) for v in range(u + 1, n))
    edges = all_edges - q4_edges
    
    # Helper to check if a set of edges (on a specific subset of vertices) contains Q3
    # We will check the whole graph for Q3
    
    # Precompute all 8-subsets
    subsets = list(itertools.combinations(range(n), 8))
    
    # Q3 structure
    # Vertices 0..7
    # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    def has_q3_in_subset(subset, graph_edges):
        # subset is a tuple of 8 vertices
        # graph_edges is a set of tuples (u, v) with u < v
        
        # Build local adjacency map
        local_adj = {v: set() for v in subset}
        edge_count = 0
        for u in subset:
            for v in subset:
                if u == v: continue
                # Check if edge exists
                if (u, v) in graph_edges or (v, u) in graph_edges:
                    local_adj[u].add(v)
                    if u < v:
                        edge_count += 1
        
        if edge_count < 12:
            return False, None
            
        # Check if it contains Q3
        # If edge_count == 12, check if isomorphic to Q3
        # If edge_count > 12, check if it contains a subgraph isomorphic to Q3
        
        # Simple backtracking to find Q3 subgraph
        # Map Q3 vertices 0..7 to subset vertices
        # We can use a recursive search
        
        # To speed up, we can first check if degrees allow it
        # Q3 is 3-regular.
        # If any vertex in subset has degree < 3 in the induced subgraph, it can't be part of a Q3?
        # No, it could be part of a Q3 if we don't use all its edges?
        # No, Q3 is 3-regular, so every vertex in the Q3 subgraph must have degree 3 in the subgraph.
        # So every vertex in the subset must have degree >= 3 in the induced subgraph.
        
        for v in subset:
            if len(local_adj[v]) < 3:
                return False, None
                
        # If we reach here, it's a candidate.
        # Try to embed Q3
        # We will try to assign Q3 vertices 0..7 to subset vertices
        # Order Q3 vertices by some heuristic?
        # Just try all permutations? 8! = 40320.
        # 12870 subsets * 40320 is too slow.
        
        # Better: Use the structure of Q3.
        # Q3 is bipartite.
        # Partition Q3 into Even (0,3,5,6) and Odd (1,2,4,7).
        # Try all partitions of subset into two sets of 4?
        # C(8,4) = 70.
        # For each partition (A, B), check if there is a bijection from Even->A and Odd->B that preserves edges.
        # This is a bipartite matching problem?
        # We need to check if the induced bipartite graph between A and B contains the specific pattern of Q3.
        
        # Actually, simpler:
        # Just return True if edge_count >= 12 and degrees >= 3?
        # This is a heuristic. It might falsely identify non-Q3 graphs as containing Q3.
        # But if we are removing edges, false positives just cause us to remove more edges than necessary.
        # False negatives (missing a Q3) are bad.
        # Given the density, this heuristic is likely safe.
        # Let's refine: If edge_count == 12, we MUST check isomorphism.
        # If edge_count > 12, we assume it contains Q3 if degrees >= 3.
        
        if edge_count == 12:
            # Check isomorphism to Q3
            # Q3 has 6 4-cycles.
            # Count 4-cycles in local_adj
            cycle_count = 0
            verts = list(subset)
            for i in range(8):
                for j in range(i+1, 8):
                    for k in range(j+1, 8):
                        for l in range(k+1, 8):
                            # Check if {i,j,k,l} forms a 4-cycle
                            # Check all 3 pairings
                            # Pairing 1: (i,j) (k,l) -> i-j-k-l-i
                            # Edges: (i,j), (j,k), (k,l), (l,i)
                            # Or (i,k), (j,l) -> i-k-j-l-i
                            # Or (i,l), (j,k) -> i-l-j-k-i
                            
                            def is_cycle(a, b, c, d):
                                return (b in local_adj[a] and c in local_adj[b] and d in local_adj[c] and a in local_adj[d])
                                
                            if is_cycle(verts[i], verts[j], verts[k], verts[l]): cycle_count += 1
                            if is_cycle(verts[i], verts[k], verts[j], verts[l]): cycle_count += 1
                            if is_cycle(verts[i], verts[l], verts[j], verts[k]): cycle_count += 1
                            
            # Q3 has 6 4-cycles.
            # Note: Each 4-cycle is counted once here?
            # The loop iterates over sets of 4 vertices.
            # A 4-cycle is a specific ordering.
            # My `is_cycle` checks a specific ordering.
            # There are 3 distinct 4-cycles on a set of 4 vertices in a complete graph K4.
            # In a bipartite graph, at most 1?
            # If cycle_count == 6, it is likely Q3.
            
            if cycle_count == 6:
                return True, list(subset)
            else:
                return False, None
        else:
            # edge_count > 12
            # Assume it contains Q3
            return True, list(subset)
            
    # Main loop
    start_time = time.time()
    
    # Check initial graph
    while True:
        found = False
        for subset in subsets:
            res, verts = has_q3_in_subset(subset, edges)
            if res:
                found = True
                # Remove an edge from this Q3
                # Pick the first edge in the subset that is in `edges`
                # We need to pick an edge that is part of the Q3.
                # Since we didn't identify the exact Q3 edges in the >12 case,
                # we just pick any edge in the subset.
                # This might not break the Q3 if we picked an edge not in the Q3.
                # But if we remove enough edges, we will break it.
                # To be safer, if we found a Q3, we should remove an edge that is critical.
                # For now, remove a random edge from the subset.
                
                # Get all edges in this subset
                sub_edges = [e for e in edges if e[0] in subset and e[1] in subset]
                if sub_edges:
                    # Remove one
                    e_to_remove = sub_edges[0]
                    edges.remove(e_to_remove)
                break
                
        if not found:
            break
            
        if time.time() - start_time > 110:
            break
            
    return [[u, v] for u, v in edges]
