import time
import random
import itertools

# Q3 edges in canonical form (0..7)
_Q3_CANONICAL_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

# Precompute all subsets of vertices of size 8 from 16? 
# No, that's 12870. 
# Better: Precompute all injective mappings of Q3 vertices to 16 vertices?
# 16P8 is huge.
# Alternative: For a given graph G, check if it contains Q3.
# Since we are adding edges one by one, we only need to check if the *new* edge completes a Q3.
# A Q3 containing edge (u,v) corresponds to an embedding of Q3 where u and v map to adjacent vertices in Q3.

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    # List of all possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    best_edges = []
    best_count = 0
    
    # We will run several random trials
    # Each trial: shuffle edges, add if no Q3 created
    
    # Optimization: 
    # To check if adding (u,v) creates a Q3, we need to see if there exists a set of 6 other vertices 
    # and an embedding of Q3 such that u,v are mapped to adjacent Q3 vertices and all other Q3 edges exist in G.
    
    # Precomputation for Q3 structure:
    # For each pair of adjacent vertices (i,j) in Q3, what are the other vertices and edges required?
    # Q3 vertices: 0..7
    # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    
    # Let's precompute for each edge in Q3, the set of other edges required.
    # If we map Q3 edge (i,j) to graph edge (u,v), we need to map the other 6 Q3 vertices to 6 distinct graph vertices
    # such that all other Q3 edges are present in the graph.
    
    # This is still complex to check quickly.
    # Alternative approach: 
    # Since n=16 is small, we can maintain the adjacency matrix.
    # When adding (u,v), check all possible embeddings of Q3 that include (u,v).
    # How many such embeddings?
    # Number of ways to map Q3 to 16 vertices is P(16,8) * (automorphisms).
    # But we fix u and v to specific positions in Q3.
    # There are 12 edges in Q3. For each edge, there are 2 orientations.
    # Total embeddings of a specific edge (u,v) into Q3 positions:
    # Choose which Q3 edge corresponds to (u,v). (12 choices)
    # Orient u,v (2 choices).
    # Map remaining 6 Q3 vertices to 6 distinct vertices from the remaining 14 graph vertices.
    # 14P6 = 14*13*12*11*10*9 = 2,162,160.
    # 12 * 2 * 2,162,160 = ~52 million checks per edge addition?
    # That is too slow for ~100 edge additions.
    
    # We need a faster check.
    
    # Let's look at the structure of Q3.
    # Q3 is bipartite.
    # If we add (u,v), u and v must be in opposite partitions of the Q3 embedding.
    
    # Alternative: Use a known Q3-free graph construction?
    # The seed K3,13 has 39 edges.
    # The observed child p6 scored 1.05 (40 edges?) and p4 scored 1.38 (54 edges?).
    # 1.38 * 39 = 53.82. So ~54 edges.
    
    # Can we find a 54+ edge Q3-free graph?
    # Turan graph T(16, 3) has 16/3 = 5.33 -> sizes 5,5,6.
    # Edges: 5*5 + 5*6 + 5*6 = 25 + 30 + 30 = 85.
    # Does T(16,3) contain Q3?
    # Q3 is bipartite with 4+4 vertices.
    # T(16,3) is 3-partite. It contains no odd cycles, but Q3 has 4-cycles.
    # T(16,3) definitely contains 4-cycles. Does it contain Q3?
    # A Q3 requires 8 vertices.
    # In T(16,3), max independent set is 6.
    # Q3 has independent sets of size 4.
    # Let's test T(16,3) for Q3.
    
    def check_q3_in_graph(adj):
        # adj is n x n boolean matrix
        # Check all subsets of 8 vertices? 16C8 = 12870.
        # For each subset, check if it induces a Q3 or contains a Q3?
        # "Containing a subgraph isomorphic to Q3" means there is a mapping of 8 vertices to the subset
        # such that all 12 edges of Q3 are present in the induced subgraph on those 8 vertices?
        # No, "subgraph" usually means the edges must be present in G, not necessarily induced.
        # But if we pick 8 vertices, we check if the graph induced by them contains a Q3 as a subgraph?
        # Actually, simpler: iterate over all injective mappings of Q3 vertices to G vertices?
        # Too many.
        
        # Iterate over all 8-vertex subsets of G.
        # For each subset S, let H be the subgraph of G induced by S.
        # Check if H contains a Q3.
        # Since |S|=8, H has at most 28 edges. Q3 has 12.
        # We can check if H is isomorphic to Q3? No, H could have extra edges.
        # We need to check if there is a subset of 12 edges in H that forms a Q3.
        # This is equivalent to checking if H has a Q3 subgraph.
        # Since H has 8 vertices, we can check all 8! = 40320 permutations of the 8 vertices?
        # Or simpler: 
        # The Q3 is the only 3-regular bipartite graph on 8 vertices?
        # No, there are others.
        
        # Efficient check for 8-vertex graph containing Q3:
        # We can precompute all 8-vertex graphs that contain a Q3? No.
        
        # Let's just check isomorphism to Q3 for the induced subgraph?
        # If the induced subgraph IS Q3, it contains Q3.
        # If it has extra edges, it still contains Q3 if we can find the 12 edges.
        
        # For a fixed set of 8 vertices, checking if it contains a Q3 subgraph:
        # We can try all mappings of Q3 vertices to these 8 vertices.
        # 8! = 40320 mappings.
        # For each mapping, check if the 12 edges exist in adj.
        # 12870 subsets * 40320 mappings * 12 edge checks = 6.2 billion operations. Too slow.
        
        # Optimization: 
        # Q3 is 3-regular.
        # If H contains Q3, then H must have at least 12 edges.
        # Also, the degrees of vertices in H must be at least 3? No, the subgraph Q3 has degree 3, 
        # but vertices in H might have higher degree.
        # However, if we look at the subgraph formed by the 12 edges of Q3, those vertices have degree 3 *within that subgraph*.
        
        # Let's use a different strategy for the check.
        # When adding edge (u,v), we only need to check Q3s that use (u,v).
        # As calculated before, ~52 million checks.
        # Can we reduce this?
        # Most Q3 embeddings will fail early.
        # We can use backtracking to find a Q3 embedding containing (u,v).
        # 
        # Let's implement a fast backtracker for "Does G contain Q3 using edge (u,v)?".
        
        pass # Placeholder
        
    # Let's try a different approach.
    # Use the random greedy method but with a fast Q3 check.
    # If the check is too slow, we can limit the number of trials or edges.
    
    # Actually, let's look at the score 1.38.
    # 1.38 * 39 = 53.8.
    # So we need ~54 edges.
    # K3,13 is 39.
    # Adding edges within the parts of K3,13?
    # Parts: A (size 3), B (size 13).
    # Edges in A: 3C2 = 3.
    # Edges in B: 13C2 = 78.
    # Total possible: 39 + 3 + 78 = 120.
    # We can add some edges in B and A.
    
    # Let's try to construct a graph based on K3,13 plus some internal edges.
    # Or better, just run the random greedy with a timeout.
    
    # To make the Q3 check fast, we can precompute the "forbidden configurations" for each edge?
    # No.
    
    # Let's try a specific construction known to be Q3-free or close.
    # The complement of a matching?
    # K16 minus 8 disjoint edges.
    # Edges: 120 - 8 = 112.
    # Does it contain Q3?
    # Q3 has 8 vertices.
    # If we pick 8 vertices, they might form a Q3.
    # In K16 - M, two vertices are non-adjacent only if they are matched.
    # Q3 is 3-regular.
    # In K16-M, degrees are 14 or 15.
    # It is very likely to contain Q3.
    
    # Let's go with Random Greedy + Fast Check.
    # To speed up the check:
    # When adding (u,v), we want to know if there is a Q3 containing (u,v).
    # We can iterate over all pairs of neighbors?
    
    # Let's try a simpler heuristic:
    # Start with K3,13.
    # Try to add all edges within the 13-set and 3-set in random order.
    # Check if adding edge creates Q3.
    
    # We need a fast `creates_q3(u, v, adj)`.
    
    # Precompute:
    # For each edge e in Q3, let N(e) be the set of other edges in Q3.
    # This doesn't help directly with vertex mapping.
    
    # Let's use the fact that Q3 is small.
    # We can hardcode the check for small n?
    
    # Alternative: Use `networkx`? No, standard library only.
    
    # Let's write a optimized backtracker.
    # 
    # Function: `has_q3_with_edge(u, v, adj)`
    # 1. Identify all embeddings of Q3 where u and v are mapped to adjacent vertices.
    # 2. For each such embedding, check if all edges exist.
    
    # To avoid iterating 52M embeddings, we can prune.
    # 
    # Let's try a different tactic: 
    # Run the random greedy construction, but if the Q3 check is too slow, 
    # we can skip it and just verify the final graph?
    # No, we must ensure validity.
    
    # Let's assume the Q3 check can be made fast enough with pruning.
    # 
    # Precompute Q3 adjacency lists.
    # q3_adj = {0: [1,2,4], 1: [0,3,5], ...}
    
    # When adding (u,v):
    # We try to map u->0, v->1. Then we must map the rest.
    # We also try u->1, v->0.
    # And all other edge pairs.
    
    # There are 12 edges in Q3.
    # For each edge (i,j) in Q3:
    #   Map u->i, v->j.
    #   Map u->j, v->i.
    #   Use backtracking to assign the other 6 Q3 vertices to distinct vertices in 0..15 \ {u,v}.
    #   At each step, check if edges to already assigned vertices exist in adj.
    
    # This is a CSP.
    # Variables: 6 Q3 vertices.
    # Domains: 14 vertices.
    # Constraints: 
    #   1. Distinctness.
    #   2. Edges to assigned vertices must exist in adj.
    #   3. Edges between unassigned vertices must be consistent? (Checked later or during).
    
    # This should be very fast if adj is dense (many edges exist, so constraints are loose? 
    # Wait, if adj is dense, it's MORE likely to have Q3. 
    # If adj is sparse, constraints are tight (few edges exist), so backtracking prunes fast.
    # Since we start with K3,13 (sparse) and add edges, it stays somewhat sparse initially.
    
    # Let's implement this.

    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    def creates_q3(u, v, adj):
        # Try all 24 orientations (12 edges * 2 directions)
        for i, j in q3_edges:
            # Case 1: u->i, v->j
            if _check_embedding(i, j, u, v, adj, q3_adj):
                return True
            # Case 2: u->j, v->i
            if _check_embedding(j, i, u, v, adj, q3_adj):
                return True
        return False
        
    def _check_embedding(q3_u, q3_v, g_u, g_v, adj, q3_adj):
        # q3_u, q3_v are adjacent in Q3.
        # g_u, g_v are the corresponding vertices in G.
        # We need to map the remaining 6 Q3 vertices to 6 distinct G vertices.
        
        # Precompute which Q3 vertices are already mapped
        mapped_q3 = {q3_u, q3_v}
        mapped_g = {g_u, g_v}
        
        # Unmapped Q3 vertices
        unmapped_q3 = [x for x in range(8) if x not in mapped_q3]
        
        # For backtracking, order unmapped_q3 by degree of constraints?
        # Just use fixed order for now.
        
        # We will use recursion.
        # To speed up, we can check if any unmapped Q3 vertex has an edge to a mapped Q3 vertex
        # that doesn't exist in G.
        
        # Let's use a simple recursive backtracker.
        
        # Optimization: 
        # If time is tight, we might need to bail out.
        
        # We pass the current mapping as a dict q3_vertex -> g_vertex
        current_mapping = {q3_u: g_u, q3_v: g_v}
        
        def backtrack(idx):
            if idx == len(unmapped_q3):
                # All mapped. Check if all Q3 edges are present in G.
                # Actually, we should have checked edges to mapped vertices during backtracking.
                # We still need to check edges between unmapped vertices that are now all mapped.
                # But if we check edges to *all* previously mapped vertices at each step, 
                # then by the end, all edges are checked.
                return True
                
            q3_v = unmapped_q3[idx]
            
            # Candidate G vertices
            candidates = [x for x in range(16) if x not in mapped_g]
            
            # Prune candidates:
            # For each mapped Q3 neighbor of q3_v, the corresponding G vertex must be adjacent to candidate.
            for q3_neighbor in q3_adj[q3_v]:
                if q3_neighbor in current_mapping:
                    g_neighbor = current_mapping[q3_neighbor]
                    # Filter candidates to those adjacent to g_neighbor
                    candidates = [c for c in candidates if adj[c][g_neighbor]]
                    if not candidates:
                        return False
            
            for g_cand in candidates:
                # Assign
                current_mapping[q3_v] = g_cand
                mapped_g.add(g_cand)
                
                if backtrack(idx + 1):
                    return True
                    
                # Unassign
                del current_mapping[q3_v]
                mapped_g.remove(g_cand)
                
            return False
            
        return backtrack(0)

    # Run random greedy
    trials = 10
    for t in range(trials):
        if time.time() > deadline:
            break
            
        random.seed(t)
        adj = [[False] * n for _ in range(n)]
        
        # Start with K3,13? Or empty?
        # Starting with K3,13 gives a good baseline.
        for u in range(3):
            for v in range(3, 16):
                adj[u][v] = True
                adj[v][u] = True
                
        edges = list(itertools.combinations(range(n), 2))
        # Remove existing edges
        existing = set()
        for u in range(3):
            for v in range(3, 16):
                existing.add((u,v))
                existing.add((v,u))
                
        # We want to add edges.
        # Let's shuffle the remaining edges.
        remaining_edges = [e for e in edges if e[0] != e[1] and (e[0], e[1]) not in existing]
        random.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            if not creates_q3(u, v, adj):
                adj[u][v] = True
                adj[v][u] = True
                
        # Count edges
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    count += 1
                    
        if count > best_count:
            best_count = count
            # Store best graph
            best_adj = [row[:] for row in adj]
            
    # Convert best_adj to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
                
    return result
