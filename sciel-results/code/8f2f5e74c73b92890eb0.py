import time
import random

def construct():
    # Time limit guard
    start_time = time.time()
    time_limit = 145  # Leave 5s margin
    
    # 1. Construct known valid n=29 Paley graph
    # q = 29 is a prime, 29 % 4 == 1
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = [0 if (j - i) % q in squares else 1 for i in range(q) for j in range(i + 1, q)]
    best_n = 29
    best_colors = colors_29
    
    # 2. Attempt to find n=30 coloring
    # This is a heuristic search. If it fails, we return n=29.
    # Given the difficulty, we might not find it.
    # We'll try a simple simulated annealing / hill climbing.
    
    n_target = 30
    edges = n_target * (n_target - 1) // 2
    
    def is_valid(colors, n):
        # Check if colors list (length n*(n-1)/2) has no K2,2,2 in either color
        # This is slow, so we only call it occasionally.
        # Build adjacency matrix
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Check for K2,2,2
        # A K2,2,2 is defined by 3 pairs of vertices (a,b), (c,d), (e,f)
        # such that all edges between the pairs are of the same color.
        # Iterate over all combinations of 6 vertices
        # There are C(30, 6) = 593775 combinations.
        # For each, check if it contains a K2,2,2.
        # A set of 6 vertices contains a K2,2,2 if there is a partition into 3 pairs
        # such that all 12 cross-edges are color 0 OR all 12 are color 1.
        
        # Optimization: Precompute for each pair (u,v) the colors to all other vertices?
        # Let's just iterate.
        from itertools import combinations
        
        for verts in combinations(range(n), 6):
            # Check if this 6-set contains a K2,2,2
            # We need to check all 15 partitions into 3 pairs.
            # Partitions of {0,1,2,3,4,5} into 3 pairs:
            # (0,1)(2,3)(4,5), (0,1)(2,4)(3,5), etc.
            
            # Helper to check a specific partition
            def check_partition(p1, p2, p3):
                # p1, p2, p3 are tuples of 2 vertices
                c0 = 0
                c1 = 0
                # Edges between p1 and p2
                for u in p1:
                    for v in p2:
                        if adj[u][v] == 0: c0 += 1
                        else: c1 += 1
                # Edges between p1 and p3
                for u in p1:
                    for v in p3:
                        if adj[u][v] == 0: c0 += 1
                        else: c1 += 1
                # Edges between p2 and p3
                for u in p2:
                    for v in p3:
                        if adj[u][v] == 0: c0 += 1
                        else: c1 += 1
                        
                if c0 == 12 or c1 == 12:
                    return True
                return False
            
            # Generate all 15 partitions of 6 elements into 3 pairs
            # Fixed first pair (0,1) to avoid duplicates, then partition remaining 4
            # Pairs for remaining 4: (2,3)(4,5), (2,4)(3,5), (2,5)(3,4)
            # This covers all 3 partitions where 0 and 1 are paired.
            # But 0 and 1 might not be paired in the K2,2,2.
            # We must check all 15 partitions.
            
            v0, v1, v2, v3, v4, v5 = verts
            # 15 partitions:
            # 1: (0,1)(2,3)(4,5)
            if check_partition((v0,v1), (v2,v3), (v4,v5)): return False
            # 2: (0,1)(2,4)(3,5)
            if check_partition((v0,v1), (v2,v4), (v3,v5)): return False
            # 3: (0,1)(2,5)(3,4)
            if check_partition((v0,v1), (v2,v5), (v3,v4)): return False
            # 4: (0,2)(1,3)(4,5)
            if check_partition((v0,v2), (v1,v3), (v4,v5)): return False
            # 5: (0,2)(1,4)(3,5)
            if check_partition((v0,v2), (v1,v4), (v3,v5)): return False
            # 6: (0,2)(1,5)(3,4)
            if check_partition((v0,v2), (v1,v5), (v3,v4)): return False
            # 7: (0,3)(1,2)(4,5)
            if check_partition((v0,v3), (v1,v2), (v4,v5)): return False
            # 8: (0,3)(1,4)(2,5)
            if check_partition((v0,v3), (v1,v4), (v2,v5)): return False
            # 9: (0,3)(1,5)(2,4)
            if check_partition((v0,v3), (v1,v5), (v2,v4)): return False
            # 10: (0,4)(1,2)(3,5)
            if check_partition((v0,v4), (v1,v2), (v3,v5)): return False
            # 11: (0,4)(1,3)(2,5)
            if check_partition((v0,v4), (v1,v3), (v2,v5)): return False
            # 12: (0,4)(1,5)(2,3)
            if check_partition((v0,v4), (v1,v5), (v2,v3)): return False
            # 13: (0,5)(1,2)(3,4)
            if check_partition((v0,v5), (v1,v2), (v3,v4)): return False
            # 14: (0,5)(1,3)(2,4)
            if check_partition((v0,v5), (v1,v3), (v2,v4)): return False
            # 15: (0,5)(1,4)(2,3)
            if check_partition((v0,v5), (v1,v4), (v2,v3)): return False
            
        return True

    # Search for n=30
    # We use a random initial coloring and try to reduce violations.
    # Since full check is expensive, we use a random restart approach with limited iterations.
    
    found_30 = False
    max_search_time = 30 # seconds
    
    if time.time() - start_time < max_search_time:
        for restart in range(10): # 10 restarts
            if time.time() - start_time > max_search_time:
                break
            
            # Initialize random coloring
            current_colors = [random.randint(0, 1) for _ in range(edges)]
            
            # Simple hill climbing
            # We don't have a fast delta, so we'll do random flips and check validity periodically.
            # This is inefficient but might stumble upon a solution if one exists and is "close".
            # Actually, without a good gradient, random search on 435 bits is hopeless.
            
            # Better: Start from a known structure? 
            # The Paley graph on 29 is a subgraph of K30? 
            # We can extend the n=29 coloring to n=30 by adding a 30th vertex.
            # We need to assign colors to edges (29, j) for j=0..28.
            # This is a constraint satisfaction problem on the new edges.
            
            # Let's try extending the n=29 solution.
            # Vertices 0..28 are colored as in Paley(29).
            # Vertex 29 is new.
            # Edges (29, j) for j=0..28 need colors.
            # Total 29 new edges.
            # We can try all $2^{29}$ combinations? No.
            # We can use backtracking or local search on these 29 edges.
            
            # Let's try random assignments for the 29 new edges and check validity.
            # Validity check for n=30 is expensive, but we can optimize it.
            
            # Optimize is_valid for the specific case of extending n=29.
            # Most K2,2,2s will involve the new vertex 29 or not.
            # If a K2,2,2 does not involve 29, it was already present in n=29 (which is valid).
            # So we only need to check K2,2,2s that involve vertex 29.
            
            # A K2,2,2 involving 29 must have 29 in one of the pairs, or in a pair that is connected to others.
            # Actually, 29 is just a vertex.
            # A K2,2,2 is a set of 6 vertices.
            # If 29 is in the set, we check that specific 6-set.
            # Number of 6-sets containing 29: C(29, 5) = 118755.
            # This is much smaller than 593775.
            
            def is_valid_extension(new_edge_colors):
                # new_edge_colors is a list of 29 colors for edges (29, j)
                # Build adj matrix for n=30
                adj = [[0]*30 for _ in range(30)]
                idx = 0
                for i in range(29):
                    for j in range(i+1, 29):
                        c = colors_29[idx]
                        adj[i][j] = c
                        adj[j][i] = c
                        idx += 1
                for j in range(29):
                    c = new_edge_colors[j]
                    adj[29][j] = c
                    adj[j][29] = c
                
                # Check all 6-sets containing 29
                from itertools import combinations
                # Choose 5 other vertices from 0..28
                for others in combinations(range(29), 5):
                    verts = tuple(sorted([29] + list(others)))
                    v0, v1, v2, v3, v4, v5 = verts
                    
                    # Check 15 partitions
                    # Reuse logic from is_valid but hardcoded for speed?
                    # For now, use the same check_partition logic.
                    
                    def check_part(p1, p2, p3):
                        c0 = 0
                        c1 = 0
                        for u in p1:
                            for v in p2:
                                if adj[u][v] == 0: c0 += 1
                                else: c1 += 1
                        for u in p1:
                            for v in p3:
                                if adj[u][v] == 0: c0 += 1
                                else: c1 += 1
                        for u in p2:
                            for v in p3:
                                if adj[u][v] == 0: c0 += 1
                                else: c1 += 1
                        return c0 == 12 or c1 == 12

                    if check_part((v0,v1), (v2,v3), (v4,v5)): return False
                    if check_part((v0,v1), (v2,v4), (v3,v5)): return False
                    if check_part((v0,v1), (v2,v5), (v3,v4)): return False
                    if check_part((v0,v2), (v1,v3), (v4,v5)): return False
                    if check_part((v0,v2), (v1,v4), (v3,v5)): return False
                    if check_part((v0,v2), (v1,v5), (v3,v4)): return False
                    if check_part((v0,v3), (v1,v2), (v4,v5)): return False
                    if check_part((v0,v3), (v1,v4), (v2,v5)): return False
                    if check_part((v0,v3), (v1,v5), (v2,v4)): return False
                    if check_part((v0,v4), (v1,v2), (v3,v5)): return False
                    if check_part((v0,v4), (v1,v3), (v2,v5)): return False
                    if check_part((v0,v4), (v1,v5), (v2,v3)): return False
                    if check_part((v0,v5), (v1,v2), (v3,v4)): return False
                    if check_part((v0,v5), (v1,v3), (v2,v4)): return False
                    if check_part((v0,v5), (v1,v4), (v2,v3)): return False
                    
                return True

            # Try random extensions
            # 2^29 is too large.
            # We can try a few hundred random ones.
            for _ in range(100):
                if time.time() - start_time > max_search_time:
                    break
                new_colors = [random.randint(0, 1) for _ in range(29)]
                if is_valid_extension(new_colors):
                    # Construct full colors list
                    full_colors = colors_29 + new_colors
                    # Note: The order in the output must be lexicographic for all pairs (i,j) i<j.
                    # colors_29 covers (i,j) for 0<=i<j<29.
                    # new_colors covers (j, 29) for 0<=j<29? 
                    # Lexicographic order for n=30:
                    # (0,1)...(0,28), (0,29), (1,2)...(1,28), (1,29), ...
                    # My colors_29 is in order (0,1)..(0,28), (1,2)..(1,28) etc.
                    # The new edges are (0,29), (1,29), ..., (28,29).
                    # In lexicographic order, (0,29) comes after (0,28).
                    # (1,29) comes after (1,28).
                    # So we cannot just append new_colors.
                    
                    # We need to interleave them.
                    final_colors = []
                    idx29 = 0
                    for i in range(29):
                        for j in range(i+1, 29):
                            final_colors.append(colors_29[idx29]) # Wait, idx29 logic is wrong.
                            # colors_29 is a flat list.
                            # Let's just rebuild the flat list properly.
                    # Rebuild:
                    final_colors = []
                    idx = 0
                    for i in range(29):
                        for j in range(i+1, 29):
                            final_colors.append(colors_29[idx])
                            idx += 1
                        # Edge (i, 29)
                        final_colors.append(new_colors[i])
                    
                    return {'n': 30, 'colors': final_colors}
                    
    # Return best found (n=29)
    return {'n': best_n, 'colors': best_colors}
