import time
import sys

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Try n=29 with Paley graph
    # If valid, return it.
    # Fallback to n=17 if invalid or time runs out (though generation is fast).
    
    def check_validity(n, colors):
        # Check if any color class contains K2,2,2
        # K2,2,2 is a 3-partite graph with parts A, B, C of size 2.
        # Vertices: a1, a2, b1, b2, c1, c2
        # Edges: all pairs between different parts.
        # No edges within parts.
        # So we need to find 6 vertices {v1..v6} that can be partitioned into 3 pairs 
        # such that all 12 edges between the pairs are the same color.
        
        # This is a heavy check for n=29 (combinations of 6 from 29 is ~475k).
        # For each combination, check if it's a K2,2,2.
        # A set of 6 vertices induces a subgraph.
        # It is a K2,2,2 if and only if it can be partitioned into 3 independent sets of size 2
        # and all edges between these sets exist.
        
        # Optimization: 
        # Iterate over all pairs (i, j) as potential part A.
        # If color(i,j) == c, then i and j are NOT connected in the other color.
        # Wait, K2,2,2 requires edges between parts to be color c.
        # So if (i,j) is color 0, they cannot be in the same part of a color-0 K2,2,2?
        # No, in K2,2,2, vertices in the same part are NOT connected.
        # So if we look for a color 0 K2,2,2, the pairs (a1,a2), (b1,b2), (c1,c2) must have color 1 (or be non-edges in color 0).
        # And all edges between these pairs must be color 0.
        
        # So:
        # 1. Pick a pair (i, j) with color 1. (This is a candidate for part A).
        # 2. Find all vertices k such that color(i,k)==0 and color(j,k)==0. (Candidates for B and C).
        # 3. Among these candidates, we need to find two disjoint pairs (b1,b2) and (c1,c2) 
        #    such that:
        #    - color(b1,b2)==1
        #    - color(c1,c2)==1
        #    - All edges between {b1,b2} and {c1,c2} are color 0.
        #    - Also edges between A and B, A and C are 0 (already ensured by step 2).
        
        # This check is feasible for n=29.
        
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
                
        # Check for color 0 K2,2,2
        # Part A: pair (i,j) with adj[i][j] == 1
        # Part B: pair (k,l) with adj[k][l] == 1
        # Part C: pair (m,o) with adj[m][o] == 1
        # All cross edges must be 0.
        
        # Precompute candidates for each pair (i,j) with adj[i][j]==1
        # Candidates are vertices v such that adj[i][v]==0 and adj[j][v]==0.
        
        for c in [0, 1]:
            # We are looking for K2,2,2 in color c.
            # This means edges WITHIN parts are NOT c (i.e., 1-c).
            # Edges BETWEEN parts ARE c.
            
            # Step 1: Identify all "non-edges" of color c (i.e., edges of color 1-c)
            # These form the potential parts.
            # Let's iterate over potential Part A (i,j) where adj[i][j] == 1-c.
            
            # To make it faster, we can just iterate over all 6-tuples? No, too slow.
            # Use the intersection method.
            
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] != 1-c:
                        continue
                    
                    # i, j is a candidate for Part A.
                    # Find all v such that adj[i][v] == c and adj[j][v] == c.
                    # These v's are candidates for Parts B and C.
                    candidates = []
                    for v in range(n):
                        if v == i or v == j:
                            continue
                        if adj[i][v] == c and adj[j][v] == c:
                            candidates.append(v)
                    
                    # We need to pick two disjoint pairs (b1,b2) and (c1,c2) from candidates
                    # such that:
                    # 1. adj[b1][b2] == 1-c
                    # 2. adj[c1][c2] == 1-c
                    # 3. All 4 edges between {b1,b2} and {c1,c2} are c.
                    
                    # Iterate over pairs in candidates for Part B
                    # To avoid duplicates, we can enforce an ordering, but for correctness check, 
                    # finding ONE violation is enough.
                    
                    found_b = False
                    for k in range(len(candidates)):
                        b1 = candidates[k]
                        for l in range(k+1, len(candidates)):
                            b2 = candidates[l]
                            if adj[b1][b2] != 1-c:
                                continue
                            
                            # b1, b2 is a candidate for Part B.
                            # Now find Part C from remaining candidates.
                            # Remaining candidates are those not in {b1, b2}.
                            # Also, C must be connected to A and B with color c.
                            # A is connected to all candidates by definition.
                            # B is connected to all candidates? 
                            # We only checked A. We need to ensure adj[b1][v]==c and adj[b2][v]==c for v in C.
                            
                            valid_c_candidates = []
                            for m in range(len(candidates)):
                                c1 = candidates[m]
                                if c1 == b1 or c1 == b2:
                                    continue
                                if adj[b1][c1] == c and adj[b2][c1] == c:
                                    valid_c_candidates.append(c1)
                            
                            # Now pick two from valid_c_candidates for Part C
                            for p in range(len(valid_c_candidates)):
                                c1 = valid_c_candidates[p]
                                for q in range(p+1, len(valid_c_candidates)):
                                    c2 = valid_c_candidates[q]
                                    if adj[c1][c2] != 1-c:
                                        continue
                                    # Check edges between B and C are already checked? 
                                    # Yes, adj[b1][c1]==c, adj[b2][c1]==c etc were checked in valid_c_candidates selection.
                                    # We just need to ensure adj[c1][c2] == 1-c (checked).
                                    # And edges between A and C are c (ensured by candidates definition).
                                    # And edges between A and B are c (ensured by candidates definition).
                                    # So this is a valid K2,2,2 in color c.
                                    return False
        return True

    # Generate Paley 29
    q = 29
    squares = {x*x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i+1, q):
            # Color 0 if difference is square, else 1
            # Note: (j-i) % q or (i-j) % q? Squares are symmetric.
            d = (j - i) % q
            if d in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)
                
    if check_validity(q, colors_29):
        return {'n': q, 'colors': colors_29}
    
    # Fallback to Paley 17
    q = 17
    squares = {x*x % q for x in range(1, q)}
    colors_17 = []
    for i in range(q):
        for j in range(i+1, q):
            d = (j - i) % q
            if d in squares:
                colors_17.append(0)
            else:
                colors_17.append(1)
                
    return {'n': q, 'colors': colors_17}
