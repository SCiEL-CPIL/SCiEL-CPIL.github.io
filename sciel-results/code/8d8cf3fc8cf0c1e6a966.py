import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit to be safe

    n = 29
    # Quadratic residues mod 29
    residues = set()
    for x in range(1, 29):
        residues.add((x * x) % 29)
    
    # Generate colors
    # colors[i] corresponds to pair (v_i, v_j) in lexicographic order
    # We need to map pair (i, j) to index in colors list
    # Index for (i, j) with i < j:
    # Number of pairs before i: i * (n - 1) - i * (i - 1) / 2 ? No.
    # Pairs involving 0: (0,1)..(0,n-1) -> n-1 pairs
    # Pairs involving 1: (1,2)..(1,n-1) -> n-2 pairs
    # ...
    # Pairs involving i: (i, i+1)..(i, n-1) -> n-1-i pairs
    # Total pairs before i: sum_{k=0}^{i-1} (n-1-k) = i(n-1) - (i-1)i/2
    # Offset for j: j - i - 1
    
    def get_index(i, j):
        return i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1)
        
    colors = [0] * (n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors[get_index(i, j)] = 0
            else:
                colors[get_index(i, j)] = 1
                
    # Verification function
    def is_valid(n, colors):
        # Build adjacency bitmasks for color 0 and 1
        # adj0[i] is a bitmask of vertices connected to i via color 0
        # adj1[i] is a bitmask of vertices connected to i via color 1
        adj0 = [0] * n
        adj1 = [0] * n
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                    
        # Check for mono K2,2,2
        # K2,2,2 has parts A={a1,a2}, B={b1,b2}, C={c1,c2}
        # All edges between parts must be present.
        # Equivalently:
        # For any 6 distinct vertices, if they form K2,2,2 in color c, invalid.
        # Faster check:
        # For each pair (u, v), let N_u_v be the set of common neighbors in color c.
        # If |N_u_v| >= 4, check if N_u_v contains a K2,2 in the "complement" sense?
        # No, K2,2,2 is not induced. It just requires edges to exist.
        # If we pick u, v from part A, then all 4 vertices in B and C must be connected to u and v in color c.
        # So, if there exist u, v such that the set of common neighbors S has size >= 4,
        # AND S can be partitioned into two sets B, C of size 2 such that all edges between B and C exist in color c?
        # No, that's not right.
        # In K2,2,2, every vertex in A is connected to every vertex in B and C.
        # Every vertex in B is connected to every vertex in A and C.
        # Every vertex in C is connected to every vertex in A and B.
        # There are NO edges required within A, within B, or within C.
        # So, we need to find 6 vertices a1,a2,b1,b2,c1,c2 such that:
        # Edges (ai, bj), (ai, ck), (bj, ck) all exist in color c for all i,j,k in {1,2}.
        
        # Algorithm:
        # Iterate over pairs (u, v).
        # Let S = common neighbors of u and v in color c.
        # If |S| < 4, skip.
        # We need to pick 4 vertices from S, say b1,b2,c1,c2, such that
        # Edges (b1,c1), (b1,c2), (b2,c1), (b2,c2) exist in color c.
        # This is equivalent to saying that the subgraph induced by S in color c contains a K2,2.
        # So, for each pair (u,v) in color c, check if the common neighborhood S contains a K2,2 in color c.
        
        def contains_k22_in_subset(S_mask, adj_mask, n):
            # S_mask is a bitmask of vertices in S
            # adj_mask is the adjacency matrix (list of masks) for color c
            # We need to find b1,b2,c1,c2 in S such that edges exist between {b1,b2} and {c1,c2}
            # This is checking if the induced subgraph on S has a K2,2.
            # A K2,2 exists if there are two vertices with at least 2 common neighbors in the induced subgraph?
            # No. K2,2 is a 4-cycle.
            # It exists if there are two vertices x, y in S such that |N(x) ∩ N(y) ∩ S| >= 2.
            # Let's verify: If x, y have common neighbors z, w in S, then edges x-z, x-w, y-z, y-w exist.
            # This forms a K2,2 with parts {x,y} and {z,w}.
            # So yes, checking for two vertices with >=2 common neighbors in S is sufficient.
            
            # Iterate over pairs in S
            s_list = [i for i in range(n) if (S_mask >> i) & 1]
            for idx_b1 in range(len(s_list)):
                b1 = s_list[idx_b1]
                for idx_b2 in range(idx_b1 + 1, len(s_list)):
                    b2 = s_list[idx_b2]
                    # Common neighbors of b1 and b2 in S
                    common = (adj_mask[b1] & adj_mask[b2]) & S_mask
                    # Count bits
                    if bin(common).count('1') >= 2:
                        return True
            return False

        # Check color 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj0[u] >> v) & 1: # Edge u-v is color 0
                    S = adj0[u] & adj0[v]
                    if bin(S).count('1') >= 4:
                        if contains_k22_in_subset(S, adj0, n):
                            return False
        # Check color 1
        for u in range(n):
            for v in range(u + 1, n):
                if (adj1[u] >> v) & 1: # Edge u-v is color 1
                    S = adj1[u] & adj1[v]
                    if bin(S).count('1') >= 4:
                        if contains_k22_in_subset(S, adj1, n):
                            return False
                            
        return True

    if is_valid(n, colors):
        return {'n': n, 'colors': colors}
    else:
        # Fallback to n=17 if 29 fails (should not happen)
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        colors_17 = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
        return {'n': 17, 'colors': colors_17}
