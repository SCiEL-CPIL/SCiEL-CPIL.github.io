import time

def _paley_coloring(n):
    """
    Generate the coloring for the Paley graph on n vertices.
    n must be a prime p = 1 mod 4.
    Edge (i, j) is color 0 if (j-i) is a quadratic residue mod n, else 1.
    """
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(0 if d in squares else 1)
    return colors

def _has_k222(color_list, n, deadline=None):
    """
    Check if the coloring contains K2,2,2 in either color.
    Uses bitset adjacency for efficiency.
    Returns True if K2,2,2 found, False otherwise.
    """
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = color_list[idx]
            idx += 1
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
    
    for c in [0, 1]:
        adj = adj0 if c == 0 else adj1
        found = False
        # Iterate over all pairs (i, j)
        for i in range(n):
            if deadline and time.time() > deadline:
                # If we run out of time, we assume the coloring is invalid to be safe?
                # Or return False to accept? 
                # For a known valid coloring like Paley(29), this should never happen.
                # If it does, returning True (invalid) is safer for correctness, 
                # but might cause us to fallback to a smaller n.
                # However, Paley(29) is small, this loop is very fast.
                return True 
            for j in range(i + 1, n):
                # Common neighbors of i and j in color c
                common_mask = adj[i] & adj[j]
                # Exclude i and j themselves
                common_mask &= ~(1 << i) & ~(1 << j)
                
                # We need to find 4 distinct vertices in common_mask that form a K4 in the induced subgraph?
                # No, K2,2,2 is the octahedron. It consists of 6 vertices.
                # The condition for a monochromatic K2,2,2 in color c is:
                # There exist 3 disjoint pairs of vertices (a,b), (c,d), (e,f) such that
                # all edges between different pairs are color c.
                # Equivalently, the graph of color c contains K2,2,2.
                # K2,2,2 is the complement of 3 disjoint edges.
                # Another way: K2,2,2 is a subgraph if there are 6 vertices that can be partitioned into 3 pairs
                # such that every edge connecting two different pairs is present.
                
                # Let's stick to the search in the common neighborhood.
                # If we fix i and j, they form one "part" of the K2,2,2? No.
                # K2,2,2 has parts of size 2. Let the parts be P1, P2, P3.
                # If i, j are in P1, then we need to find P2 = {k, l} and P3 = {m, n}
                # such that k, l, m, n are all connected to i and j (in color c),
                # and k, l are NOT connected to each other? No, K2,2,2 is not induced.
                # "Not necessarily induced" means we just need the edges to be present.
                # In K2,2,2, there are NO edges within the parts.
                # So if i, j are in the same part, they must NOT be connected?
                # Wait, the definition of K2,2,2 (octahedron) is the complete 3-partite graph with parts of size 2.
                # Edges exist between all vertices in different parts.
                # No edges exist within a part.
                # So, if we find a subgraph isomorphic to K2,2,2, it means there is a set of 6 vertices
                # that can be partitioned into 3 pairs such that:
                # 1. No edges within any pair.
                # 2. All edges between any two different pairs exist.
                
                # My previous check in the provided code was:
                # "cand_mask = common_mask & adj[b1] & adj[b2]..."
                # This checks if b1, b2, v1, v2 form a K4 in the common neighborhood?
                # If b1, b2, v1, v2 are all connected to i and j, and form a K4 among themselves,
                # then {i, j} is one part? No, if they form a K4, they are all connected to each other.
                # In K2,2,2, vertices in the same part are NOT connected.
                # So if i, j are in the same part, they should NOT be connected by an edge of color c.
                # But we are iterating over ALL pairs (i, j). We don't know if they are in the same part.
                
                # Let's re-evaluate the K2,2,2 detection.
                # A graph G contains K2,2,2 if there exist 6 vertices v1..v6
                # that can be ordered as (a,b), (c,d), (e,f) such that:
                # - (a,b), (c,d), (e,f) are non-edges in G.
                # - All other pairs among these 6 vertices are edges in G.
                
                # Algorithm:
                # For each color c:
                #   For each pair (i, j) that is a NON-EDGE in G_c:
                #       Let N(i, j) be the set of vertices connected to BOTH i and j in G_c.
                #       We need to find two disjoint non-edges (k, l) and (m, n) within N(i, j)
                #       such that k, l, m, n are all connected to each other (forming a K4 in the induced subgraph on N(i,j) excluding the non-edges?).
                #       Actually, if k, l, m, n are in N(i, j), they are connected to i and j.
                #       We need (k, l) to be a non-edge, (m, n) to be a non-edge.
                #       And we need all edges between {k, l} and {m, n} to be present.
                #       i.e., k-m, k-n, l-m, l-n must be edges.
                #       This is equivalent to finding a K4 in the graph G_c induced by N(i, j) 
                #       where the edges of the K4 are NOT required to be non-edges?
                #       Wait. If (k, l) is a non-edge, and (m, n) is a non-edge,
                #       and k, l, m, n form a "K4 minus two opposite edges"? No.
                #       The edges present are:
                #       i-j: No
                #       k-l: No
                #       m-n: No
                #       i-k, i-l, i-m, i-n: Yes
                #       j-k, j-l, j-m, j-n: Yes
                #       k-m, k-n, l-m, l-n: Yes
                #       
                #       So we need to find 4 vertices in N(i, j) that can be partitioned into two pairs (k, l) and (m, n)
                #       such that (k, l) and (m, n) are non-edges in G_c,
                #       and the 4 cross edges k-m, k-n, l-m, l-n are edges in G_c.
                #       
                #       This is equivalent to finding a K4 in the complement of G_c induced by N(i, j)?
                #       No.
                #       Let's just iterate.
                
                S = []
                m = common_mask
                while m:
                    v = (m & -m).bit_length() - 1
                    S.append(v)
                    m &= m - 1
                
                s_len = len(S)
                if s_len < 4:
                    continue
                
                # We need to find two disjoint pairs (k, l) and (m, n) in S
                # such that:
                # 1. (k, l) is a non-edge in G_c (i.e., not in adj)
                # 2. (m, n) is a non-edge in G_c
                # 3. k-m, k-n, l-m, l-n are edges in G_c
                
                for bi in range(s_len):
                    b1 = S[bi]
                    for bj in range(bi + 1, s_len):
                        b2 = S[bj]
                        # Check if (b1, b2) is a non-edge
                        if adj[b1] & (1 << b2):
                            continue # It is an edge, so cannot be a pair in the same part
                
                        # Now find (b3, b4) in S, disjoint from {b1, b2}
                        # such that (b3, b4) is a non-edge
                        # and b1-b3, b1-b4, b2-b3, b2-b4 are edges.
                        
                        # Candidates for b3, b4 must be neighbors of b1 and b2.
                        cand_mask = common_mask & adj[b1] & adj[b2] & ~(1 << b1) & ~(1 << b2)
                        # cand_mask contains vertices in S that are connected to b1 and b2.
                        # We need to pick two from cand_mask that are NOT connected to each other.
                        
                        # Iterate through pairs in cand_mask
                        # To be efficient, we can iterate through vertices in cand_mask
                        # and for each, check if there is another vertex in cand_mask
                        # that is NOT connected to it.
                        
                        # Let's extract the list of candidates
                        C = []
                        mc = cand_mask
                        while mc:
                            v = (mc & -mc).bit_length() - 1
                            C.append(v)
                            mc &= mc - 1
                        
                        c_len = len(C)
                        for ci in range(c_len):
                            b3 = C[ci]
                            # Check if b3 is connected to b1 and b2 (already ensured by cand_mask)
                            # Now look for b4 in C[ci+1:]
                            for cj in range(ci + 1, c_len):
                                b4 = C[cj]
                                # Check if (b3, b4) is a non-edge
                                if adj[b3] & (1 << b4):
                                    continue
                                # If we get here, we found:
                                # (i, j) non-edge? Wait, did we check if (i, j) is a non-edge?
                                # In the outer loop, we iterate over ALL pairs (i, j).
                                # We MUST ensure (i, j) is a non-edge for this to be a valid K2,2,2 part.
                                # Let's add that check.
                                if adj[i] & (1 << j):
                                    # (i, j) is an edge, so they cannot be in the same part of K2,2,2.
                                    # So this configuration is invalid.
                                    continue
                                
                                found = True
                                break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # The known record is n=29. Paley(29) is a valid K2,2,2-free coloring.
    n = 29
    colors = _paley_coloring(n)
    
    # Validate with deadline guard. Paley(29) is known valid, so this should pass.
    # The check above is O(n^5) worst case but with bitsets and early termination, it's fast for n=29.
    if _has_k222(colors, n, deadline):
        # Fallback to smaller known-good Paley graph if validation fails
        n = 17
        colors = _paley_coloring(n)
    
    return {'n': n, 'colors': colors}
