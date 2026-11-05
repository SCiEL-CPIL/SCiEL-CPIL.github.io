import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s

    def time_exceeded():
        return time.time() - start_time > deadline

    # Precompute for q=29 (Paley coloring)
    q = 29

    def build_paley_colors(n):
        """Build Paley coloring for prime n ≡ 1 (mod 4)."""
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)

    def edge_index(n, i, j):
        """Get the index of edge (i,j) in the colors list, where i < j."""
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adj(n, colors):
        """Build bitmask adjacency for each color."""
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def find_k222_fast(n, adj):
        """
        Fast K_{2,2,2} check using bitmask operations.
        For color c, K_{2,2,2} exists iff there are 6 vertices {a,b,c,d,e,f}
        partitioned into 3 pairs such that all 12 cross-pair edges have color c.
        
        Equivalently: there exist vertices a,b (non-adjacent in c) with >=4 common
        neighbors, and among those common neighbors there exist pairs (k,l) and (m,o)
        such that k,l are non-adjacent in c AND m,o are non-adjacent in c.
        
        Actually more precisely: K_{2,2,2} in color c means 3 disjoint pairs
        (A,B), (C,D), (E,F) where every edge between different pairs has color c,
        and edges within each pair do NOT have color c (they're in the other color).
        
        Wait - the problem says "not necessarily induced" subgraph. So K_{2,2,2} as
        a subgraph means: 6 vertices partitioned into 3 pairs, and ALL 12 edges
        between vertices in different pairs must have the target color. Edges within
        pairs can be anything.
        
        So: find 6 vertices v1..v6 and a partition into 3 pairs such that all
        12 cross-pair edges have color c.
        
        Fast check: For each pair (i,j) that is NOT an edge in color c,
        compute common neighbors CN(i,j) in color c. We need to find 4 more
        vertices k,l,m,o in CN(i,j) such that:
        - {k,l} is not an edge in c (or is, doesn't matter for "not induced")
        - {m,o} is not an edge in c
        - All edges between {k,l} and {m,o} have color c
        - All edges between {k,l} and {i,j} have color c (guaranteed by CN)
        - All edges between {m,o} and {i,j} have color c (guaranteed by CN)
        
        Wait, I need to re-read. K_{2,2,2} has bipartition... no, K_{2,2,2} is
        the complete tripartite graph with parts of size 2. So vertices are
        partitioned into 3 groups of 2, and ALL edges between different groups
        are present (in the target color). Edges within groups may or may not be present.
        
        So for a fixed color c, K_{2,2,2} exists iff there are 3 disjoint pairs
        (a1,a2), (b1,b2), (c1,c2) such that for every pair of groups, all 4
        cross-edges have color c.
        
        Algorithm:
        For each unordered pair (i,j) in the first group:
          CN = common neighbors of i and j in color c (vertices adjacent to both i and j in c)
          We need to find 2 disjoint pairs (k1,k2) and (m1,m2) from CN such that
          all 4 edges between {k1,k2} and {m1,m2} have color c.
          
          This means: for the set S = {k1,k2,m1,m2} ⊆ CN, the induced subgraph on S
          in color c must contain all 4 edges between the two pairs.
          
          Equivalently: pick 4 vertices from CN. They form K_{2,2,2} with (i,j) if
          they can be partitioned into 2 pairs such that all 4 cross-edges exist in c.
          
          For 4 vertices {p,q,r,s} in CN, check all 3 ways to partition into 2 pairs:
          - (p,q)|(r,s): need edges p-r, p-s, q-r, q-s all in c
          - (p,r)|(q,s): need edges p-q, p-s, r-q, r-s all in c  
          - (p,s)|(q,r): need edges p-q, p-r, s-q, s-r all in c
          
          But we also need the pairs to be disjoint from (i,j), which they are since
          they're in CN (and CN excludes i,j... actually CN should exclude i,j).
        
        Let me implement this efficiently.
        """
        for c in range(2):
            adj_c = adj[c]
            for i in range(n):
                for j in range(i + 1, n):
                    # (i,j) is a pair in the K_{2,2,2} - they don't need to be non-adjacent
                    # since it's not induced. But they must both be connected to all
                    # vertices in the other two pairs.
                    # Common neighbors of i and j in color c:
                    cn = adj_c[i] & adj_c[j]
                    # Remove i and j themselves (they shouldn't be in their own neighborhood
                    # but let's be safe)
                    cn &= ~(1 << i) & ~(1 << j)
                    
                    # Get list of common neighbors
                    cn_list = []
                    mask = cn
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        cn_list.append(v)
                        mask ^= lsb
                    
                    if len(cn_list) < 4:
                        continue
                    
                    # Need to find 4 vertices from cn_list that can be split into
                    # 2 pairs with all 4 cross-edges in color c.
                    # For efficiency, precompute for each pair in cn_list whether
                    # they are connected in color c.
                    L = len(cn_list)
                    # connected[a][b] = True if cn_list[a] and cn_list[b] are both in adj_c
                    # and the edge between them has color c
                    # i.e., adj_c[cn_list[a]] has bit cn_list[b] set
                    # Actually we just need: for 4 vertices, check if they form K_{2,2}
                    # in color c (all 4 cross edges present).
                    
                    # For each 4-subset of cn_list, check if it contains a K_{2,2}
                    # in color c. A K_{2,2} in color c on 4 vertices means there's a
                    # partition into 2 pairs where all 4 cross-edges exist in c.
                    # This is equivalent to: the 4-vertex induced subgraph in c has
                    # at least 4 edges, and specifically... 
                    
                    # Actually, 4 vertices form a K_{2,2} subgraph (not necessarily induced)
                    # in color c iff there exist 2 disjoint pairs with all 4 cross-edges
                    # in c. This is true iff the complement (in c) on these 4 vertices
                    # has no perfect matching... no wait.
                    
                    # Let me just check: for 4 vertices a,b,c,d, they contain K_{2,2}
                    # in color c iff at least one of these 3 conditions holds:
                    # (a,b)|(c,d): edges a-c, a-d, b-c, b-d all in c
                    # (a,c)|(b,d): edges a-b, a-d, c-b, c-d all in c
                    # (a,d)|(b,c): edges a-b, a-c, d-b, d-c all in c
                    
                    # For speed, precompute the "c-adjacency" among cn_list vertices.
                    # Let's represent it as a bitmask for each vertex.
                    cn_adj = [0] * L  # cn_adj[a] has bit b set if cn_list[a]-cn_list[b] edge in c
                    for a in range(L):
                        va = cn_list[a]
                        for b in range(a + 1, L):
                            vb = cn_list[b]
                            if adj_c[va] & (1 << vb):
                                cn_adj[a] |= (1 << b)
                                cn_adj[b] |= (1 << a)
                    
                    # Now check all 4-subsets
                    for a in range(L - 3):
                        for b in range(a + 1, L - 2):
                            for c_idx in range(b + 1, L - 1):
                                for d in range(c_idx + 1, L):
                                    # Check if {a,b,c,d} contains K_{2,2} in c
                                    # Partition 1: (a,b)|(c,d)
                                    if (cn_adj[a] & (1 << c_idx)) and (cn_adj[a] & (1 << d)) and \
                                       (cn_adj[b] & (1 << c_idx)) and (cn_adj[b] & (1 << d)):
                                        return (c, (i, j, cn_list[a], cn_list[b], cn_list[c_idx], cn_list[d]))
                                    # Partition 2: (a,c)|(b,d)
                                    if (cn_adj[a] & (1 << b)) and (cn_adj[a] & (1 << d)) and \
                                       (cn_adj[c_idx] & (1 << b)) and (cn_adj[c_idx] & (1 << d)):
                                        return (c, (i, j, cn_list[a], cn_list[c_idx], cn_list[b], cn_list[d]))
                                    # Partition 3: (a,d)|(b,c)
                                    if (cn_adj[a] & (1 << b)) and (cn_adj[a] & (1 << c_idx)) and \
                                       (cn_adj[d] & (1 << b)) and (cn_adj[d] & (1 << c_idx)):
                                        return (c, (i, j, cn_list[a], cn_list[d], cn_list[b], cn_list[c_idx]))
        
        return (None, None)

    def find_k222(n, colors):
        """Find a K_{2,2,2} in either color class."""
        adj = build_adj(n, colors)
        return find_k222_fast(n, adj)

    # Verify Paley(29) is valid
    color, vertices = find_k222(q, colors_29)

    if color is None:
        # Paley(29) is valid! Try to extend to n=30.
        best_n = q
        best_colors = colors_29

        if not time_exceeded():
            random.seed(42)

            # Strategy: vertex doubling with Paley-guided initialization
            # For vertex 29 (new vertex), assign colors based on Paley structure
            # to minimize K_{2,2,2} violations.
            
            # Precompute adj for the 29-vertex Paley coloring
            adj_29 = build_adj(29, colors_29)
            
            # For each vertex i in 0..28, determine a "good" color for edge (i, 29)
            # Heuristic: count for each color c how many pairs (j,k) with j<k, j≠i, k≠i
            # would create a potential K_{2,2,2} if edge(i,29)=c.
            # This is expensive, so use a simpler heuristic:
            # For vertex 29, its color to vertex i should be chosen to balance
            # the degrees in each color class.
            
            max_attempts = 20
            for attempt in range(max_attempts):
                if time_exceeded():
                    break

                colors_30 = colors_29[:]

                if attempt == 0:
                    # Paley-inspired: use the "distance" in the cyclic group
                    # Vertex 29 is like "infinity" - connect based on some rule
                    for i in range(29):
                        # Use quadratic residue test with a shifted modulus
                        # or just use a pattern
                        colors_30.append(i % 2)
                elif attempt == 1:
                    # All 0s
                    for i in range(29):
                        colors_30.append(0)
                elif attempt == 2:
                    # All 1s
                    for i in range(29):
                        colors_30.append(1)
                elif attempt == 3:
                    # Based on degree balance: connect to vertices with high degree in color 0
                    # with color 1, and vice versa
                    deg0 = [bin(adj_29[0][i]).count('1') for i in range(29)]
                    deg1 = [bin(adj_29[1][i]).count('1') for i in range(29)]
                    for i in range(29):
                        if deg0[i] > deg1[i]:
                            colors_30.append(1)  # connect with color 1 to balance
                        else:
                            colors_30.append(0)
                elif attempt == 4:
                    # Opposite of attempt 3
                    deg0 = [bin(adj_29[0][i]).count('1') for i in range(29)]
                    deg1 = [bin(adj_29[1][i]).count('1') for i in range(29)]
                    for i in range(29):
                        if deg0[i] > deg1[i]:
                            colors_30.append(0)
                        else:
                            colors_30.append(1)
                else:
                    # Random initialization with bias
                    for i in range(29):
                        colors_30.append(random.randint(0, 1))

                # Local search with fast K_{2,2,2} check
                adj_30 = build_adj(30, colors_30)
                max_iter = 2000
                for iteration in range(max_iter):
                    if time_exceeded():
                        break

                    c, verts = find_k222_fast(30, adj_30)
                    if c is None:
                        return {'n': 30, 'colors': colors_30}

                    # Break the K_{2,2,2} by flipping a cross-pair edge
                    pairs = [(verts[0], verts[1]),
                             (verts[2], verts[3]),
                             (verts[4], verts[5])]

                    cross_edges = []
                    for p1 in range(3):
                        for p2 in range(p1 + 1, 3):
                            for v1 in pairs[p1]:
                                for v2 in pairs[p2]:
                                    cross_edges.append((v1, v2))

                    # Flip a random cross-pair edge
                    v1, v2 = random.choice(cross_edges)
                    if v1 > v2:
                        v1, v2 = v2, v1
                    idx = edge_index(30, v1, v2)
                    old_color = colors_30[idx]
                    new_color = 1 - old_color
                    colors_30[idx] = new_color

                    # Update adj efficiently
                    adj_30[old_color][v1] &= ~(1 << v2)
                    adj_30[old_color][v2] &= ~(1 << v1)
                    adj_30[new_color][v1] |= (1 << v2)
                    adj_30[new_color][v2] |= (1 << v1)

                # Final check
                adj_30 = build_adj(30, colors_30)
                c, verts = find_k222_fast(30, adj_30)
                if c is None:
                    return {'n': 30, 'colors': colors_30}

            # If we couldn't find n=30, return the best valid coloring
            return {'n': best_n, 'colors': best_colors}
        else:
            return {'n': q, 'colors': colors_29}
    else:
        # Paley(29) is NOT valid - need to find another construction
        random.seed(42)

        colors = colors_29[:]
        adj = build_adj(29, colors)
        max_iter = 3000
        for iteration in range(max_iter):
            if time_exceeded():
                break

            c, verts = find_k222_fast(29, adj)
            if c is None:
                return {'n': 29, 'colors': colors}

            pairs = [(verts[0], verts[1]),
                     (verts[2], verts[3]),
                     (verts[4], verts[5])]

            cross_edges = []
            for p1 in range(3):
                for p2 in range(p1 + 1, 3):
                    for v1 in pairs[p1]:
                        for v2 in pairs[p2]:
                            cross_edges.append((v1, v2))

            v1, v2 = random.choice(cross_edges)
            if v1 > v2:
                v1, v2 = v2, v1
            idx = edge_index(29, v1, v2)
            old_color = colors[idx]
            new_color = 1 - old_color
            colors[idx] = new_color

            adj[old_color][v1] &= ~(1 << v2)
            adj[old_color][v2] &= ~(1 << v1)
            adj[new_color][v1] |= (1 << v2)
            adj[new_color][v2] |= (1 << v1)

        return {'n': 29, 'colors': colors}
