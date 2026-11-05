"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    p = 29
    squares = {x * x % p for x in range(1, p)}
    
    # Generate colors in lexicographic order (i, j) with i < j
    base_colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                base_colors.append(0)
            else:
                base_colors.append(1)
    
    # Try to extend to n=30 using local search with bitmask representation
    n = 30
    num_edges = n * (n - 1) // 2
    
    # Build edge index map
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    def colors_to_bitmasks(colors_list):
        """Convert color list to bitmask neighbors for each vertex and color."""
        n = len(colors_list) * 2 // (len(colors_list) * 2 - 1) if False else 0
        # Reconstruct n from colors_list length
        # num_edges = n*(n-1)//2
        # Solve: n = (1 + sqrt(1+8*num_edges))/2
        ne = len(colors_list)
        n = int((1 + (1 + 8 * ne) ** 0.5) // 2)
        adj = [[0] * 2 for _ in range(n)]
        for (i, j), idx in edge_idx.items():
            c = colors_list[idx]
            adj[i][c] |= (1 << j)
            adj[j][c] |= (1 << i)
        return adj, n
    
    def has_k222(adj, n):
        """Check if either color class contains K_{2,2,2}.
        K_{2,2,2} = complete tripartite K_{2,2,2}.
        For a fixed color c, we need 3 disjoint pairs (A,B,C) such that
        all cross-edges between the pairs have color c.
        
        Alternative: For each pair of vertices (u,v), let N_c(u,v) = N_c(u) & N_c(v).
        If N_c(u,v) contains two disjoint pairs {a1,a2} and {b1,b2} such that
        all edges between {a1,a2} and {b1,b2} have color c, then {u,v,a1,a2,b1,b2}
        might form K_{2,2,2} if also all edges between {u,v} and {a1,a2,b1,b2} have color c.
        
        Actually simpler: K_{2,2,2} exists in color c iff there exist 6 vertices partitioned
        into 3 pairs such that all 12 cross-edges between different pairs have color c.
        
        Efficient check: For each pair of "pair-sets", but that's expensive.
        
        Let me use: For color c, K_{2,2,2} exists iff there exist vertices a,b,c,d,e,f
        such that all edges between {a,b},{c,d},{e,f} have color c.
        
        We can check: for each pair (a,b), let S = N_c(a) & N_c(b) (vertices connected to both a and b in color c).
        Then we need S to contain a K_{2,2} in color c, i.e., two disjoint pairs {c,d} and {e,f} in S
        such that all edges between {c,d} and {e,f} have color c.
        
        But wait, we also need that c,d are NOT connected to each other in color c? No, K_{2,2,2}
        doesn't require non-edges within parts. The parts are just the pairs, and edges within a pair
        can be anything. So we just need: there exist 6 distinct vertices a,b,c,d,e,f such that
        all edges between {a,b} and {c,d}, between {a,b} and {e,f}, and between {c,d} and {e,f}
        have color c.
        
        So for a fixed pair (a,b) in color c, let S = N_c(a) & N_c(b). We need S to contain
        two disjoint pairs {c,d} and {e,f} such that all edges between {c,d} and {e,f} have color c.
        This means the subgraph induced by S in color c must contain a K_{2,2}.
        
        K_{2,2} in a graph exists iff there are two vertices with common neighborhood of size >= 2
        that are not adjacent? No, K_{2,2} just needs two disjoint pairs with all cross-edges present.
        In the subgraph on S with edges of color c, K_{2,2} exists iff there are vertices c,d in S
        such that N_c(c) & N_c(d) & S has size >= 2 (and those two vertices are distinct from c,d).
        
        Actually, for K_{2,2}: we need c,d,e,f in S, all distinct, such that all four edges
        (c,e),(c,f),(d,e),(d,f) have color c.
        
        So: for fixed (a,b), compute S = N_c(a) & N_c(b). Then check if the color-c subgraph on S
        contains a K_{2,2}.
        
        To check K_{2,2} in a subgraph: for each pair (c,d) in S, compute T = N_c(c) & N_c(d) & S.
        If |T| >= 2 and T doesn't contain c or d (which it won't since c,d not in their own neighborhoods),
        then K_{2,2} exists.
        
        This is O(n^4) worst case but with bitmasks should be fast for n=30.
        """
        for c in range(2):
            for a in range(n):
                for b in range(a + 1, n):
                    # Check if edge (a,b) has color c
                    if not (adj[a][c] & (1 << b)):
                        continue
                    S = adj[a][c] & adj[b][c]
                    # S contains vertices connected to both a and b in color c
                    # Now check if color-c subgraph on S contains K_{2,2}
                    # For each pair (c2, d2) in S, check if they have >= 2 common neighbors in S
                    c2_list = []
                    mask = S
                    while mask:
                        lb = mask & (-mask)
                        v = lb.bit_length() - 1
                        c2_list.append(v)
                        mask ^= lb
                    for i2 in range(len(c2_list)):
                        for j2 in range(i2 + 1, len(c2_list)):
                            c2 = c2_list[i2]
                            d2 = c2_list[j2]
                            T = adj[c2][c] & adj[d2][c] & S
                            if bin(T).count('1') >= 2:
                                return True
        return False
    
    # Initialize: extend Paley(29) with vertex 29
    # For the new vertex 29, try different colorings of edges to existing vertices
    best_n = 29
    best_colors = base_colors[:]
    
    # Try to find a valid coloring for n=30
    # Start with a random initial coloring for edges incident to vertex 29
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        colors_list = base_colors[:]
        # Add edges for vertex 29
        for i in range(29):
            colors_list.append(random.randint(0, 1))
        
        adj, n_adj = colors_to_bitmasks(colors_list)
        n_adj = 30
        
        # Local search: flip edges to eliminate K_{2,2,2}
        improved = True
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not has_k222(adj, 30):
                # Found valid coloring for n=30!
                best_n = 30
                best_colors = colors_list[:]
                break
            # Find a K_{2,2,2} and flip an edge in it
            # For efficiency, just flip a random edge
            e = random.randint(0, num_edges - 1)
            colors_list[e] ^= 1
            # Rebuild adj (slow but n=30 is small)
            adj, _ = colors_to_bitmasks(colors_list)
        else:
            continue
    
    if best_n == 30:
        return {'n': 30, 'colors': best_colors}
    else:
        return {'n': 29, 'colors': base_colors}
# EVOLVE-BLOCK-END
