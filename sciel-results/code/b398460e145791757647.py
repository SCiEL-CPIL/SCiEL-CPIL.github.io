import random

def construct():
    # Paley graph construction for n=29
    n_base = 29
    q = n_base
    # Quadratic residues mod 29
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % q in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    # Verify base coloring
    if is_valid(n_base, colors_base):
        # Try to extend to n=30
        n_target = 30
        # We need to add 29 edges for the new vertex 29
        # Edges are (29, 0), (29, 1), ..., (29, 28)
        # In lexicographic order, these are the last 29 entries in the list of size C(30,2)=435
        
        # We will try to find a valid assignment for these 29 edges
        # We use a random restart strategy
        
        # To speed up, we can fix some edges to 0 to break symmetry?
        # The Paley graph is vertex transitive, so vertex 29 is equivalent to any other.
        # However, the existing graph is fixed.
        
        # Let's try 100 random assignments
        for _ in range(100):
            new_edges = [random.randint(0, 1) for _ in range(n_base)]
            # Construct full color list
            # The first C(29,2) = 406 entries are colors_base
            # The last 29 entries are new_edges
            # Wait, lexicographic order:
            # Pairs (i,j) with i<j.
            # For n=30, pairs involving 29 are (0,29), (1,29), ..., (28,29).
            # These are NOT the last 29 in a simple concatenation if we just append.
            # Let's check the order.
            # i=0: (0,1)...(0,28), (0,29)
            # i=1: (1,2)...(1,28), (1,29)
            # ...
            # i=28: (28,29)
            # So the edges involving 29 are scattered.
            
            # It is easier to build the full list from scratch or map indices.
            # Let's build a helper to get color for (i,j) from a list.
            # Or just generate the full list.
            
            full_colors = [0] * (30 * 29 // 2) # 435
            
            # Fill base part
            idx = 0
            for i in range(n_base):
                for j in range(i + 1, n_base):
                    full_colors[idx] = colors_base[idx]
                    idx += 1
            # This is wrong. The base colors are for n=29.
            # The indices for n=30 are different.
            
            # Let's just generate the full list directly in the loop.
            full_colors = []
            for i in range(30):
                for j in range(i + 1, 30):
                    if i < n_base and j < n_base:
                        # Use base coloring
                        # Need to map (i,j) in n=29 to index in colors_base
                        # Index in n=29: sum_{k=0}^{i-1} (29-1-k) + (j-i-1)
                        # Or simpler: just recompute? No, use the list.
                        # Let's precompute a mapping or just re-derive from squares.
                        if (j - i) % q in squares:
                            full_colors.append(0)
                        else:
                            full_colors.append(1)
                    else:
                        # One of i or j is 29
                        # Since i < j, j must be 29
                        # Edge is (i, 29)
                        # i ranges from 0 to 28
                        # We stored new_edges[i] for edge (i, 29)
                        full_colors.append(new_edges[i])
            
            if is_valid(30, full_colors):
                return {'n': 30, 'colors': full_colors}
                
        # If no extension found, return base
        return {'n': n_base, 'colors': colors_base}
    else:
        # Fallback to n=17 if 29 fails (should not happen)
        squares_17 = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'colors': [0 if (j - i) % 17 in squares_17 else 1 for i in range(17) for j in range(i + 1, 17)]}

def is_valid(n, colors):
    # Check if neither color class contains K2,2,2
    # Build adjacency matrix or list for each color
    # adj[c][i][j] = True if edge (i,j) has color c
    
    adj = [[False] * n for _ in range(n)] # adj[0]
    adj1 = [[False] * n for _ in range(n)] # adj[1]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj[i][j] = True
                adj[j][i] = True
            else:
                adj1[i][j] = True
                adj1[j][i] = True
            idx += 1
            
    # Check for K2,2,2 in adj[0]
    if has_k222(n, adj):
        return False
    # Check for K2,2,2 in adj[1]
    if has_k222(n, adj1):
        return False
        
    return True

def has_k222(n, adj):
    # Check if graph (n, adj) contains K2,2,2
    # Iterate over all pairs (u, v) as part A
    for u in range(n):
        for v in range(u + 1, n):
            # Find common neighbors of u and v
            # w such that adj[u][w] and adj[v][w]
            common = []
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    common.append(w)
            
            # We need to find two disjoint pairs (b1, b2) and (c1, c2) from common
            # such that all edges between {b1, b2} and {c1, c2} exist in adj
            # Note: b1, b2 are in common, so they are connected to u, v.
            # c1, c2 are in common, so they are connected to u, v.
            # We need edges between b1, b2 and c1, c2.
            
            # To optimize, we can iterate over pairs in common
            # But common size can be up to n-2.
            # n=30, common size ~14.
            # Pairs in common: C(14, 2) ~ 91.
            # For each pair B, find neighbors in common \ B.
            
            for i in range(len(common)):
                b1 = common[i]
                for j in range(i + 1, len(common)):
                    b2 = common[j]
                    # Find candidates for C: vertices in common \ {b1, b2}
                    # that are connected to both b1 and b2
                    candidates = []
                    for k in range(len(common)):
                        c = common[k]
                        if c == b1 or c == b2:
                            continue
                        if adj[b1][c] and adj[b2][c]:
                            candidates.append(c)
                    
                    if len(candidates) >= 2:
                        # We found c1, c2
                        # Check if we have a valid K2,2,2
                        # Parts: {u, v}, {b1, b2}, {c1, c2}
                        # Edges:
                        # u-v: Not required (same part)
                        # b1-b2: Not required (same part)
                        # c1-c2: Not required (same part)
                        # u-b, u-c, v-b, v-c: Yes (by definition of common)
                        # b-c: Yes (by definition of candidates)
                        return True
    return False
