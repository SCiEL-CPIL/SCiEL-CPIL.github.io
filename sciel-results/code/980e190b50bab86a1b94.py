"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for return overhead
    
    # Base: Paley(29)
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
    
    # Try to extend to n=30
    n = 30
    num_edges = n * (n - 1) // 2
    
    # Build edge index map
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
            
    # Reverse map for quick access
    idx_edge = [None] * num_edges
    for (i, j), idx_val in edge_idx.items():
        idx_edge[idx_val] = (i, j)

    def init_state(colors_list):
        """Initialize adjacency bitmasks and intersection bitmasks."""
        # adj[c][v] is a bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(2)]
        for idx_val, c in enumerate(colors_list):
            i, j = idx_edge[idx_val]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        # inter[c][u][v] = adj[c][u] & adj[c][v]
        # We only need this for u < v to save space/time, or just full n x n
        # n=30, so 30x30 is fine.
        inter = [[0] * n for _ in range(n)] # inter[c] will be a list of lists? 
        # Let's use inter[c][u][v]
        inter = [[[0] * n for _ in range(n)] for _ in range(2)]
        for c in range(2):
            for u in range(n):
                for v in range(u + 1, n):
                    inter[c][u][v] = adj[c][u] & adj[c][v]
                    inter[c][v][u] = inter[c][u][v]
        
        return adj, inter

    def update_edge(idx_val, new_color, adj, inter):
        """Update adj and inter after flipping edge idx_val to new_color."""
        i, j = idx_edge[idx_val]
        old_color = 1 - new_color
        
        # Remove from old color
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        
        # Add to new color
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)
        
        # Update inter for pairs involving i or j
        # inter[c][x][y] changes if x==i or y==i or x==j or y==j
        
        # For old color, the edge (i,j) is removed.
        # This affects inter[old_color][i][k] for all k != i
        # and inter[old_color][j][k] for all k != j
        # and inter[old_color][k][i] and inter[old_color][k][j]
        
        # Actually, inter[c][u][v] = adj[c][u] & adj[c][v].
        # If we change adj[c][i], then inter[c][i][v] and inter[c][v][i] change for all v.
        # If we change adj[c][j], then inter[c][j][v] and inter[c][v][j] change for all v.
        
        # Since we flip (i,j), adj[old_color][i] loses j, adj[old_color][j] loses i.
        # adj[new_color][i] gains j, adj[new_color][j] gains i.
        
        # Update inter for old_color
        # For any v, inter[old_color][i][v] = adj[old_color][i] & adj[old_color][v]
        # Since adj[old_color][i] changed, we recompute inter[old_color][i][v] for all v.
        # Similarly for j.
        
        for c in [old_color, new_color]:
            # Update row i
            for v in range(n):
                if v == i: continue
                inter[c][i][v] = adj[c][i] & adj[c][v]
                inter[c][v][i] = inter[c][i][v]
            # Update row j
            for v in range(n):
                if v == j: continue
                inter[c][j][v] = adj[c][j] & adj[c][v]
                inter[c][v][j] = inter[c][j][v]

    def has_violation(adj, inter):
        """Check for K222 in either color. Returns (found, edge_idx_to_flip)."""
        for c in range(2):
            for u in range(n):
                # Only check if u has at least 4 neighbors in color c?
                # K222 requires u to connect to 4 others.
                if adj[c][u].bit_count() < 4:
                    continue
                for v in range(u + 1, n):
                    # Check if (u,v) is color c
                    if not (adj[c][u] & (1 << v)):
                        continue
                    # S = common neighbors of u and v in color c
                    S = inter[c][u][v]
                    # We need to find a,b in S such that |N_c(a) & N_c(b) & S| >= 2
                    # Iterate a in S
                    mask_a = S
                    while mask_a:
                        lb_a = mask_a & (-mask_a)
                        a = lb_a.bit_length() - 1
                        mask_a ^= lb_a
                        
                        # Iterate b in S, b > a? No, just b != a.
                        # To avoid duplicates, we can just check all b in S.
                        # But if we find one, we return.
                        mask_b = S
                        while mask_b:
                            lb_b = mask_b & (-mask_b)
                            b = lb_b.bit_length() - 1
                            mask_b ^= lb_b
                            if a == b:
                                continue
                            
                            # Check if there are at least 2 common neighbors of a and b in S
                            # common_ab_in_S = inter[c][a][b] & S
                            if (inter[c][a][b] & S).bit_count() >= 2:
                                # Violation found involving u, v, a, b
                                # We can flip an edge. Which one?
                                # Flipping (u,v) might be good as it breaks the top level.
                                # Or flipping (a,b) if it's in the "inner" K22.
                                # Let's flip (u,v)
                                return True, edge_idx[(u, v)] if u < v else edge_idx[(v, u)]
        return False, -1

    best_n = 29
    best_colors = base_colors[:]

    # Local search for n=30
    # Heuristic initialization: 
    # Since Paley(29) is self-complementary, maybe a balanced coloring for the new vertex is good.
    # Let's try a few different initial patterns for the 29 edges incident to vertex 29.
    
    patterns = [
        [0]*29,
        [1]*29,
        [i % 2 for i in range(29)],
        [random.randint(0,1) for _ in range(29)]
    ]
    
    # Add more random patterns
    for _ in range(20):
        patterns.append([random.randint(0,1) for _ in range(29)])
    
    # Shuffle patterns to vary order
    random.shuffle(patterns)

    for pattern in patterns:
        if time.time() > deadline:
            break
            
        colors_list = base_colors[:] + pattern
        
        adj, inter = init_state(colors_list)
        
        # Local search
        max_iters = 50000 # Increased because check is faster
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            found, e_idx = has_violation(adj, inter)
            
            if not found:
                best_n = 30
                best_colors = colors_list[:]
                break
                
            # Flip the edge
            old_c = colors_list[e_idx]
            new_c = 1 - old_c
            colors_list[e_idx] = new_c
            update_edge(e_idx, new_c, adj, inter)

    if best_n == 30:
        return {'n': 30, 'colors': best_colors}
    else:
        return {'n': 29, 'colors': base_colors}
# EVOLVE-BLOCK-END
