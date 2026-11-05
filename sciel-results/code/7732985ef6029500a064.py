import time
import random

def construct():
    deadline = time.time() + 140  # safety margin for the 150s kill

    # --- Build a base n=32 colouring (Greenwood-Gleason, Clebsch graph ×3 + bipartite) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def edge_color_core(i, j):
        # i, j in 0..15
        return log_map[i ^ j] % 3

    # Build n=32 edge colors
    n_base = 32
    edge_colors = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                c = edge_color_core(i % 16, j % 16)
            else:
                c = 3
            edge_colors[idx] = c
            idx += 1

    # Precompute edge index lookup for fast access
    # edge_idx[i][j] = index in colors list for edge (i,j), i<j
    def make_edge_idx(n):
        ei = [[0]*n for _ in range(n)]
        idx2 = 0
        for i in range(n):
            for j in range(i+1, n):
                ei[i][j] = idx2
                ei[j][i] = idx2
                idx2 += 1
        return ei

    # --- Extension phase with improved backtracking ---
    def try_extend(edge_colors, n_start, max_n, deadline):
        n = n_start
        colors_list = list(edge_colors)
        
        while n < max_n and time.time() < deadline:
            # Build edge index for current n
            ei = make_edge_idx(n)
            
            # For the new vertex (index n), we need to assign colors to edges (k, n) for k in 0..n-1
            # Constraint: for any pair (a, b) where both edges to new vertex have color c,
            # the edge (a, b) must NOT have color c
            
            # Backtracking with ordering: try vertices that are most constrained first
            # A vertex k is more constrained if it has many neighbors with same color assignments
            # Simple heuristic: order by degree in the "conflict" sense
            
            # Try all 4^ n combinations with backtracking
            # But n can be up to 62, so we need smart pruning
            
            # Better approach: for each color c, the set of vertices connected to new vertex
            # with color c must form an independent set in the subgraph of color c edges
            
            # Let's use a different strategy: build the assignment greedily with backtracking
            # Order vertices by how many constraints they impose
            
            # Compute for each vertex k, how many other vertices l have edge (k,l) with each color
            # This gives us a measure of constraint
            
            # Actually, let's just do standard backtracking but with good variable ordering
            # and value ordering
            
            # Variable ordering: sort vertices by number of constraints (descending)
            # A vertex k is constrained by all previously assigned vertices that share a color
            # with it in the new vertex's incident edges
            
            # Simple: just go in order 0, 1, ..., n-1 but prune early
            
            assignment = [-1] * n
            
            # Precompute for each color c and vertex k, which vertices l < k have edge (l,k) of color c
            # This helps with fast conflict checking
            conflict_sets = {}
            for c in range(4):
                conflict_sets[c] = [[] for _ in range(n)]
                for k in range(n):
                    for l in range(k):
                        if colors_list[ei[l][k]] == c:
                            conflict_sets[c][k].append(l)
            
            found = False
            
            def backtrack(pos):
                nonlocal found
                if time.time() > deadline:
                    return False
                if pos == n:
                    found = True
                    return True
                
                k = pos
                for c in range(4):
                    # Check if assigning color c to edge (k, new_vertex) is valid
                    # For all l < k with assignment[l] == c, edge (l, k) must NOT have color c
                    valid = True
                    for l in conflict_sets[c][k]:
                        if assignment[l] == c:
                            valid = False
                            break
                    if valid:
                        assignment[k] = c
                        if backtrack(pos + 1):
                            return True
                        assignment[k] = -1
                return False
            
            if backtrack(0):
                # Successfully found a coloring for the new vertex
                for k in range(n):
                    colors_list.append(assignment[k])
                n += 1
            else:
                break
        
        return colors_list, n
    
    # --- Try multiple different base colorings ---
    best_colors = list(edge_colors)
    best_n = n_base
    
    # Generate several different base colorings
    base_colorings = []
    
    # Base coloring 1: original (Clebsch + bipartite all color 3)
    base_colorings.append(list(edge_colors))
    
    # Base coloring 2: Clebsch + bipartite with some variation
    for variant in range(10):
        if time.time() > deadline:
            break
        ec = list(edge_colors)
        # Randomly flip some bipartite edges
        bipartite_indices = []
        idx3 = 0
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if i // 16 != j // 16:
                    bipartite_indices.append(idx3)
                idx3 += 1
        random.shuffle(bipartite_indices)
        # Flip a few
        for bi in bipartite_indices[:5]:
            ec[bi] = (ec[bi] + 1 + variant) % 4
        # Validate
        valid = True
        ei32 = make_edge_idx(n_base)
        for c in range(4):
            for i in range(n_base - 2):
                if not valid:
                    break
                for j in range(i + 1, n_base - 1):
                    if not valid:
                        break
                    for k in range(j + 1, n_base):
                        if ec[ei32[i][j]] == ec[ei32[j][k]] == ec[ei32[i][k]] == c:
                            valid = False
                            break
        if valid:
            base_colorings.append(ec)
    
    # Try extending each base coloring
    for base in base_colorings:
        if time.time() > deadline:
            break
        colors_list, n_final = try_extend(base, n_base, min(n_base + 30, 62), deadline)
        if n_final > best_n:
            best_n = n_final
            best_colors = colors_list
        if best_n >= 51:
            break
    
    # Ensure validity of final answer
    def validate(colors_list, n_val):
        ei = make_edge_idx(n_val)
        for c in range(4):
            for i in range(n_val - 2):
                for j in range(i + 1, n_val - 1):
                    for k in range(j + 1, n_val):
                        if colors_list[ei[i][j]] == colors_list[ei[j][k]] == colors_list[ei[i][k]] == c:
                            return False
        return True
    
    if not validate(best_colors, best_n):
        best_colors = list(edge_colors)
        best_n = n_base
    
    return {'n': best_n, 'colors': best_colors}
