"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns a 4-colouring of K_n with no monochromatic triangle, maximising n.
    Base: Greenwood-Gleason K16 (Clebsch graph x3) doubled to K32.
    Then attempts to extend to larger n using efficient CSP solving for the new vertex.
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # --- Build base K16 Greenwood-Gleason colouring ---
    # GF(16) with polynomial x^4 + x + 1 (0b10011)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    # Base K16: colour of edge (i,j) = log[i^j] % 3 for i!=j, i,j in 0..15
    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log[i ^ j] % 3

    # --- Build K32: two copies of K16, cross edges all colour 3 ---
    def get_color_32(i, j):
        if i // 16 == j // 16:
            return base16[(i % 16, j % 16)]
        else:
            return 3

    # Precompute adjacency for K32
    n32 = 32
    adj = [[0] * n32 for _ in range(n32)]
    for i in range(n32):
        for j in range(i + 1, n32):
            c = get_color_32(i, j)
            adj[i][j] = c
            adj[j][i] = c

    # --- Efficiently check if a new vertex can be added ---
    def can_add_vertex(adj, n, color_map):
        """
        Check if a new vertex (n) with the given color assignment to existing vertices
        creates any monochromatic triangle.
        """
        groups = {0: [], 1: [], 2: [], 3: []}
        for i in range(n):
            groups[color_map[i]].append(i)

        for c in range(4):
            grp = groups[c]
            for a_idx in range(len(grp)):
                for b_idx in range(a_idx + 1, len(grp)):
                    a, b = grp[a_idx], grp[b_idx]
                    if adj[a][b] == c:
                        return False
        return True

    def try_add_vertex(adj, n, deadline):
        """
        Try to find a valid color assignment for a new vertex using backtracking
        with constraint propagation.
        Returns the color_map if found, None otherwise.
        """
        # For each color c, we need the set of vertices assigned to c to be independent
        # in the subgraph of color c.
        
        # Precompute: for each vertex i, which other vertices j have color c edge between them?
        # Actually, let's use a different approach:
        # For each vertex i, the color assigned to edge (new_vertex, i) must be such that
        # no two vertices i, j with the same color c have a color-c edge between them.
        
        # This is equivalent to: for each color c, the set {i : color_map[i] = c} must be
        # an independent set in G_c (the subgraph of edges of color c).
        
        # We can model this as a graph coloring problem where:
        # - We have 4 colors
        # - For each color c, we cannot assign color c to both i and j if edge (i,j) has color c
        # - This is a 4-coloring problem on the "conflict graph" where (i,j) is an edge
        #   if they conflict for some color
        # - But actually, it's simpler: for each pair (i,j), they cannot share a color c
        #   if adj[i][j] == c.
        
        # Let's use backtracking with MRV (minimum remaining values) heuristic.
        
        # For each vertex i, the forbidden colors are those c where there exists j < i
        # with color_map[j] = c and adj[i][j] = c.
        
        # Actually, let's just use a simple backtracking with randomization and pruning.
        
        n_vars = n
        assignment = [-1] * n_vars
        
        # Precompute for each vertex i, and each color c, the list of vertices j
        # such that adj[i][j] == c. This helps with quick pruning.
        # Actually, we can just check on the fly.
        
        def backtrack(idx, current_assignment):
            if time.time() > deadline:
                return False
            
            if idx == n_vars:
                return True
            
            # Try each color for vertex idx
            # Use a heuristic: try colors that are less constrained
            colors_to_try = [0, 1, 2, 3]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check if assigning color c to vertex idx is consistent
                valid = True
                for j in range(idx):
                    if current_assignment[j] == c and adj[idx][j] == c:
                        valid = False
                        break
                
                if valid:
                    current_assignment[idx] = c
                    if backtrack(idx + 1, current_assignment):
                        return True
                    current_assignment[idx] = -1
            
            return False
        
        # Run multiple attempts with different random seeds
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Randomly shuffle the order of colors tried
            if backtrack(0, assignment[:]):
                return assignment[:]
        
        return None

    # --- Attempt to extend K32 to larger n ---
    current_adj = [row[:] for row in adj]
    current_n = 32

    # Try to extend to n = 33, 34, ..., up to 50 or until time runs out
    for target_n in range(33, 51):
        if time.time() > deadline - 10.0:  # Leave some time for final processing
            break

        # Try to add one vertex
        color_map = try_add_vertex(current_adj, current_n, deadline - 5.0)
        
        if color_map is not None:
            # Add the new vertex
            new_n = current_n + 1
            new_row = color_map[:] + [0]
            for i in range(current_n):
                current_adj[i].append(color_map[i])
            current_adj.append(new_row)
            current_n = new_n
        else:
            # Cannot extend further
            break

    # --- Build final result ---
    n_final = current_n
    colors = []
    for i in range(n_final):
        for j in range(i + 1, n_final):
            colors.append(current_adj[i][j])
    return {'n': n_final, 'colors': colors}

# EVOLVE-BLOCK-END
