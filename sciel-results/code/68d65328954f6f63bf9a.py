"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Build the 32-vertex base coloring (Greenwood-Gleason: two copies of 16-vertex Clebsch)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def base_colors(n=32):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def color_index(n, i, j):
        """Get index in colors array for edge (i,j), i<j in lexicographic order."""
        if i > j:
            i, j = j, i
        return i * (2*n - i - 1) // 2 + (j - i - 1)

    # Start with 32-vertex base
    n = 32
    colors = base_colors(n)
    best_n = n
    best_colors = colors[:]

    # Precompute adjacency lists for each color class
    def build_color_adj(colors, n):
        """Build adjacency lists for each color. adj[c][v] = set of vertices connected to v by color c."""
        adj = [set() for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[color_index(n, i, j)]
                adj[c].add(j) if False else None  # placeholder
        # Rebuild properly
        adj = [set() for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[color_index(n, i, j)]
                adj[c].add(j) if False else None
        # Actually, let's store as list of sets per vertex
        adj = [[set() for _ in range(4)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[color_index(n, i, j)]
                adj[i][c].add(j)
                adj[j][c].add(i)
        return adj

    def try_add_vertex(colors, n, adj, deadline):
        """
        Try to add one new vertex (vertex n) to the coloring.
        For the new vertex, we assign colors to edges (n, i) for i in 0..n-1.
        Constraint: for each color c, the set S_c = {i : color(n,i) = c} must be
        an independent set in the color-c subgraph (no two vertices in S_c connected by color c).
        
        We use backtracking with constraint propagation.
        Returns (success, new_colors, new_n, new_adj) or (False, None, None, None).
        """
        if time.time() > deadline:
            return False, None, None, None

        # For each existing vertex i, determine which colors are still available
        # A color c is available for edge (new, i) if:
        #   - For all j < i already assigned color c to (new, j), i is NOT in adj[j][c]
        #   - For all j > i not yet assigned, we'll check later
        
        # Actually, the constraint is: if color(new, i) = c and color(new, j) = c,
        # then i and j must NOT be connected by color c in the existing graph.
        
        # This is equivalent to: S_c must be an independent set in G_c (the color-c graph).
        
        # We'll use backtracking: assign colors to vertices 0, 1, ..., n-1 in order.
        # For vertex i, available colors are those c such that:
        #   for all j < i with assignment[j] = c, i is not in adj[j][c]
        
        assignment = [-1] * n
        new_colors = colors[:]
        
        # Precompute for each vertex i and color c: which previously assigned vertices
        # with color c would conflict (i.e., are in adj[i][c])
        
        def backtrack(i):
            nonlocal assignment, new_colors
            if time.time() > deadline:
                return False
            if i == n:
                # All assigned, success
                return True
            
            # Try each color for vertex i
            for c in [0, 1, 2, 3]:
                # Check if c is compatible with all previously assigned vertices
                compatible = True
                for j in range(i):
                    if assignment[j] == c:
                        # Check if i and j are connected by color c
                        if j in adj[i][c]:
                            compatible = False
                            break
                if not compatible:
                    continue
                
                assignment[i] = c
                if backtrack(i + 1):
                    return True
                assignment[i] = -1
            
            return False

        if backtrack(0):
            # Success! Add the new vertex
            new_n = n + 1
            # Append colors for edges (n, i) for i in 0..n-1
            for i in range(n):
                new_colors.append(assignment[i])
            
            # Update adjacency
            new_adj = [list(s) for s in adj]  # copy
            # Actually, let's rebuild
            new_adj = [[set() for _ in range(4)] for _ in range(new_n)]
            for i in range(new_n):
                for j in range(i + 1, new_n):
                    if i < n and j < n:
                        c = colors[color_index(n, i, j)]
                    elif i == n:
                        c = assignment[j]
                    else:
                        c = assignment[i]
                    new_adj[i][c].add(j)
                    new_adj[j][c].add(i)
            
            return True, new_colors, new_n, new_adj
        
        return False, None, None, None

    # Main loop: try to extend the coloring
    adj = build_color_adj(colors, n)
    
    for attempt in range(30):  # Try to add up to 30 more vertices
        if time.time() > deadline:
            break
        
        success, new_colors, new_n, new_adj = try_add_vertex(colors, n, adj, deadline)
        
        if success:
            colors = new_colors
            n = new_n
            adj = new_adj
            best_n = max(best_n, n)
            best_colors = colors[:]
        else:
            # Try a different ordering or randomized approach
            # Shuffle the vertex order for the next attempt
            break

    # If we got stuck, try a randomized restart with different base
    # But first, check if we can do better with a different strategy
    
    # Alternative: try to use the structure more cleverly
    # The 32-vertex coloring has two 16-vertex blocks. Maybe we can find a third block?
    
    # For now, return the best we found
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
