"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

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

    # Build K16 coloring: Clebsch graph in 3 colors
    def build_k16_colors():
        colors = []
        for i in range(16):
            for j in range(i + 1, 16):
                colors.append(log[i ^ j] % 3)
        return colors

    # Build K32 coloring: two copies of K16 with color 3 between them
    def build_k32_colors():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def check_no_triangle(colors, n):
        """Verify no color class contains a triangle."""
        # Build adjacency for each color
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        
        # Check each color for triangles
        for c in range(4):
            edges = adj[c]
            # For each pair of edges sharing a vertex, check if they form a triangle
            # More efficient: for each vertex, check if any two neighbors are connected
            vertex_adj = [set() for _ in range(n)]
            for (u, v) in edges:
                vertex_adj[u].add(v)
                vertex_adj[v].add(u)
            
            for u in range(n):
                neighbors = vertex_adj[u]
                for v in neighbors:
                    if u < v:
                        # Check if any common neighbor w of u and v (w != u, v)
                        common = neighbors & vertex_adj[v]
                        if common:
                            return False
        return True

    def is_valid_incremental(base_colors, base_n, new_vertex_colorings):
        """Check if adding a new vertex with given colorings to all existing vertices is valid."""
        n = base_n + 1
        # Reconstruct full color list
        full_colors = base_colors[:]
        full_colors.extend(new_vertex_colorings)
        return check_no_triangle(full_colors, n)

    # Start with K32
    best_colors = build_k32_colors()
    best_n = 32

    # Try to extend by adding vertices one at a time
    # For each new vertex, we need to assign colors to edges from it to all existing vertices
    # such that no monochromatic triangle is formed
    
    # Precompute for efficiency: for each color c, and each pair (i,j), whether edge (i,j) has color c
    # When adding vertex n, we assign colors to edges (i, n) for i in 0..n-1
    # A triangle (i, j, n) with colors c_ij, c_in, c_jn is monochromatic iff all three are the same color
    
    # Strategy: try all possible colorings of the new vertex's edges
    # This is 4^(n-1) possibilities, which is too many for large n
    # Instead, use constraint propagation / backtracking
    
    def try_extend(current_colors, current_n, max_time_remaining):
        """Try to add one vertex to the current coloring."""
        if time.time() > max_time_remaining:
            return None
        
        # For each existing vertex i, we need to choose a color for edge (i, n)
        # Constraint: for any pair (i, j) with i < j < n, if edge (i,j) has color c,
        # then not both edge (i,n) and edge (j,n) can have color c
        
        # Let's use backtracking with constraint propagation
        # colors_to_assign[i] = color for edge (i, new_vertex)
        
        # First, gather constraints: for each color c, the set of vertices that form a triangle
        # with the new vertex if two of them are connected by color c
        
        # More precisely: for each color c, let S_c be the set of vertices i such that
        # there exists j > i with edge (i,j) having color c. If we assign color c to both
        # edge (i, new) and edge (j, new), we get a monochromatic triangle.
        
        # Actually, the constraint is: for each color c, the set of vertices assigned color c
        # to their edge with the new vertex must be an independent set in the graph formed
        # by color-c edges in the current coloring.
        
        # Build for each color c, the adjacency structure
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(current_n):
            for j in range(i + 1, current_n):
                c = current_colors[idx]
                adj[c].add((i, j))
                idx += 1
        
        # Convert to per-vertex adjacency
        vertex_adj = [set() for _ in range(4)]
        for c in range(4):
            for (u, v) in adj[c]:
                vertex_adj[c][u].add(v)
                vertex_adj[c][v].add(u)
        
        # Backtracking search
        assignment = [-1] * current_n
        order = list(range(current_n))
        
        def backtrack(pos):
            if time.time() > max_time_remaining:
                return None
            if pos == current_n:
                return assignment[:]
            
            v = order[pos]
            for c in range(4):
                # Check if assigning color c to edge (v, new) is consistent
                # i.e., no already-assigned vertex u with color c has edge (u,v) in color c
                valid = True
                for u in range(pos):
                    if assignment[u] == c:
                        if v in vertex_adj[c][u]:
                            valid = False
                            break
                if valid:
                    assignment[pos] = c
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
            assignment[pos] = -1
            return None
        
        result = backtrack(0)
        if result is not None:
            new_colors = current_colors + result
            return new_colors
        return None

    # Try to extend from K32
    for attempt in range(20):  # Try up to 20 extensions
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        new_colors = try_extend(best_colors, best_n, deadline)
        if new_colors is None:
            break
        best_colors = new_colors
        best_n += 1

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
