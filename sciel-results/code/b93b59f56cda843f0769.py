"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with the known 32-vertex coloring (Greenwood-Gleason doubled)
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
    
    # Build the 32-vertex coloring
    n_start = 32
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    # Helper to check if a coloring is valid (no monochromatic triangles)
    def is_valid(n, colors):
        # Build adjacency matrices for each color
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Check for monochromatic triangles
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                if c == 0:
                    continue
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        return False
        return True
    
    # Helper to count monochromatic triangles
    def count_mono_triangles(n, colors):
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                if c == 0:
                    continue
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count
    
    # Verify the starting coloring is valid
    if not is_valid(n_start, colors):
        # This shouldn't happen, but just in case
        return {'n': n_start, 'colors': colors}
    
    best_n = n_start
    best_colors = colors[:]
    
    # Try to extend the coloring by adding vertices one at a time
    # For each new vertex, we need to assign colors to edges from it to all existing vertices
    # such that no monochromatic triangle is created
    
    def can_add_vertex(n, colors, new_vertex_colors):
        """Check if we can add a new vertex with the given edge colors to existing vertices."""
        # new_vertex_colors[i] is the color of edge between new vertex and vertex i
        # Check for monochromatic triangles involving the new vertex
        for i in range(n):
            c1 = new_vertex_colors[i]
            if c1 == 0:
                continue
            for j in range(i + 1, n):
                c2 = new_vertex_colors[j]
                if c2 == 0:
                    continue
                if c1 == c2:
                    # Check if edge (i,j) has the same color
                    idx = 0
                    for a in range(n):
                        for b in range(a + 1, n):
                            if a == i and b == j:
                                if colors[idx] == c1:
                                    return False
                            idx += 1
                    break
        return True
    
    def add_vertex(n, colors):
        """Try to add a new vertex to the coloring. Returns (new_n, new_colors) or None."""
        # For each possible assignment of colors to edges from new vertex to existing vertices
        # We need to avoid monochromatic triangles
        
        # Use backtracking to find a valid assignment
        new_vertex_colors = [0] * n
        n_edges = n * (n - 1) // 2
        
        # Precompute edge colors for quick lookup
        edge_color = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_color[(i, j)] = colors[idx]
                idx += 1
        
        def backtrack(pos):
            if pos == n:
                return True
            for c in [1, 2, 3]:
                new_vertex_colors[pos] = c
                # Check if this creates any monochromatic triangle
                valid = True
                for j in range(pos):
                    if new_vertex_colors[j] == c:
                        if edge_color.get((j, pos), 0) == c or edge_color.get((pos, j), 0) == c:
                            # Need to check edge (j, pos)
                            if j < pos:
                                if edge_color[(j, pos)] == c:
                                    valid = False
                                    break
                            else:
                                if edge_color[(pos, j)] == c:
                                    valid = False
                                    break
                if valid:
                    if backtrack(pos + 1):
                        return True
            new_vertex_colors[pos] = 0
            return False
        
        if backtrack(0):
            # Build new colors list
            new_colors = colors[:]
            for i in range(n):
                new_colors.append(new_vertex_colors[i])
            return n + 1, new_colors
        return None
    
    # Try to extend the coloring
    current_n = n_start
    current_colors = colors[:]
    
    while time.time() < deadline - 5:
        result = add_vertex(current_n, current_colors)
        if result is None:
            break
        current_n, current_colors = result
        best_n = current_n
        best_colors = current_colors[:]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
