"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140.0  # safety margin
    
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
    
    # Base: n=32 construction
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    best_n = 32
    best_colors = base_colors[:]
    
    # Try to extend by adding vertices
    # We'll try to add vertices one at a time, checking for monochromatic triangles
    
    def has_mono_triangle(colors, n):
        """Check if any color class contains a triangle."""
        # Build adjacency for each color
        adj = [[False] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i][j] = True
                adj[c][j][i] = True
                idx += 1
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[c][i][j]:
                        for k in range(j + 1, n):
                            if adj[c][i][k] and adj[c][j][k]:
                                return True
        return False
    
    def try_add_vertex(colors, n):
        """Try to add a new vertex n with some color assignment to all edges to existing vertices."""
        # For the new vertex, we need to assign colors to edges (n, i) for i in 0..n-1
        # such that no monochromatic triangle is formed
        # A triangle (n, i, j) is monochromatic color c if:
        #   colors[i,j] == c and colors[n,i] == c and colors[n,j] == c
        
        # For each pair (i, j), if colors[i,j] = c, then we cannot have both colors[n,i]=c and colors[n,j]=c
        # This is a constraint satisfaction problem
        
        # Try random assignments
        for _ in range(1000):
            if time.time() > deadline:
                return None
            new_colors = [random.randint(0, 3) for _ in range(n)]
            # Check constraints
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[i * (i + 1) // 2 + j - i - 1]  # index of edge (i,j)
                    if new_colors[i] == c and new_colors[j] == c:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return new_colors
        return None
    
    # Try to extend from n=32
    current_n = 32
    current_colors = base_colors[:]
    
    while current_n < 62:
        if time.time() > deadline:
            break
        new_vertex_colors = try_add_vertex(current_colors, current_n)
        if new_vertex_colors is None:
            break
        # Add the new vertex
        for i in range(current_n):
            current_colors.append(new_vertex_colors[i])
        current_n += 1
        if current_n > best_n:
            best_n = current_n
            best_colors = current_colors[:]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
