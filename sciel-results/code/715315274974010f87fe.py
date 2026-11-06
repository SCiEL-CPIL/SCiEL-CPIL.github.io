"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

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

    # Base: Greenwood-Gleason K16 doubled to K32
    def base_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def edge_index(n, i, j):
        """Return the index in the colors list for edge (i,j) in K_n."""
        # edges are in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        idx = 0
        for k in range(i):
            idx += (n - 1 - k)
        idx += (j - i - 1)
        return idx

    def has_mono_triangle(colors, n, v):
        """Check if vertex v is part of any monochromatic triangle in the coloring."""
        # For each pair (a,b) with a < b < v, check if edges (v,a), (v,b), (a,b) all same color
        for a in range(v):
            ca = colors[edge_index(n, a, v)]
            for b in range(a + 1, v):
                cb = colors[edge_index(n, b, v)]
                if ca == cb:
                    cab = colors[edge_index(n, a, b)]
                    if cab == ca:
                        return True
        return False

    def try_extend(colors, n):
        """Try to add vertex n with some color assignment to all edges to existing vertices."""
        # We need to assign colors to edges (0,n), (1,n), ..., (n-1,n)
        # such that no monochromatic triangle is formed.
        # Greedy: for each new edge, pick a color that doesn't create a mono triangle.
        # This is a constraint satisfaction problem on n edges.
        # Use backtracking with forward checking.
        
        def backtrack(pos, assignment, used_colors_per_vertex):
            if pos == n:
                return assignment
            
            # Try each color for edge (pos, n)
            for c in range(4):
                # Check: does assigning color c to edge (pos, n) create a mono triangle?
                # Triangle (a, pos, n) is mono if color(a,pos) == color(a,n) == color(pos,n) = c
                # So for each a < pos: if color(a,pos) == c AND color(a,n) == c, conflict
                ok = True
                for a in range(pos):
                    if assignment[a] == c:
                        # color(a,n) = assignment[a] = c, and color(a,pos) must be checked
                        # Wait: assignment[a] is color(a, n). We need color(a, pos).
                        # color(a, pos) is already in the base coloring (edges among 0..n-1)
                        # But we're building colors for K_{n+1}, so edges among 0..n-1 are fixed.
                        # Actually, the base colors for edges among 0..n-1 are already set.
                        # Let me restructure: we have base_colors for K_n, and we're adding vertex n.
                        # Edges (i, n) for i in 0..n-1 need colors.
                        # Triangle (a, b, n) is mono if color(a,b) == color(a,n) == color(b,n).
                        # So for the new edge (pos, n) with color c:
                        # For each a < pos: if color_base(a, pos) == c AND color(a, n) == c, conflict.
                        # color_base(a, pos) is the original color in the K_n coloring.
                        # color(a, n) = assignment[a].
                        if assignment[a] == c:
                            # Need to check if color(a, pos) in base == c
                            # But we don't have easy access... let me precompute.
                            pass
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # More careful check
                conflict = False
                for a in range(pos):
                    # color(a, n) = assignment[a]
                    # color(a, pos) = base color
                    # color(pos, n) = c
                    # Triangle mono if all three equal
                    if assignment[a] == c:
                        # Check base color of (a, pos)
                        # We need access to base colors
                        # Let me pass base_colors as a closure
                        if base_colors[edge_index(n, a, pos)] == c:
                            conflict = True
                            break
                
                if conflict:
                    continue
                
                assignment[pos] = c
                if backtrack(pos + 1, assignment, used_colors_per_vertex):
                    return assignment
            
            return None
        
        # Precompute base colors for K_n
        base_colors = colors[:]  # colors for K_n
        
        assignment = [0] * n
        result = backtrack(0, assignment, None)
        if result is not None:
            return result
        return None

    # Start with K32
    n = 32
    colors = base_coloring(n)
    best_n = n
    best_colors = colors[:]

    # Try to extend by adding vertices one at a time
    while n < 62:
        if time.time() > deadline - 5:
            break
        
        ext = try_extend(colors, n)
        if ext is not None:
            # Append the new edge colors
            colors.extend(ext)
            n += 1
            best_n = n
            best_colors = colors[:]
        else:
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
