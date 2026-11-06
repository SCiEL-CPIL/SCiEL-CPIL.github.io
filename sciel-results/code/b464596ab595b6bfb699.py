import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Try to construct a valid coloring for n=29 using Paley's method
    # Paley graph on F_29
    p = 29
    n = p
    squares = {x * x % p for x in range(1, p)}
    
    # colors[i][j] for i < j
    # We store colors as a flat list in lexicographic order
    # Or just a 2D matrix for ease of checking
    colors_mat = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                colors_mat[i][j] = 0
            else:
                colors_mat[i][j] = 1
            colors_mat[j][i] = colors_mat[i][j]

    # Verify if this coloring is valid (no mono K2,2,2)
    def is_valid(mat, n):
        # Check all 6-subsets
        # A 6-subset contains a mono K2,2,2 in color 0 if:
        # The induced subgraph on these 6 vertices has 12 edges of color 0 
        # and the 3 missing edges (color 1) form a perfect matching.
        # Similarly for color 1.
        
        for verts in itertools.combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = verts
            # Count color 1 edges
            c1_edges = []
            for a in range(6):
                for b in range(a + 1, 6):
                    if mat[verts[a]][verts[b]] == 1:
                        c1_edges.append((a, b))
            
            if len(c1_edges) == 3:
                # Check if they form a perfect matching
                # Perfect matching on 6 vertices: 3 disjoint edges covering all 6
                # Sort edges
                edges = sorted(c1_edges)
                # Possible perfect matchings on 6 nodes (labeled 0..5)
                # We can just check if the set of edges is one of the 15 perfect matchings
                # Or simpler: check if each vertex is covered exactly once
                covered = [0] * 6
                valid_match = True
                for a, b in edges:
                    covered[a] += 1
                    covered[b] += 1
                if not all(c == 1 for c in covered):
                    valid_match = False
                if valid_match:
                    return False
            
            # Check for mono K2,2,2 in color 1
            # This happens if there are 12 edges of color 1, i.e., 3 edges of color 0
            # which form a perfect matching.
            # Since total edges is 15, if len(c1_edges) == 12, then len(c0_edges) == 3.
            if len(c1_edges) == 12:
                c0_edges = []
                for a in range(6):
                    for b in range(a + 1, 6):
                        if mat[verts[a]][verts[b]] == 0:
                            c0_edges.append((a, b))
                
                # c0_edges must be a perfect matching
                covered = [0] * 6
                valid_match = True
                for a, b in c0_edges:
                    covered[a] += 1
                    covered[b] += 1
                if not all(c == 1 for c in covered):
                    valid_match = False
                if valid_match:
                    return False
        
        return True

    if is_valid(colors_mat, n):
        # Flatten to lexicographic order
        flat_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                flat_colors.append(colors_mat[i][j])
        return {'n': n, 'colors': flat_colors}
    else:
        # Fallback to known valid n=17
        p = 17
        n = p
        squares = {x * x % p for x in range(1, p)}
        flat_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in squares:
                    flat_colors.append(0)
                else:
                    flat_colors.append(1)
        return {'n': n, 'colors': flat_colors}
