"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_valid_coloring(n, colors):
    """Check if coloring has no monochromatic C4."""
    # Build adjacency per color
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    
    # For each color, check no two vertices share two common neighbors
    for c in range(5):
        # Build neighbor lists
        neigh = [set() for _ in range(n)]
        for (i, j) in adj[c]:
            neigh[i].add(j)
            neigh[j].add(i)
        # Check for C4: two vertices with >=2 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                if len(neigh[i] & neigh[j]) >= 2:
                    return False
    return True

def _lazy_woldar_25():
    """Construct a valid 5-coloring of K_25 using Z_5 x Z_5."""
    n = 25
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = i // 5, i % 5
            xj, yj = j // 5, j % 5
            if xi == xj:
                c = 4
            else:
                dx = (xj - xi) % 5
                dy = (yj - yi) % 5
                # dx^{-1} mod 5
                inv = pow(dx, -1, 5)
                c = (dy * inv) % 5
            colors.append(c)
    return n, colors

def construct():
    deadline = time.time() + 100  # 100s safety margin
    
    # Start with the known 25-vertex construction
    n, colors = _lazy_woldar_25()
    best_n, best_colors = n, colors[:]
    
    # Verify base construction
    if not _is_valid_coloring(n, colors):
        # Fallback to smaller known-good
        n_small = 11
        return {'n': n_small, 'colors': [min(j - i, n_small - j + i) - 1 for i in range(n_small) for j in range(i + 1, n_small)]}
    
    # Try to extend to 26 vertices
    # Vertex 25 is the new vertex. We need to assign colors to edges (25, i) for i in 0..24.
    # Each edge (25, i) gets a color c_i in {0,1,2,3,4}.
    # Constraint: for any color c, the set of vertices connected to 25 by color c,
    # combined with existing color-c edges, must not create a C4.
    
    # A C4 involving vertex 25 would be: 25 - a - b - c - 25 where all edges are same color.
    # That means: color(25,a)=color(a,b)=color(b,c)=color(c,25)=c0.
    # So if color(25,a)=c0 and color(25,c)=c0, then there must be no path a-b-c of length 2
    # in color c0 (i.e., a and c must not share a common neighbor b in color c0).
    
    # For each color c0, let S_c0 = {i : color(25,i) = c0}.
    # Then for any a, c in S_c0, a and c must not have a common neighbor in color c0.
    
    # Let's precompute for each color c0 and each pair (a,c), whether they share a common neighbor in color c0.
    # common_neighbor_count[c0][a][c] = number of common neighbors of a and c in color c0.
    
    # Build adjacency per color for the 25-vertex graph
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(25):
        for j in range(i+1, 25):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    
    neigh = [set() for _ in range(25)]
    for c in range(5):
        for (i, j) in adj[c]:
            neigh[i].add(j)
            neigh[j].add(i)
    
    # common[c0][a][c] = |neigh_c0[a] ∩ neigh_c0[c]|
    # We need: for each c0, for all a != c in S_c0, common[c0][a][c] == 0.
    
    # Precompute incompatibility: for each color c0, incompatible[c0] is a set of pairs (a,c)
    # where a and c share a common neighbor in color c0.
    
    incompatible = [set() for _ in range(5)]
    for c0 in range(5):
        for a in range(25):
            for c in range(a+1, 25):
                if len(neigh[a] & neigh[c]) >= 1:  # They share at least one common neighbor in color c0
                    incompatible[c0].add((a, c))
    
    # Now we need to assign colors to edges (25, i) for i in 0..24.
    # This is a constraint satisfaction problem.
    # Let's use backtracking with heuristics.
    
    def try_assign(n_vertices, current_colors, edge_colors):
        """Try to assign colors to remaining edges (n_vertices, i) for i in 0..n_vertices-1."""
        if time.time() > deadline:
            return None
        
        if n_vertices == 25:
            return edge_colors[:]
        
        # Choose the next vertex to assign
        # Use MRV: pick the vertex with fewest available colors
        best_vertex = -1
        best_options = None
        min_count = 6
        
        for v in range(25):
            if v in edge_colors:
                continue
            # Check which colors are still valid for edge (25, v)
            valid = []
            for c0 in range(5):
                # If we assign color c0 to (25, v), then for all u already assigned color c0 to (25, u),
                # v and u must not be in incompatible[c0].
                ok = True
                for u, cu in edge_colors.items():
                    if cu == c0:
                        if (min(v,u), max(v,u)) in incompatible[c0]:
                            ok = False
                            break
                if ok:
                    valid.append(c0)
            if len(valid) < min_count:
                min_count = len(valid)
                best_vertex = v
                best_options = valid
                if min_count == 0:
                    break
        
        if best_vertex == -1 or not best_options:
            return None
        
        for c0 in best_options:
            edge_colors[best_vertex] = c0
            result = try_assign(n_vertices, current_colors, edge_colors)
            if result is not None:
                return result
            del edge_colors[best_vertex]
        
        return None
    
    edge_colors = {}
    result = try_assign(25, colors, edge_colors)
    
    if result is not None:
        # Build the full coloring for 26 vertices
        new_colors = colors[:]
        for i in range(25):
            new_colors.append(edge_colors[i])
        # Verify
        if _is_valid_coloring(26, new_colors):
            return {'n': 26, 'colors': new_colors}
    
    # If extension failed, return the 25-vertex solution
    return {'n': 25, 'colors': best_colors}
# EVOLVE-BLOCK-END
