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

    # Helper: get edge index for (i,j) in a graph of size n, lexicographic order
    def edge_idx(i, j, n):
        # i < j
        # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # In row i, position of j: j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # --- Extension phase ---
    def try_extend(edge_colors, n_start, max_n, deadline):
        # Try to add vertices one by one using DFS backtracking
        n = n_start
        colors_list = list(edge_colors)
        # We'll try to add vertices from n_start to max_n
        while n < max_n:
            # Try all 4 colors for each of the n existing vertices
            best_assign = None
            found = False
            
            # Use backtracking
            def backtrack(vertex_idx, assignment):
                nonlocal found
                if time.time() > deadline:
                    return False
                if vertex_idx == n:
                    found = True
                    return True
                
                for c in range(4):
                    # Check if assigning color c to (new_vertex, vertex_idx) creates a monochromatic triangle
                    valid = True
                    for k in range(vertex_idx):
                        if assignment[k] == c:
                            # Check if edge (k, vertex_idx) has color c
                            ei = edge_idx(k, vertex_idx, n)
                            if colors_list[ei] == c:
                                valid = False
                                break
                    if valid:
                        assignment[vertex_idx] = c
                        if backtrack(vertex_idx + 1, assignment):
                            return True
                return False
            
            assignment = [0] * n
            if backtrack(0, assignment):
                # Successfully found a coloring for the new vertex
                # Append to colors_list
                for c in assignment:
                    colors_list.append(c)
                n += 1
            else:
                break  # Cannot extend further
            
            if time.time() > deadline:
                break
        
        return colors_list, n
    
    # Try extension
    best_colors = list(edge_colors)
    best_n = n_base
    
    # Multiple attempts with different random orderings of vertices
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Try extending the base coloring
        colors_list, n_final = try_extend(edge_colors, n_base, min(n_base + 30, 62), deadline)
        
        if n_final > best_n:
            best_n = n_final
            best_colors = colors_list
        
        if best_n >= 51:
            break
        
        # If we can't extend, try perturbing the base coloring slightly
        # This helps escape local minima
        if attempt % 10 == 5:
            # Try a random perturbation of the bipartite edges
            perturbed = list(edge_colors)
            # Swap some 3s to other colors in the bipartite region
            for _ in range(2):
                # Find bipartite edges
                bipartite_edges = []
                idx2 = 0
                for i in range(n_base):
                    for j in range(i + 1, n_base):
                        if i // 16 != j // 16:
                            bipartite_edges.append(idx2)
                        idx2 += 1
                if bipartite_edges:
                    pos = bipartite_edges[attempt % len(bipartite_edges)]
                    perturbed[pos] = (perturbed[pos] + 1) % 4
            
            # Check validity of perturbed
            valid = True
            for c in range(4):
                # Check no triangle in color class c
                for i in range(n_base - 2):
                    if not valid:
                        break
                    for j in range(i + 1, n_base - 1):
                        if not valid:
                            break
                        for k in range(j + 1, n_base):
                            # Get colors of (i,j), (j,k), (i,k)
                            idx_ij = edge_idx(i, j, n_base)
                            idx_jk = edge_idx(j, k, n_base)
                            idx_ik = edge_idx(i, k, n_base)
                            if perturbed[idx_ij] == perturbed[idx_jk] == perturbed[idx_ik] == c:
                                valid = False
                                break
            
            if valid:
                colors_list, n_final = try_extend(perturbed, n_base, min(n_base + 30, 62), deadline)
                if n_final > best_n:
                    best_n = n_final
                    best_colors = colors_list
                if best_n >= 51:
                    break
    
    # Ensure validity of final answer
    # If invalid, fall back to base
    def validate(colors_list, n_val):
        for c in range(4):
            for i in range(n_val - 2):
                for j in range(i + 1, n_val - 1):
                    for k in range(j + 1, n_val):
                        idx_ij = edge_idx(i, j, n_val)
                        idx_jk = edge_idx(j, k, n_val)
                        idx_ik = edge_idx(i, k, n_val)
                        if colors_list[idx_ij] == colors_list[idx_jk] == colors_list[idx_ik] == c:
                            return False
        return True
    
    if not validate(best_colors, best_n):
        best_colors = list(edge_colors)
        best_n = n_base
    
    return {'n': best_n, 'colors': best_colors}
