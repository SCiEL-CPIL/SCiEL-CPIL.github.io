import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Deterministic construction for n=25 (Affine Plane of order 5)
    # This guarantees a score of 25/11 = 2.2727
    result_25 = construct_affine_25()
    
    # 2. Try to extend to n=26 by fixing K_25 and solving for vertex 25
    if time.time() < deadline - 20:
        result_26 = try_extend(result_25, 26, time_limit=10)
        if result_26 is not None:
            return result_26

    # 3. If n=26 failed, try global local search for n=26 (might find different structure)
    if time.time() < deadline - 20:
        result_26_global = local_search_delta(26, time_limit=15)
        if result_26_global is not None:
            return result_26_global

    # 4. If n=26 failed, try to extend n=25 to n=27? Unlikely if 26 failed, but try incremental
    # Actually, if 26 failed, 27 is harder. 
    # Let's try global local search for n=25 just in case affine plane has a bug (it shouldn't)
    # But affine plane is solid.
    
    # 5. Fallback to n=25
    return result_25

def construct_affine_25():
    n = 25
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = [0] * len(edges)
    
    # Precompute inverses for mod 5
    inv = {1: 1, 2: 3, 3: 2, 4: 4}
    
    for idx, (u, v) in enumerate(edges):
        xu = u % 5
        yu = u // 5
        xv = v % 5
        yv = v // 5
        
        if xu == xv:
            colors[idx] = 4
        else:
            dx = (xv - xu) % 5
            dy = (yv - yu) % 5
            slope = (dy * inv[dx]) % 5
            colors[idx] = slope
            
    return {'n': n, 'colors': colors}

def try_extend(base_result, new_n, time_limit=10):
    """
    Takes a valid coloring for K_n (n = new_n - 1) and tries to add one vertex.
    Fixes the existing edges, only recolors edges incident to the new vertex.
    """
    n_old = base_result['n']
    n = new_n
    if n_old != n - 1:
        return None
        
    old_colors = base_result['colors']
    edges_old = [(i, j) for i in range(n_old) for j in range(i + 1, n_old)]
    
    # Build neighbor masks for the fixed K_n_old
    # neighbors[c][u] is bitmask of neighbors of u in color c
    neighbors = [[0] * n_old for _ in range(5)]
    for idx, (u, v) in enumerate(edges_old):
        c = old_colors[idx]
        neighbors[c][u] |= (1 << v)
        neighbors[c][v] |= (1 << u)
        
    # Edges for the new vertex (n_old)
    new_vertex = n_old
    new_edges = [(new_vertex, i) for i in range(n_old)]
    
    # Initial random colors for new edges
    new_colors = [random.randint(0, 4) for _ in range(n_old)]
    
    def count_violations(new_col_list):
        # Violations only come from C4s involving the new vertex
        # C4: new_vertex - u - w - z - new_vertex
        # Colors: c_new_u, c_uw, c_wz, c_new_z must be equal.
        # c_uw and c_wz are fixed.
        # So for each pair (u, z) and intermediate w, if c_new_u == c_uw == c_wz == c_new_z, it's a violation.
        # More efficiently:
        # For each color c, consider the set of vertices U_c = {u | new_color[u] == c}.
        # For each pair (u, z) in U_c, count the number of common neighbors w in the fixed graph with color c.
        # If count > 0, it forms a C4?
        # Wait. A C4 is a cycle of length 4.
        # new_vertex - u - w - z - new_vertex.
        # Edges: (new, u), (u, w), (w, z), (z, new).
        # If (u, w) and (w, z) are color c, and (new, u) and (new, z) are color c, then we have a monochromatic C4.
        # So for each color c, and for each u, z in U_c, if there exists w such that (u,w) is c and (w,z) is c, then we have a violation.
        # This is equivalent to saying that in the subgraph of color c, the set U_c must be an independent set in the *square* of the graph?
        # No, it means no two vertices in U_c can be at distance 2 in the color-c graph.
        
        violations = 0
        for c in range(5):
            U_c = [i for i in range(n_old) if new_col_list[i] == c]
            # For each u in U_c, check neighbors w in color c.
            # For each w, check neighbors z in U_c.
            # If z != u, then we have a path u-w-z.
            # Since u, z in U_c, edges (new,u) and (new,z) are c.
            # So new-u-w-z-new is a C4.
            # We just need to count such paths.
            # To avoid double counting, we can just check if any such path exists.
            # For efficiency, we can iterate u in U_c, then w in neighbors[c][u], then check if w has any neighbor z in U_c with z != u.
            
            # Precompute set of U_c for O(1) lookup?
            # n_old is 25. U_c size is ~5.
            # Just loop.
            for u in U_c:
                nu = neighbors[c][u]
                # Iterate through bits of nu
                w_mask = nu
                while w_mask:
                    # Extract lowest set bit
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    
                    # Check if w has any neighbor z in U_c, z != u
                    nw = neighbors[c][w]
                    # We need z in U_c such that z is in nw and z != u.
                    # Let's just check all z in U_c
                    for z in U_c:
                        if z == u:
                            continue
                        if nw & (1 << z):
                            violations += 1
                            # Break early? We just need to know if it's > 0 for SA, but count helps.
                            # Actually, just returning 1 if any violation is enough for boolean check, 
                            # but for SA we want magnitude.
                            
        return violations

    current_violations = count_violations(new_colors)
    best_violations = current_violations
    best_new_colors = new_colors[:]
    
    if current_violations == 0:
        # Build final colors
        final_colors = old_colors[:]
        # Append new colors in lexicographic order for edges (new_vertex, i)
        # Edges are (i, j) with i < j.
        # New vertex is n_old.
        # Edges involving new vertex are (i, n_old) for i < n_old.
        # In lexicographic order, these come AFTER all edges (i, j) with j < n_old.
        # So we just append them.
        final_colors.extend(best_new_colors)
        return {'n': n, 'colors': final_colors}

    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() < start_time + time_limit:
        idx = random.randint(0, n_old - 1)
        old_val = new_colors[idx]
        new_val = random.randint(0, 4)
        if new_val == old_val:
            continue
            
        new_colors[idx] = new_val
        new_violations = count_violations(new_colors)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < pow(2.71828, -delta / max(temperature, 0.01)):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_new_colors = new_colors[:]
                if best_violations == 0:
                    final_colors = old_colors[:]
                    final_colors.extend(best_new_colors)
                    return {'n': n, 'colors': final_colors}
        else:
            new_colors[idx] = old_val
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None

def local_search_delta(n, time_limit=15):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build neighbor bitmasks per color
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    def count_c4s():
        total = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                for v in range(u + 1, n):
                    common = nu & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        total += cnt * (cnt - 1) // 2
        return total
    
    current_violations = count_c4s()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    start_time = time.time()
    temperature = 3.0
    cooling_rate = 0.9998
    min_temp = 0.05
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Compute delta in violations
        delta = 0
        
        nu_old = neighbors[old_color][u]
        nv_old = neighbors[old_color][v]
        nu_new = neighbors[new_color][u]
        nv_new = neighbors[new_color][v]
        
        for w in range(n):
            if w == u or w == v:
                continue
            nw_old = neighbors[old_color][w]
            nw_new = neighbors[new_color][w]
            
            # Old color: removing (u,v)
            if (1 << v) & nw_old:
                old_k = (nu_old & nw_old).bit_count()
                delta -= (old_k - 1)
            if (1 << u) & nw_old:
                old_k = (nv_old & nw_old).bit_count()
                delta -= (old_k - 1)
            
            # New color: adding (u,v)
            if (1 << v) & nw_new:
                old_k = (nu_new & nw_new).bit_count()
                delta += old_k
            if (1 << u) & nw_new:
                old_k = (nv_new & nw_new).bit_count()
                delta += old_k
        
        # Apply the change
        colors[e_idx] = new_color
        neighbors[old_color][u] &= ~(1 << v)
        neighbors[old_color][v] &= ~(1 << u)
        neighbors[new_color][u] |= (1 << v)
        neighbors[new_color][v] |= (1 << u)
        
        current_violations += delta
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
        
        # Simulated annealing acceptance
        if delta >= 0:
            if random.random() < pow(2.71828, -delta / max(temperature, min_temp)):
                pass  # accept
            else:
                # Revert
                colors[e_idx] = old_color
                neighbors[new_color][u] &= ~(1 << v)
                neighbors[new_color][v] &= ~(1 << u)
                neighbors[old_color][u] |= (1 << v)
                neighbors[old_color][v] |= (1 << u)
                current_violations -= delta
        
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
    
    return None

