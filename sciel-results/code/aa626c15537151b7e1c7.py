import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin

    # Fallback solution n=11 (from seed)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    def count_c4s_in_coloring(colors, n):
        """
        Count total monochromatic C4s in a coloring of K_n.
        colors: list of length n*(n-1)//2
        """
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return float('inf')
        
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        total_cost = 0.0
        for m in mats:
            M = m @ m
            # M[u, v] is number of common neighbors
            # C4 count = sum_{u < v} binom(M[u, v], 2)
            tri_u, tri_v = np.triu_indices(n, 1)
            vals = M[tri_u, tri_v]
            total_cost += np.sum(vals * (vals - 1) / 2.0)
        return total_cost

    def verify(colors, n):
        return count_c4s_in_coloring(colors, n) == 0

    # --- Strategy: Projective Plane of Order 5 (n=31) ---
    
    def get_pg25_points():
        """
        Generate all 31 points of PG(2, 5) as normalized tuples (x, y, z)
        where x, y, z are in GF(5) and the first non-zero is 1.
        """
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: divide by first non-zero
                    if x != 0:
                        inv_x = pow(x, 3, 5) # x^(-1) mod 5
                        points.append((1, (y * inv_x) % 5, (z * inv_x) % 5))
                    elif y != 0:
                        inv_y = pow(y, 3, 5)
                        points.append((0, 1, (z * inv_y) % 5))
                    else: # z != 0
                        inv_z = pow(z, 3, 5)
                        points.append((0, 0, 1))
        return points

    def get_pg25_lines():
        """
        Generate all 31 lines of PG(2, 5) as normalized coefficient tuples (a, b, c)
        representing ax + by + cz = 0.
        Same structure as points.
        """
        return get_pg25_points()

    def get_incidence(points, lines):
        """
        Return incidence matrix P (31x31) where P[i, j] = 1 if point i is on line j.
        Also return intersection matrix I (31x31) where I[i, j] = index of point 
        at intersection of line i and line j.
        """
        n_pts = len(points)
        n_lns = len(lines)
        
        # Incidence: Point p is on Line l if dot(p, l) == 0
        inc = np.zeros((n_pts, n_lns), dtype=np.int8)
        for i, p in enumerate(points):
            for j, l in enumerate(lines):
                val = (p[0]*l[0] + p[1]*l[1] + p[2]*l[2]) % 5
                if val == 0:
                    inc[i, j] = 1
        
        # Intersection: Line l1 and l2 intersect at point p.
        # p is the cross product of l1 and l2 in GF(5)^3.
        # l1 = (a1, b1, c1), l2 = (a2, b2, c2)
        # p = (b1*c2 - c1*b2, c1*a2 - a1*c2, a1*b2 - b1*a2)
        
        # Map normalized point tuple to index
        point_to_idx = {p: i for i, p in enumerate(points)}
        
        inter = np.zeros((n_lns, n_lns), dtype=np.int32)
        for i in range(n_lns):
            for j in range(n_lns):
                if i == j:
                    continue
                l1 = lines[i]
                l2 = lines[j]
                a1, b1, c1 = l1
                a2, b2, c2 = l2
                
                px = (b1*c2 - c1*b2) % 5
                py = (c1*a2 - a1*c2) % 5
                pz = (a1*b2 - b1*a2) % 5
                
                # Normalize (px, py, pz)
                if px != 0:
                    inv = pow(px, 3, 5)
                    norm_p = (1, (py*inv)%5, (pz*inv)%5)
                elif py != 0:
                    inv = pow(py, 3, 5)
                    norm_p = (0, 1, (pz*inv)%5)
                else:
                    inv = pow(pz, 3, 5)
                    norm_p = (0, 0, 1)
                
                inter[i, j] = point_to_idx[norm_p]
        
        return inc, inter

    def solve_pg25():
        """
        Search for a point-coloring of PG(2,5) that induces a C4-free edge-coloring on lines.
        """
        points = get_pg25_points()
        lines = get_pg25_lines()
        n = 31
        inc, inter = get_incidence(points, lines)
        
        # Precompute for each point p, the list of lines passing through it.
        # This helps in delta updates or just checking.
        # But we will use full recount for simplicity first, as n=31 is small.
        
        best_point_colors = None
        best_cost = float('inf')
        
        max_restarts = 50
        max_iter = 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization of point colors
            p_colors = [random.randint(0, 4) for _ in range(n)]
            
            # Function to get edge colors from point colors
            def get_edge_colors(p_col):
                e_colors = [0] * (n * (n - 1) // 2)
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        # Edge (line i, line j) color is color of intersection point
                        p_idx = inter[i, j]
                        e_colors[idx] = p_col[p_idx]
                        idx += 1
                return e_colors
            
            e_colors = get_edge_colors(p_colors)
            current_cost = count_c4s_in_coloring(e_colors, n)
            
            if current_cost == 0:
                return n, e_colors
            
            best_cost = min(best_cost, current_cost)
            if current_cost < best_cost:
                best_point_colors = p_colors.copy()
            
            # Local Search on Point Colors
            for i in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Pick random point to recolor
                p_idx = random.randint(0, n - 1)
                old_c = p_colors[p_idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                
                p_colors[p_idx] = new_c
                e_colors_new = get_edge_colors(p_colors)
                new_cost = count_c4s_in_coloring(e_colors_new, n)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_point_colors = p_colors.copy()
                    if current_cost == 0:
                        return n, e_colors_new
                else:
                    # Revert
                    p_colors[p_idx] = old_c
                    # e_colors is still valid for old state
            
        if best_point_colors is not None:
            e_colors = get_edge_colors(best_point_colors)
            if count_c4s_in_coloring(e_colors, n) == 0:
                return n, e_colors
                
        return None

    # Try Projective Plane
    res = solve_pg25()
    if res:
        best_n, best_colors = res
    else:
        # Fallback to corrected stochastic search for n up to 26
        # We know n=26 is possible.
        
        def solve_stochastic(n):
            num_edges = n * (n - 1) // 2
            max_iter = 50000
            max_restarts = 10
            
            endpoints = []
            edge_index = {}
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    edge_index[(i, j)] = idx
                    endpoints.append((i, j))
                    idx += 1
            
            for restart in range(max_restarts):
                if time.time() > deadline - 5:
                    return None
                
                colors = [random.randint(0, 4) for _ in range(num_edges)]
                
                # Build symmetric CN matrices
                CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
                neighbors = [ [set() for _ in range(5)] for _ in range(n) ]
                
                for e_idx in range(num_edges):
                    u, v = endpoints[e_idx]
                    c = colors[e_idx]
                    neighbors[u][c].add(v)
                    neighbors[v][c].add(u)
                
                for c in range(5):
                    for k in range(n):
                        nbrs = list(neighbors[k][c])
                        for i in range(len(nbrs)):
                            u = nbrs[i]
                            for j in range(i + 1, len(nbrs)):
                                v = nbrs[j]
                                CN[c][u, v] += 1
                                CN[c][v, u] += 1
                
                current_cost = 0.0
                for c in range(5):
                    tri_u, tri_v = np.triu_indices(n, 1)
                    vals = CN[c][tri_u, tri_v]
                    current_cost += np.sum(vals * (vals - 1) / 2.0)
                
                if current_cost == 0:
                    return colors
                
                for i in range(max_iter):
                    if time.time() > deadline - 5:
                        return None
                    if current_cost == 0:
                        return colors
                    
                    e_idx = random.randint(0, num_edges - 1)
                    u, v = endpoints[e_idx]
                    old_c = colors[e_idx]
                    new_c = random.randint(0, 4)
                    if old_c == new_c:
                        continue
                    
                    delta = 0.0
                    # Calculate delta
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        
                        key_vx = (v, x) if v < x else (x, v)
                        idx_vx = edge_index[key_vx]
                        c_vx = colors[idx_vx]
                        
                        key_ux = (u, x) if u < x else (x, u)
                        idx_ux = edge_index[key_ux]
                        c_ux = colors[idx_ux]
                        
                        if c_vx == old_c:
                            val = CN[old_c][u, x]
                            delta -= (val - 1)
                        if c_ux == old_c:
                            val = CN[old_c][v, x]
                            delta -= (val - 1)
                        if c_vx == new_c:
                            val = CN[new_c][u, x]
                            delta += val
                        if c_ux == new_c:
                            val = CN[new_c][v, x]
                            delta += val
                        
                    new_cost = current_cost + delta
                    if new_cost < current_cost:
                        current_cost = new_cost
                        colors[e_idx] = new_c
                        
                        # Update CN
                        for x in range(n):
                            if x == u or x == v:
                                continue
                            
                            key_vx = (v, x) if v < x else (x, v)
                            idx_vx = edge_index[key_vx]
                            # Use old colors for update logic? 
                            # c_vx was read before update. c_ux was read before update.
                            # But we need to update based on old_c and new_c.
                            
                            if c_vx == old_c:
                                CN[old_c][u, x] -= 1
                                CN[old_c][x, u] -= 1
                            if c_ux == old_c:
                                CN[old_c][v, x] -= 1
                                CN[old_c][x, v] -= 1
                            if c_vx == new_c:
                                CN[new_c][u, x] += 1
                                CN[new_c][x, u] += 1
                            if c_ux == new_c:
                                CN[new_c][v, x] += 1
                                CN[new_c][x, v] += 1
                            
                        # Update neighbors
                        neighbors[u][old_c].remove(v)
                        neighbors[v][old_c].remove(u)
                        neighbors[u][new_c].add(v)
                        neighbors[v][new_c].add(u)
                        
                        if current_cost == 0:
                            return colors

        # Try n=26 first, then down to 12
        for n in range(26, 11, -1):
            if time.time() > deadline - 10:
                break
            res = solve_stochastic(n)
            if res:
                best_n = n
                best_colors = res
                break

    return {'n': best_n, 'colors': best_colors}
