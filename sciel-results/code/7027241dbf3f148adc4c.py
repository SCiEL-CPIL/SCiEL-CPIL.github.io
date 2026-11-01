"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def hex_vertices(cx, cy, side, angle_deg):
        theta = np.radians(angle_deg)
        angles = theta + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])
    
    def min_dist_polygons(v1, v2):
        """Compute minimum distance between two convex polygons. Negative if overlapping."""
        min_d = np.inf
        n1 = len(v1)
        n2 = len(v2)
        
        # For each edge of v1, find closest point on v2
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1) % n1]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-10:
                continue
            edge_dir = edge / edge_len
            
            # Project all vertices of v2 onto this edge
            for j in range(n2):
                v = v2[j]
                t = np.dot(v - p1, edge_dir)
                if t < 0:
                    closest = p1
                elif t > edge_len:
                    closest = p2
                else:
                    closest = p1 + t * edge_dir
                d = np.linalg.norm(v - closest)
                if d < min_d:
                    min_d = d
            
            # Also check if any vertex of v2 is inside the strip
            # and compute perpendicular distance
            for j in range(n2):
                v = v2[j]
                t = np.dot(v - p1, edge_dir)
                if 0 <= t <= edge_len:
                    perp = v - (p1 + t * edge_dir)
                    d = np.linalg.norm(perp)
                    if d < min_d:
                        min_d = d
        
        # Check vertex-to-vertex (redundant but safe)
        for i in range(n1):
            for j in range(n2):
                d = np.linalg.norm(v1[i] - v2[j])
                if d < min_d:
                    min_d = d
        
        # Check if polygons overlap using SAT
        def project_poly(verts, axis):
            dots = np.dot(verts, axis)
            return np.min(dots), np.max(dots)
        
        def get_axes(verts):
            axes = []
            for i in range(len(verts)):
                edge = verts[(i+1) % len(verts)] - verts[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    axes.append(normal / norm)
            return axes
        
        axes = get_axes(v1) + get_axes(v2)
        for axis in axes:
            min1, max1 = project_poly(v1, axis)
            min2, max2 = project_poly(v2, axis)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return min_d  # Disjoint
        
        # Overlapping - compute penetration depth approximation
        # Use the minimum of the negative gaps
        for axis in axes:
            min1, max1 = project_poly(v1, axis)
            min2, max2 = project_poly(v2, axis)
            gap = min(max1 - min2, max2 - min1)
            if gap < 0:
                min_d = min(min_d, gap)
        
        return min_d
    
    def soft_overlap_penalty(verts_list, beta=20.0):
        """Smooth penalty for overlaps using soft-min."""
        penalty = 0.0
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                d = min_dist_polygons(verts_list[i], verts_list[j])
                if d < 0.05:
                    # Soft penalty: smooth approximation of max(0, -d)^2
                    penalty += np.exp(-beta * d) if d < 0 else 0.0
        return penalty
    
    def compute_required_side(inner_hexes, outer_cx=0.0, outer_cy=0.0, outer_angle=0.0):
        """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
        max_s = 0.0
        for (ix, iy, iang) in inner_hexes:
            verts = hex_vertices(ix, iy, 1.0, iang)
            for (vx, vy) in verts:
                dx = vx - outer_cx
                dy = vy - outer_cy
                theta = np.radians(outer_angle)
                cos_t = np.cos(-theta)
                sin_t = np.sin(-theta)
                lx = dx * cos_t - dy * sin_t
                ly = dx * sin_t + dy * cos_t
                # For a regular hexagon with side s centered at origin, the support function
                # in direction (nx, ny) is s * max_k (nx*cos(60k) + ny*sin(60k))
                # The constraint is: proj <= s * (sqrt(3)/2) for the "flat" directions
                # Actually, for a hexagon with vertices at angles 0,60,...,300 and side s,
                # the distance from center to edge (apothem) is s*sqrt(3)/2
                # The support in direction u is s * max_k (u · e_k) where e_k are unit normals to edges
                # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
                for k in range(6):
                    normal_angle = np.radians(30 + 60 * k)
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    proj = lx * nx + ly * ny
                    if proj > 0:
                        s_req = proj / (np.sqrt(3) / 2)
                        if s_req > max_s:
                            max_s = s_req
        return max_s
    
    def objective_smooth(params, outer_angle=0.0):
        """Smooth objective: required side + overlap penalty."""
        inner_hexes = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            inner_hexes.append((x, y, 0.0))
        
        # Center the configuration
        cx = np.mean([h[0] for h in inner_hexes])
        cy = np.mean([h[1] for h in inner_hexes])
        
        side = compute_required_side(inner_hexes, cx, cy, outer_angle)
        
        verts_list = [hex_vertices(x - cx, y - cy, 1.0, 0.0) for (x, y, _) in inner_hexes]
        pen = soft_overlap_penalty(verts_list)
        
        return side + pen
    
    def objective_hard(params, outer_angle=0.0):
        """Hard objective for verification."""
        inner_hexes = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            inner_hexes.append((x, y, 0.0))
        
        cx = np.mean([h[0] for h in inner_hexes])
        cy = np.mean([h[1] for h in inner_hexes])
        
        side = compute_required_side(inner_hexes, cx, cy, outer_angle)
        
        verts_list = [hex_vertices(x - cx, y - cy, 1.0, 0.0) for (x, y, _) in inner_hexes]
        
        # Check for overlaps
        has_overlap = False
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                d = min_dist_polygons(verts_list[i], verts_list[j])
                if d < -1e-6:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            return side + 1000.0
        
        return side
    
    # Generate honeycomb initial configuration
    sqrt3 = np.sqrt(3)
    
    def honeycomb_config(n_rows=3, n_cols=4, offset=0.0):
        """Generate honeycomb lattice positions for N hexagons."""
        params = []
        count = 0
        for row in range(n_rows):
            y = row * sqrt3
            row_offset = 0.0 if row % 2 == 0 else 0.75
            for col in range(n_cols):
                if count >= N:
                    break
                x = col * 1.5 - 2.25 + row_offset + offset
                params.append(x)
                params.append(y)
                count += 1
        return np.array(params)
    
    best_params = None
    best_score = np.inf
    
    # Try multiple initial configurations and optimization methods
    configs = [
        honeycomb_config(3, 4, 0.0),
        honeycomb_config(3, 4, 0.1),
        honeycomb_config(3, 4, -0.1),
        honeycomb_config(3, 4, 0.2),
    ]
    
    # Also try a more compact arrangement
    # 2 rows of 6, but staggered
    params2 = []
    for row in range(2):
        y = row * sqrt3
        for col in range(6):
            x = col * 1.5 - 3.75
            if row == 1:
                x += 0.75
            params2.append(x)
            params2.append(y)
    configs.append(np.array(params2))
    
    # 4 rows of 3
    params3 = []
    for row in range(4):
        y = row * sqrt3
        for col in range(3):
            x = col * 1.5 - 1.5
            if row % 2 == 1:
                x += 0.75
            params3.append(x)
            params3.append(y)
    configs.append(np.array(params3))
    
    for cfg in configs:
        if time.time() > deadline - 15.0:
            break
        
        # Center the config
        cx_arr = cfg[0::2]
        cy_arr = cfg[1::2]
        cfg[0::2] = cx_arr - np.mean(cx_arr)
        cfg[1::2] = cy_arr - np.mean(cy_arr)
        
        # First, evaluate hard objective
        hard_score = objective_hard(cfg)
        if hard_score < best_score and hard_score < 100:
            best_score = hard_score
            best_params = cfg.copy()
        
        # Run Nelder-Mead on smooth objective
        try:
            result = minimize(objective_smooth, cfg, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
            if result.fun < best_score:
                # Verify with hard objective
                hard_s = objective_hard(result.x)
                if hard_s < best_score:
                    best_score = hard_s
                    best_params = result.x.copy()
        except Exception:
            pass
        
        # Also try L-BFGS-B with smooth objective
        try:
            result2 = minimize(objective_smooth, cfg, method='L-BFGS-B',
                              options={'maxiter': 1000, 'gtol': 1e-6})
            if result2.fun < best_score:
                hard_s = objective_hard(result2.x)
                if hard_s < best_score:
                    best_score = hard_s
                    best_params = result2.x.copy()
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead on the best found
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            result_final = minimize(objective_smooth, best_params, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9,
                                            'adaptive': True})
            hard_s = objective_hard(result_final.x)
            if hard_s < best_score:
                best_score = hard_s
                best_params = result_final.x.copy()
        except Exception:
            pass
    
    # Construct final answer
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[best_params[2*i], best_params[2*i+1], 0.0] for i in range(N)]
    
    # Center the inner hexagons
    cx = np.mean([h[0] for h in inner])
    cy = np.mean([h[1] for h in inner])
    for i in range(N):
        inner[i][0] -= cx
        inner[i][1] -= cy
    
    # Verify no overlaps
    verts_list = [hex_vertices(x, y, 1.0, 0.0) for (x, y, _) in inner]
    has_overlap = False
    for i in range(len(verts_list)):
        for j in range(i+1, len(verts_list)):
            d = min_dist_polygons(verts_list[i], verts_list[j])
            if d < -1e-6:
                has_overlap = True
                break
        if has_overlap:
            break
    
    if has_overlap:
        # Fall back to a valid but suboptimal config
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Compute the actual required side length
    side = compute_required_side(inner, 0.0, 0.0, 0.0)
    
    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
