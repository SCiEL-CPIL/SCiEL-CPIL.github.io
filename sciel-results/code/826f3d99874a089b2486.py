import numpy as np
import time

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices_batch(xys_rots):
    """Batch compute vertices for all hexagons. xys_rots is (N, 3) array."""
    n_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    return np.stack((x, y), axis=-1)

def minkowski_hexagon_vertices(rot1_deg, rot2_deg):
    """
    Compute vertices of the Minkowski sum of two unit regular hexagons
    with given rotations. Returns (M, 2) array of vertices in CCW order.
    """
    def hex_verts(rot):
        angles = np.radians(np.array([0, 60, 120, 180, 240, 300]) + rot)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    v1 = hex_verts(rot1_deg)
    v2 = hex_verts(rot2_deg)
    
    # Minkowski sum: all pairwise sums
    sums = np.array([v1[i] + v2[j] for i in range(6) for j in range(6)])
    
    # Compute convex hull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(sums)
        hull_verts = sums[hull.vertices]
        # Ensure CCW order
        area = 0.0
        for i in range(len(hull_verts)):
            j = (i + 1) % len(hull_verts)
            area += hull_verts[i, 0] * hull_verts[j, 1] - hull_verts[j, 0] * hull_verts[i, 1]
        if area < 0:
            hull_verts = hull_verts[::-1]
        return hull_verts
    except Exception:
        return sums

def point_in_polygon(pt, polygon):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagons_disjoint(v1, v2, minkowski_verts, tol=1e-8):
    """
    Check if two unit hexagons are disjoint using Minkowski sum.
    v1, v2: (6,2) vertex arrays. minkowski_verts: (M,2) convex hull of Minkowski sum.
    """
    # Check if center of v2 is inside Minkowski sum of v1 and -v2
    # Equivalent: check if any vertex of v1 is inside Minkowski sum shifted by v2 center
    # Simpler: use SAT on the two hexagons
    all_axes = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                all_axes.append(n)
    
    for n in all_axes:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
            return True  # Separated
    return False

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """
    Compute outer hex side from a list of vertex arrays.
    """
    all_points = np.vstack(vertices_list)
    return outer_hex_side(all_points, outer_rot_deg)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_two_rings(gap=0.0):
        """Two concentric rings: 6 inner + 6 outer offset."""
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(6):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def ccd_trajectory_place(positions, angles, outer_rot, max_steps=200, step_size=0.1):
        """
        CCD-based trajectory optimization: slide hexagons into place
        using Minkowski sum collision detection.
        """
        if time.time() - start_time > DEADLINE - 10:
            return positions, angles, outer_rot
        
        # Precompute Minkowski sums for all angle pairs
        minkowski_cache = {}
        def get_minkowski(a1, a2):
            key = (round(a1, 1), round(a2, 1))
            if key not in minkowski_cache:
                minkowski_cache[key] = minkowski_hexagon_vertices(a1, a2)
            return minkowski_cache[key]
        
        # Try to improve by sliding each hexagon
        improved = True
        iterations = 0
        while improved and iterations < max_steps and time.time() - start_time < DEADLINE - 15:
            improved = False
            iterations += 1
            
            for i in range(N):
                if time.time() - start_time > DEADLINE - 15:
                    break
                
                # Try small perturbations in 8 directions
                best_pos = positions[i].copy()
                best_angle = angles[i]
                best_cost = outer_hex_side_batch(
                    [get_hex_vertices_batch(np.array([positions[i]]))[0] for i in range(N)], 
                    outer_rot
                )
                
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = positions[i].copy()
                        new_pos[0] += dx
                        new_pos[1] += dy
                        
                        # Check collision with all other hexagons
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            v1 = get_hex_vertices_batch(np.array([new_pos]))[0]
                            v2 = get_hex_vertices_batch(np.array([positions[j]]))[0]
                            if not hexagons_disjoint(v1, v2, None):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Compute new cost
                        test_positions = positions.copy()
                        test_positions[i] = new_pos
                        test_vertices = get_hex_vertices_batch(test_positions)
                        L = outer_hex_side_batch(test_vertices, outer_rot)
                        
                        if L < best_cost - 1e-8:
                            best_cost = L
                            best_pos = new_pos
                            best_angle = angles[i]
                            improved = True
                
                positions[i] = best_pos
                angles[i] = best_angle
        
        return positions, angles, outer_rot
    
    def local_search_refine(positions, angles, outer_rot, max_iter=100):
        """
        Local search with random restarts and simulated annealing.
        """
        if time.time() - start_time > DEADLINE - 10:
            return positions, angles, outer_rot
        
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_outer_rot = outer_rot
        best_L = outer_hex_side_batch(get_hex_vertices_batch(best_positions), best_outer_rot)
        
        T = 0.5
        T_min = 0.01
        cooling = 0.98
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Random perturbation
            new_positions = best_positions.copy()
            new_angles = best_angles.copy()
            new_outer_rot = best_outer_rot
            
            idx = np.random.randint(N)
            delta = np.random.randn(2) * 0.2
            new_positions[idx] += delta
            new_angles[idx] += np.random.randn() * 10
            new_outer_rot += np.random.randn() * 5
            
            # Check validity
            vertices = get_hex_vertices_batch(new_positions)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not hexagons_disjoint(vertices[i], vertices[j], None):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            L = outer_hex_side_batch(vertices, new_outer_rot)
            delta_L = L - best_L
            
            if delta_L < 0 or np.random.random() < np.exp(-delta_L / T):
                best_positions = new_positions
                best_angles = new_angles
                best_outer_rot = new_outer_rot
                best_L = L
            
            T *= cooling
        
        return best_positions, best_angles, best_outer_rot
    
    best_L = float('inf')
    best_result = None
    
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_two_rings(gap=gap))
    
    outer_rots = [0.0, 30.0, 15.0]
    
    from scipy.optimize import minimize
    
    # Phase 1: Nelder-Mead with smooth objective
    def objective_smooth(params, beta=50.0):
        if time.time() - start_time > DEADLINE - 10:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        vertices_list = get_hex_vertices_batch(xys_rots)
        
        # Simple overlap penalty using center distances
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(xys_rots[i, :2] - xys_rots[j, :2])
                if d < 2.0:
                    penalty += (2.0 - d) ** 2
        
        all_points = np.vstack(vertices_list)
        angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        proj = all_points @ normals.T
        abs_proj = np.abs(proj)
        scaled = beta * abs_proj
        max_scaled = np.max(scaled)
        soft_max = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
        L_soft = (2.0 / np.sqrt(3.0)) * soft_max
        
        return L_soft + 50.0 * penalty
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 25:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 25:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            try:
                res = minimize(
                    objective_smooth, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 80,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                
                vertices_list = get_hex_vertices_batch(xys_rots)
                
                # Check validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not hexagons_disjoint(vertices_list[i], vertices_list[j], None):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    L_val = outer_hex_side_batch(vertices_list, outer_rot)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_result = (xys_rots.copy(), outer_rot)
                        
            except Exception:
                pass
    
    # Phase 2: CCD trajectory optimization
    if best_result is not None and time.time() - start_time < DEADLINE - 20:
        positions = best_result[0][:, :2].copy()
        angles = best_result[0][:, 2].copy()
        outer_rot = best_result[1]
        
        positions, angles, outer_rot = ccd_trajectory_place(positions, angles, outer_rot)
        
        xys_rots = np.column_stack((positions, angles))
        vertices_list = get_hex_vertices_batch(xys_rots)
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(vertices_list[i], vertices_list[j], None):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            L_val = outer_hex_side_batch(vertices_list, outer_rot)
            if L_val < best_L:
                best_L = L_val
                best_result = (xys_rots.copy(), outer_rot)
    
    # Phase 3: Local search refinement
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        positions = best_result[0][:, :2].copy()
        angles = best_result[0][:, 2].copy()
        outer_rot = best_result[1]
        
        positions, angles, outer_rot = local_search_refine(positions, angles, outer_rot)
        
        xys_rots = np.column_stack((positions, angles))
        vertices_list = get_hex_vertices_batch(xys_rots)
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(vertices_list[i], vertices_list[j], None):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            L_val = outer_hex_side_batch(vertices_list, outer_rot)
            if L_val < best_L:
                best_L = L_val
                best_result = (xys_rots.copy(), outer_rot)
    
    if best_result is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    xys_rots = best_result[0]
    outer_rot = best_result[1]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
