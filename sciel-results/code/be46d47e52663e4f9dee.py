# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    def get_hex_vertices(cx, cy, d, angle_deg):
        """
        Returns vertices of a regular hexagon.
        cx, cy: center
        d: side length
        angle_deg: rotation in degrees
        Vertices are at angle + 60*i
        """
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
        xs = cx + d * np.cos(angles)
        ys = cy + d * np.sin(angles)
        return np.column_stack((xs, ys))

    def get_edge_normals(vertices):
        """
        Returns outward normals of the edges of a convex polygon (vertices in CCW order).
        """
        n = len(vertices)
        normals = np.zeros((n, 2))
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            edge = p2 - p1
            # Outward normal for CCW polygon: (edge[1], -edge[0])
            normals[i] = np.array([edge[1], -edge[0]])
            # Normalize
            norm = np.linalg.norm(normals[i])
            if norm > 1e-10:
                normals[i] /= norm
        return normals

    def check_separation(v1, v2, normals1, normals2):
        """
        Checks if two convex polygons are separated (non-overlapping).
        Uses SAT.
        """
        all_normals = np.vstack((normals1, normals2))
        
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True # Separated
        
        return False # Overlapping or touching

    def is_point_in_polygon(point, vertices):
        """
        Check if point is inside convex polygon.
        vertices in CCW order.
        """
        n = len(vertices)
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            edge = p2 - p1
            # Normal pointing inward: (-edge[1], edge[0])
            normal_in = np.array([-edge[1], edge[0]])
            # Vector from p1 to point
            v = point - p1
            # Dot product must be >= 0 for all edges if inside
            if np.dot(normal_in, v) < -1e-9:
                return False
        return True

    def objective(params):
        """
        params: [x0, y0, ..., x11, y11, s, theta_outer]
        Returns penalty value. Lower is better. 
        """
        xys = params[:24].reshape(12, 2)
        s = params[24]
        theta_outer = params[25]
        
        # Basic bounds check
        if s < 3.46:
            return 1e6 + s
            
        # Outer hexagon vertices
        outer_verts = get_hex_vertices(0, 0, s, theta_outer)
        
        # Inner hexagons (all angle 0)
        inner_verts_list = []
        for i in range(12):
            cx, cy = xys[i]
            verts = get_hex_vertices(cx, cy, 1.0, 0.0)
            inner_verts_list.append(verts)
            
        # Check containment
        for i in range(12):
            for v in inner_verts_list[i]:
                if not is_point_in_polygon(v, outer_verts):
                    return 1000.0 + s + 10.0 * np.sum(np.abs(xys))
                    
        # Check inner-inner overlap
        ref_verts = get_hex_vertices(0, 0, 1.0, 0.0)
        ref_normals = get_edge_normals(ref_verts)
        
        for i in range(12):
            for j in range(i + 1, 12):
                v1 = inner_verts_list[i]
                v2 = inner_verts_list[j]
                
                # Quick bounding box check
                if np.all(np.abs(xys[i] - xys[j]) > 2.5):
                    continue
                    
                if not check_separation(v1, v2, ref_normals, ref_normals):
                    return 1000.0 + s + 10.0 * np.sum(np.abs(xys))
                    
        # If all constraints satisfied, return s (to minimize)
        com = np.mean(xys, axis=0)
        penalty = s + 0.01 * (com[0]**2 + com[1]**2)
        
        return penalty

    bounds = [(-10, 10)] * 24
    bounds.append((3.46, 10))
    bounds.append((0, 60))
    
    start_time = time.time()
    timeout = 500 # seconds
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=10, 
            tol=1e-6,
            polish=True
        )
    except Exception as e:
        print("DE failed:", e)
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    elapsed = time.time() - start_time
    if elapsed > timeout or not hasattr(res, 'x'):
        print("DE timed out or failed")
        if hasattr(res, 'x') and res.x is not None:
             xys = res.x[:24].reshape(12, 2)
             s = res.x[24]
             theta = res.x[25]
             com = np.mean(xys, axis=0)
             xys -= com
             inner_hex_data = np.column_stack((xys, np.zeros(12)))
             outer_hex_data = np.array([0, 0, theta])
             outer_hex_side_length = s
             return inner_hex_data, outer_hex_data, outer_hex_side_length
        else:
            inner_hex_data = np.array([
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0]
            ])
            outer_hex_data = np.array([0, 0, 0])
            outer_hex_side_length = 8
            return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Polish with Nelder-Mead
    try:
        res_local = minimize(
            objective, 
            res.x, 
            method='Nelder-Mead', 
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 2000}
        )
        if res_local.fun < res.fun:
            res = res_local
    except Exception as e:
        print("Local minimization failed:", e)

    xys = res.x[:24].reshape(12, 2)
    s = res.x[24]
    theta = res.x[25]
    
    com = np.mean(xys, axis=0)
    xys -= com
    
    inner_hex_data = np.column_stack((xys, np.zeros(12)))
    outer_hex_data = np.array([0, 0, theta])
    outer_hex_side_length = s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
