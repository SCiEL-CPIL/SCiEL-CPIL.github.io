import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def point_to_edge_distance(px, py, ax, ay, bx, by):
        """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
        dx = bx - ax
        dy = by - ay
        len_sq = dx*dx + dy*dy
        if len_sq < 1e-15:
            return math.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax)*dx + (py - ay)*dy) / len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def min_distance_between_hexes(h1, h2):
        """
        Compute minimum distance between two convex hexagons.
        Returns negative if overlapping (depth of penetration).
        Uses vertex-to-edge checks in both directions.
        """
        min_dist = float('inf')
        
        # Check each vertex of h1 against edges of h2
        for i in range(6):
            px, py = h1[i]
            for j in range(6):
                ax, ay = h2[j]
                bx, by = h2[(j+1) % 6]
                d = point_to_edge_distance(px, py, ax, ay, bx, by)
                min_dist = min(min_dist, d)
        
        # Check each vertex of h2 against edges of h1
        for i in range(6):
            px, py = h2[i]
            for j in range(6):
                ax, ay = h1[j]
                bx, by = h1[(j+1) % 6]
                d = point_to_edge_distance(px, py, ax, ay, bx, by)
                min_dist = min(min_dist, d)
        
        # If min_dist is positive, they're disjoint
        # If negative... wait, point_to_edge_distance always returns >= 0
        # We need to detect if a vertex is INSIDE the other hexagon
        
        # Better approach: check if any vertex of one is inside the other
        # For convex polygon, point is inside if it's on the same side of all edges
        
        def point_in_convex_poly(px, py, poly):
            """Check if point is inside convex polygon (CCW)."""
            for i in range(6):
                ax, ay = poly[i]
                bx, by = poly[(i+1) % 6]
                # Cross product to check if point is to the left of edge (inside for CCW)
                cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                if cross < -1e-10:  # Point is outside
                    return False
            return True
        
        # Check if any vertex of h1 is inside h2, or vice versa
        for i in range(6):
            if point_in_convex_poly(h1[i, 0], h1[i, 1], h2):
                return -1.0  # Overlapping
        for i in range(6):
            if point_in_convex_poly(h2[i, 0], h2[i, 1], h1):
                return -1.0  # Overlapping
        
        return min_dist

    def compute_min_side(hexes, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, given orientation)
        that contains all given hexagons.
        """
        normal_angles = outer_angle + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_objective_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2, outer_angle]
        Generates 12 hexagons using 6-fold symmetry.
        Returns the minimum outer side length required to contain them + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2, outer_angle = params
        
        # Generate the 12 hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # Check overlaps using fast method
        max_violation = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                
                dist = min_distance_between_hexes(hexes[i], hexes[j])
                if dist < 0:
                    # Overlapping - use a smooth penalty
                    # The violation depth isn't directly available, so use a fixed penalty
                    # scaled by how deep the overlap might be
                    max_violation = max(max_violation, 1.0)
        
        # Compute the minimum side length required
        s = compute_min_side(hexes, outer_angle)
        
        # Smooth penalty: use exponential to make it differentiable
        penalty = 50.0 * np.exp(max_violation * 5.0) if max_violation > 0 else 0.0
        
        return s + penalty

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons and outer angle
    # Based on known packings, hexagons should be in a ring around center
    bounds = [
        (1.0, 6.0),  # x1 - radial distance
        (-1.0, 1.0), # y1 - tangential offset
        (0.0, np.pi/3), # a1 - rotation
        (1.0, 6.0),  # x2 - radial distance
        (-1.0, 1.0), # y2 - tangential offset
        (0.0, np.pi/3), # a2 - rotation
        (0.0, np.pi/6)   # outer_angle
    ]
    
    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search
    try:
        res = differential_evolution(
            compute_objective_symmetric,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=False  # We'll do our own polish
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
            
    except Exception as e:
        pass
    
    # Refinement: Multiple local minimizations from diverse starts
    if time.time() < deadline - 15:
        # Generate diverse starting points
        start_points = []
        
        # From DE result
        if best_params is not None:
            start_points.append(best_params.copy())
        
        # Known good configurations
        # Configuration 1: Two hexagons at different radii
        start_points.append(np.array([
            3.0, 0.0, 0.0,
            4.5, 0.0, 0.0,
            0.0
        ]))
        
        # Configuration 2: Symmetric arrangement
        start_points.append(np.array([
            3.5, 0.5, np.pi/6,
            3.5, -0.5, -np.pi/6,
            0.0
        ]))
        
        # Configuration 3: Closer packing
        start_points.append(np.array([
            2.5, 0.0, 0.0,
            4.0, 0.8, np.pi/12,
            np.pi/12
        ]))
        
        # Random starts
        for _ in range(8):
            p = np.array([
                np.random.uniform(2.0, 5.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(0, np.pi/3),
                np.random.uniform(2.0, 5.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(0, np.pi/3),
                np.random.uniform(0, np.pi/6)
            ])
            start_points.append(p)
        
        for p in start_points:
            if time.time() > deadline - 5:
                break
            
            # Clip to bounds
            for i in range(len(p)):
                p[i] = max(bounds[i][0], min(bounds[i][1], p[i]))
            
            val = compute_objective_symmetric(p)
            if val < best_s:
                best_s = val
                best_params = p.copy()
            
            try:
                res_local = minimize(
                    compute_objective_symmetric,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        # Fallback
        best_params = np.array([3.0, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0])
    
    x1, y1, a1, x2, y2, a2, outer_angle = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
        inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
        
    outer_center = (0.0, 0.0)
    
    # Recompute the exact side length for the final configuration
    base1 = hex_vertices(x1, y1, 1.0, a1)
    base2 = hex_vertices(x2, y2, 1.0, a2)
    hexes = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        rot = np.array([[c, -s_], [s_, c]])
        h1 = base1 @ rot.T
        h2 = base2 @ rot.T
        hexes.append(h1)
        hexes.append(h2)
        
    outer_side = compute_min_side(hexes, outer_angle)
    outer_angle_deg = np.degrees(outer_angle)
    
    return inner, outer_center, outer_side, outer_angle_deg
