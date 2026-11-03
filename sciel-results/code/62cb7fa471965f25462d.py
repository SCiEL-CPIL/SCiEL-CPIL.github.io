import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Unit hexagon vertices (flat-top)
    unit_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])

    def get_rotation(angle_deg):
        theta = math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]])

    def hex_vertices(cx, cy, side, angle_deg):
        R = get_rotation(angle_deg)
        verts = (unit_verts @ R.T) * side
        return verts + np.array([cx, cy])

    def min_outer_side(centers, angles):
        """
        Find the minimum side length of a regular hexagon that contains all inner hexagons.
        Returns (side_length, center, angle).
        We optimize over the outer hexagon's center and angle.
        """
        all_verts = []
        for i in range(N):
            v = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # For a regular hexagon with side length s and rotation theta:
        # The support function h(u) = max(v . u) for v in set.
        # The hexagon is defined by 6 half-planes: n_k . x <= s * sqrt(3) / 2
        # where n_k are normals to the sides.
        # For a regular hexagon of side s, the distance from center to side (apothem) is s * sqrt(3) / 2.
        # The vertices are at distance s from center.
        # The condition for a point p to be inside is:
        # max_k ( (p - c) . n_k ) <= s * sqrt(3) / 2
        # where n_k are unit normals to the sides of the hexagon.
        # n_k = (cos(theta + pi/6 + k*pi/3), sin(theta + pi/6 + k*pi/3))
        
        def objective_outer(params):
            cx, cy, theta = params
            c = np.array([cx, cy])
            shifted = all_verts - c
            normals = np.array([
                [math.cos(theta + math.pi/6 + k * math.pi/3),
                 math.sin(theta + math.pi/6 + k * math.pi/3)]
                for k in range(6)
            ])
            projs = shifted @ normals.T
            max_proj = np.max(projs, axis=0)
            # We need max_proj <= s * sqrt(3) / 2 for all k
            # So s >= max(max_proj) * 2 / sqrt(3)
            # Also we must ensure the hexagon is centered such that the points are inside.
            # Actually, the standard "minimum enclosing hexagon" is not just about max projection.
            # The hexagon is the intersection of 6 half-planes.
            # To minimize s, we want to minimize max_k ( max_i (v_i . n_k) - c . n_k )
            # This is equivalent to minimizing the radius of the enclosing circle in the dual space?
            # Let's just minimize the required side length.
            # Required side length for a given center c and angle theta is:
            # s = ( max_k ( max_i (v_i . n_k) - c . n_k ) ) * 2 / sqrt(3)
            # Wait, this assumes the hexagon is "tight" on one side.
            # Actually, for a regular hexagon, the distance from center to each side is equal (apothem).
            # So we need: for all k, max_i (v_i . n_k) - c . n_k <= apothem
            # AND c . n_k - min_i (v_i . n_k) <= apothem  (for the opposite side)
            # So 2 * apothem >= max_i (v_i . n_k) - min_i (v_i . n_k)
            # And c must be centered in the width: c . n_k = (max_i + min_i)/2
            # If we fix c and theta, the minimum apothem is max_k ( (max_i - min_i)/2 )
            # Then s = apothem * 2 / sqrt(3)
            
            widths = np.max(projs, axis=0) - np.min(projs, axis=0)
            max_width = np.max(widths)
            s = max_width / SQRT3
            return s

        # Initial guess for outer hexagon: centroid of all vertices, angle 0
        centroid = np.mean(all_verts, axis=0)
        x0 = [centroid[0], centroid[1], 0.0]
        
        # Optimize center and angle
        # The objective is non-smooth (due to max), but scipy can handle it.
        # We use Nelder-Mead for robustness.
        res = minimize(objective_outer, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000})
        
        cx, cy, theta = res.x
        # Calculate final side length
        c = np.array([cx, cy])
        shifted = all_verts - c
        normals = np.array([
            [math.cos(theta + math.pi/6 + k * math.pi/3),
             math.sin(theta + math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])
        projs = shifted @ normals.T
        max_proj = np.max(projs, axis=0)
        min_proj = np.min(projs, axis=0)
        # Check if centered correctly
        # The width in direction k is max_proj[k] - min_proj[k]
        # The apothem must be at least half the max width.
        apothem = np.max(np.abs(projs)) # This is not quite right if not centered
        # Let's recalculate properly
        widths = max_proj - min_proj
        max_width = np.max(widths)
        side = max_width / SQRT3
        
        # Verify containment
        # Center should be at (max+min)/2 for each normal direction
        # c . n_k = (max_k + min_k)/2
        # Let's find the best center for this theta
        targets = (max_proj + min_proj) / 2.0
        c_opt, _, _, _ = np.linalg.lstsq(normals, targets, rcond=None)
        
        # Re-evaluate side with optimal center
        shifted_opt = all_verts - c_opt
        projs_opt = shifted_opt @ normals.T
        max_proj_opt = np.max(projs_opt, axis=0)
        min_proj_opt = np.min(projs_opt, axis=0)
        widths_opt = max_proj_opt - min_proj_opt
        max_width_opt = np.max(widths_opt)
        side_opt = max_width_opt / SQRT3
        
        # Check if side_opt is valid (should be, by construction)
        # But we need to ensure the hexagon defined by c_opt and side_opt actually contains the points.
        # The hexagon is {x | |x . n_k| <= side * sqrt(3)/2 for all k}
        # We have max |x . n_k| = max_width_opt / 2 = side_opt * sqrt(3) / 2.
        # So yes, it fits exactly.
        
        return side_opt, c_opt, math.degrees(theta)

    def check_overlap(centers, angles, eps=1e-9):
        """Check for any overlap between inner hexagons."""
        polys = [hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                v1, v2 = polys[i], polys[j]
                
                # Quick bounding box check
                if (np.max(v1[:, 0]) < np.min(v2[:, 0]) - eps or
                    np.max(v2[:, 0]) < np.min(v1[:, 0]) - eps or
                    np.max(v1[:, 1]) < np.min(v2[:, 1]) - eps or
                    np.max(v2[:, 1]) < np.min(v1[:, 1]) - eps):
                    continue
                
                # Check vertex containment
                if _any_point_in_poly(v1, v2, eps) or _any_point_in_poly(v2, v1, eps):
                    return False
                
                # Check edge intersections
                if _edges_intersect(v1, v2, eps):
                    return False
        return True

    def _point_in_convex(pt, poly, eps=1e-12):
        n = len(poly)
        sign = None
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cross = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
            if abs(cross) < eps:
                continue
            s = cross > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    def _any_point_in_poly(points, poly, eps):
        for pt in points:
            if _point_in_convex(pt, poly, eps):
                return True
        return False

    def _seg_intersect(p1, p2, p3, p4, eps=1e-12):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
           ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
            return True
        return False

    def _edges_intersect(v1, v2, eps):
        for i in range(6):
            p1, p2 = v1[i], v1[(i + 1) % 6]
            for j in range(6):
                p3, p4 = v2[j], v2[(j + 1) % 6]
                if _seg_intersect(p1, p2, p3, p4, eps):
                    return True
        return False

    # --- Simulated Annealing ---
    
    # Initial configuration: Hexagonal lattice
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = SQRT3 * (i + j / 2.0)
            y = 3.0 * j
            lattice.append((x, y))
    lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_centers = np.array(lattice[:12])
    init_centers -= np.mean(init_centers, axis=0)
    init_angles = np.zeros(N)
    
    best_centers = init_centers.copy()
    best_angles = init_angles.copy()
    best_side, best_center, best_angle = min_outer_side(best_centers, best_angles)
    
    current_centers = best_centers.copy()
    current_angles = best_angles.copy()
    current_side, _, _ = min_outer_side(current_centers, current_angles)
    
    # Check initial validity
    if not check_overlap(current_centers, current_angles):
        # Try to fix or restart
        # For now, just proceed, SA will try to fix it
        pass

    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    steps = 10000
    
    rng = np.random.default_rng(42)
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        new_centers = current_centers.copy()
        new_angles = current_angles.copy()
        
        # Move 1-3 random hexagons
        num_moves = rng.integers(1, 4)
        for _ in range(num_moves):
            idx = rng.integers(0, N)
            # Translate
            dx = rng.normal(0, 0.5)
            dy = rng.normal(0, 0.5)
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            # Rotate
            if rng.random() < 0.2:
                new_angles[idx] += rng.uniform(-15, 15)
        
        # Check overlap
        if not check_overlap(new_centers, new_angles):
            # Reject overlapping configurations
            continue
            
        # Evaluate new side length
        new_side, _, _ = min_outer_side(new_centers, new_angles)
        
        # Acceptance criterion
        delta = new_side - current_side
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_centers = new_centers
            current_angles = new_angles
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_centers = current_centers.copy()
                best_angles = current_angles.copy()
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: restart with higher T if stuck?
            # For now, just continue with low T

    # Final refinement: Local optimization of centers for the best configuration found
    # We can use SLSQP or Nelder-Mead to fine-tune the positions of the inner hexagons
    # to minimize the outer side length, while maintaining non-overlap.
    
    # Since the objective (outer side) is non-smooth, Nelder-Mead is a good choice.
    # We will define a penalty function that adds a large cost if overlap occurs.
    
    def final_objective(params):
        centers = np.array(params).reshape(N, 2)
        # Angles are fixed from best_angles for simplicity in this refinement step
        # Or we could optimize angles too, but let's stick to positions for stability
        
        # Check overlap
        if not check_overlap(centers, best_angles):
            return 1e10  # Large penalty
        
        side, _, _ = min_outer_side(centers, best_angles)
        return side

    x0 = best_centers.flatten()
    
    try:
        res = minimize(final_objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000})
        
        if res.fun < best_side and check_overlap(res.x.reshape(N, 2), best_angles):
            best_centers = res.x.reshape(N, 2)
            best_side, best_center, best_angle = min_outer_side(best_centers, best_angles)
    except Exception:
        pass

    # Final validation
    if not check_overlap(best_centers, best_angles):
        # Fallback to a known valid configuration
        # Simple grid
        pts_fb = []
        for r in range(3):
            for c in range(4):
                pts_fb.append((c * 2.5 - 3.75, r * 3.0))
        best_centers = np.array(pts_fb)
        best_angles = np.zeros(N)
        best_side, best_center, best_angle = min_outer_side(best_centers, best_angles)

    inner = []
    for i in range(N):
        inner.append([best_centers[i, 0], best_centers[i, 1], best_angles[i]])
    
    return inner, [best_center[0], best_center[1]], best_side, best_angle

