import math
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 12
SQRT3 = math.sqrt(3)


def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin
    
    # Precompute unit hexagon vertices (flat-top, side length 1)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])
    
    # Precompute rotation matrices for discrete angles
    def get_rotation(angle_deg):
        theta = math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    def hex_vertices(cx, cy, side, angle_deg):
        R = get_rotation(angle_deg)
        verts = (unit_verts @ R.T) * side
        return verts + np.array([cx, cy])
    
    def min_outer_side(centers, angles, outer_angle_deg):
        """Compute minimum outer side length and center for given inner config."""
        theta = math.radians(outer_angle_deg)
        # Normals of outer hexagon (perpendicular to edges)
        normals = np.array([
            [math.cos(theta + math.pi/6 + k * math.pi/3),
             math.sin(theta + math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])
        
        all_verts = []
        for i in range(N):
            v = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        projs = all_verts @ normals.T
        min_p = np.min(projs, axis=0)
        max_p = np.max(projs, axis=0)
        widths = max_p - min_p
        
        max_width = np.max(widths)
        side = max_width / SQRT3
        
        # Find center via least squares
        targets = (max_p + min_p) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals, targets, rcond=None)
        
        # Verify and adjust if needed
        shifted = all_verts - c
        shifted_projs = shifted @ normals.T
        A = side * SQRT3 / 2
        if np.max(np.abs(shifted_projs)) > A + 1e-9:
            # Increase side to fit
            actual_max = np.max(np.abs(shifted_projs))
            side = actual_max * 2 / SQRT3 + 1e-9
        
        return side, c
    
    def check_overlap(centers, angles, eps=1e-9):
        """Check for any overlap between inner hexagons."""
        polys = [hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
        
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
    
    def objective(params):
        """
        Objective for SLSQP: minimize outer side length.
        params: [x0, y0, x1, y1, ..., x11, y11]
        """
        if time.time() > deadline:
            return 1e10
        
        centers = np.array(params).reshape(N, 2)
        angles = np.zeros(N)
        
        # Try several outer angles to find best
        best_side = float('inf')
        for oa in [0.0, 30.0, 15.0, 45.0]:
            s, _ = min_outer_side(centers, angles, oa)
            if s < best_side:
                best_side = s
        
        return best_side
    
    def constraint_func(params):
        """
        Non-overlap constraints: for each pair (i,j), distance constraint.
        We use a simplified distance-based constraint: the minimum distance
        between hexagons must be >= 0. For regular hexagons of side 1,
        if centers are at distance d and aligned, they overlap if d < 2.
        With rotations, it's more complex, but we use a conservative bound.
        """
        centers = np.array(params).reshape(N, 2)
        constraints = []
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # For unit hexagons, minimum center distance to avoid overlap
                # depends on relative rotation. Conservative: dist >= 2.0
                # But this is too restrictive. Let's use a softer approach.
                # Actually, for the SLSQP to work, we need smooth constraints.
                # Let's use: dist^2 >= (2.0 - eps)^2 as a soft constraint
                # This is conservative but smooth.
                constraints.append(dist - 1.9)  # Allow slight overlap, fix later
        
        return np.array(constraints)
    
    # --- Generate multiple initial configurations ---
    
    configs = []
    
    # Config 1: Hexagonal lattice (7-point cluster + 5 more)
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = SQRT3 * (i + j / 2.0)
            y = 3.0 * j
            lattice.append((x, y))
    lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
    init1 = np.array(lattice[:12])
    init1 -= np.mean(init1, axis=0)
    configs.append(init1.flatten().tolist())
    
    # Config 2: Two concentric rings (6 inner + 6 outer)
    pts2 = []
    # Inner ring: 6 hexagons at distance 2
    for k in range(6):
        a = math.radians(60 * k)
        pts2.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    # Outer ring: 6 hexagons at distance ~4
    for k in range(6):
        a = math.radians(60 * k + 30)
        pts2.append((4.0 * math.cos(a), 4.0 * math.sin(a)))
    init2 = np.array(pts2)
    init2 -= np.mean(init2, axis=0)
    configs.append(init2.flatten().tolist())
    
    # Config 3: 4-4-4 rows
    pts3 = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 - 3.0
            y = r * 3.0
            if r % 2 == 1:
                x += 1.0
            pts3.append((x, y))
    init3 = np.array(pts3)
    init3 -= np.mean(init3, axis=0)
    configs.append(init3.flatten().tolist())
    
    # Config 4: 3-4-3-2 rows
    row_widths = [3, 4, 3, 2]
    pts4 = []
    for r, w in enumerate(row_widths):
        x_start = -(w - 1) * 2.0 / 2.0
        for k in range(w):
            x = x_start + k * 2.0
            if r % 2 == 1:
                x += 1.0
            pts4.append((x, r * 3.0))
    init4 = np.array(pts4)
    init4 -= np.mean(init4, axis=0)
    configs.append(init4.flatten().tolist())
    
    # Config 5: Compact cluster
    pts5 = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 - 3.0
            y = r * 2.0 * SQRT3 / 2.0
            if r % 2 == 1:
                x += 1.0
            pts5.append((x, y))
    init5 = np.array(pts5)
    init5 -= np.mean(init5, axis=0)
    configs.append(init5.flatten().tolist())
    
    # Evaluate all configs
    best_params = None
    best_obj = float('inf')
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg
    
    # --- Local optimization using SLSQP ---
    # SLSQP with distance constraints
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            x0 = np.array(best_params)
        else:
            x0 = np.array(best_params) + rng.normal(0, 0.3, size=2*N)
        
        try:
            # Use SLSQP with inequality constraints
            # For simplicity, we'll use a penalty approach within SLSQP
            # since true non-overlap constraints are non-smooth
            
            def penalized_objective(params):
                if time.time() > deadline:
                    return 1e10
                
                centers = np.array(params).reshape(N, 2)
                angles = np.zeros(N)
                
                # Overlap penalty
                overlap = 0.0
                polys = [hex_vertices(centers[i][0], centers[i][1], 1.0, 0.0) for i in range(N)]
                for i in range(N):
                    for j in range(i + 1, N):
                        v1, v2 = polys[i], polys[j]
                        if (np.max(v1[:, 0]) < np.min(v2[:, 0]) - 1e-9 or
                            np.max(v2[:, 0]) < np.min(v1[:, 0]) - 1e-9 or
                            np.max(v1[:, 1]) < np.min(v2[:, 1]) - 1e-9 or
                            np.max(v2[:, 1]) < np.min(v1[:, 1]) - 1e-9):
                            continue
                        if _any_point_in_poly(v1, v2, 1e-9) or _any_point_in_poly(v2, v1, 1e-9):
                            overlap += 5.0
                        elif _edges_intersect(v1, v2, 1e-9):
                            overlap += 5.0
                    if overlap > 30:
                        break
                
                if overlap > 0:
                    return overlap + 1.0
                
                best_side = float('inf')
                for oa in [0.0, 30.0, 15.0, 45.0]:
                    s, _ = min_outer_side(centers, angles, oa)
                    if s < best_side:
                        best_side = s
                
                return best_side
            
            res = minimize(
                penalized_objective,
                x0,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass
    
    # --- Final refinement with Nelder-Mead for better local search ---
    if time.time() < deadline - 10:
        try:
            def final_objective(params):
                if time.time() > deadline:
                    return 1e10
                
                centers = np.array(params).reshape(N, 2)
                angles = np.zeros(N)
                
                overlap = 0.0
                polys = [hex_vertices(centers[i][0], centers[i][1], 1.0, 0.0) for i in range(N)]
                for i in range(N):
                    for j in range(i + 1, N):
                        v1, v2 = polys[i], polys[j]
                        if (np.max(v1[:, 0]) < np.min(v2[:, 0]) - 1e-9 or
                            np.max(v2[:, 0]) < np.min(v1[:, 0]) - 1e-9 or
                            np.max(v1[:, 1]) < np.min(v2[:, 1]) - 1e-9 or
                            np.max(v2[:, 1]) < np.min(v1[:, 1]) - 1e-9):
                            continue
                        if _any_point_in_poly(v1, v2, 1e-9) or _any_point_in_poly(v2, v1, 1e-9):
                            overlap += 10.0
                        elif _edges_intersect(v1, v2, 1e-9):
                            overlap += 10.0
                    if overlap > 50:
                        break
                
                if overlap > 0:
                    return overlap + 1.0
                
                best_side = float('inf')
                for oa in [0.0, 30.0, 15.0, 45.0]:
                    s, _ = min_outer_side(centers, angles, oa)
                    if s < best_side:
                        best_side = s
                
                return best_side
            
            res2 = minimize(
                final_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # --- Final result ---
    final_centers = np.array(best_params).reshape(N, 2)
    final_angles = np.zeros(N)
    
    # Validate
    if not check_overlap(final_centers, final_angles):
        # Try to fix by using a known valid config
        for cfg in configs:
            c = np.array(cfg).reshape(N, 2)
            if check_overlap(c, final_angles):
                final_centers = c
                break
        else:
            # Last resort: simple grid
            pts_fb = []
            for r in range(3):
                for c in range(4):
                    pts_fb.append((c * 2.5 - 3.75, r * 3.0))
            final_centers = np.array(pts_fb)
    
    # Compute final side with best outer angle
    best_side = float('inf')
    best_outer_angle = 0.0
    best_center = np.zeros(2)
    for oa in np.linspace(0, 60, 61):
        s, c = min_outer_side(final_centers, final_angles, oa)
        if s < best_side:
            best_side = s
            best_outer_angle = oa
            best_center = c
    
    inner = []
    for i in range(N):
        inner.append([final_centers[i][0], final_centers[i][1], 0.0])
    
    return inner, [best_center[0], best_center[1]], best_side, best_outer_angle
