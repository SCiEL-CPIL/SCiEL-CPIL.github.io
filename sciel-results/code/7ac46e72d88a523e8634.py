import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a valid flower packing with face-to-face contact geometry.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Valid flower packing:
    # 1 center at (0,0)
    # 6 at distance 2.0 (face-to-face contact for same-orientation hexagons)
    # 4 at distance 4.0 in the gaps
    centers = np.array([[0.0, 0.0]])
    
    r1 = 2.0
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 4.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    def compute_outer_side_exact(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        Uses support function: max projection onto face normals.
        """
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            max_proj = max(max_proj, np.max(dots))
        
        # For a regular hexagon with side S, apothem = S * sqrt(3) / 2
        # S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)
    
    def _hex_vertices(cx, cy, angle_rad):
        """Vertices of a unit hexagon centered at (cx,cy) rotated by angle_rad."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return base_vertices @ rot.T + np.array([cx, cy])
    
    def _point_in_hex(pt, center, angle_rad):
        """Check if point is inside or on boundary of a unit hexagon."""
        local_pt = pt - center
        c, s = np.cos(-angle_rad), np.sin(-angle_rad)
        rot_inv = np.array([[c, -s], [s, c]])
        local_pt = rot_inv @ local_pt
        
        for k in range(6):
            a1 = np.deg2rad(60 * k)
            a2 = np.deg2rad(60 * (k + 1))
            v1 = np.array([np.cos(a1), np.sin(a1)])
            v2 = np.array([np.cos(a2), np.sin(a2)])
            edge = v2 - v1
            normal = np.array([edge[1], -edge[0]])
            if np.dot(normal, local_pt - v1) > 1e-9:
                return False
        return True
    
    def _seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments p1-p2 and p3-p4."""
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        p3 = np.array(p3, dtype=float)
        p4 = np.array(p4, dtype=float)
        
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(p1 - p3)
        
        if a < 1e-12:
            t = f / e
            t = max(0, min(1, t))
            d = p1 - (p3 + t * d2)
            return np.linalg.norm(d)
        
        c = np.dot(d1, r)
        if e < 1e-12:
            s = c / a
            s = max(0, min(1, s))
            d = (p1 + s * d1) - p3
            return np.linalg.norm(d)
        
        b = np.dot(d1, d2)
        denom = a * e - b * b
        s = 0
        t = 0
        
        if denom > 1e-12:
            s = (b * f - c * e) / denom
            s = max(0, min(1, s))
            t = (b * s + f) / e
            if t < 0:
                t = 0
                s = max(0, min(1, c / a))
            elif t > 1:
                t = 1
                s = max(0, min(1, (b + f) / a))
            d = (p1 + s * d1) - (p3 + t * d2)
            return np.linalg.norm(d)
        else:
            t = f / e
            t = max(0, min(1, t))
            d = (p1 + t * d2) - p3
            return np.linalg.norm(d)
    
    def _overlap_penalty(phis, weight=1000.0):
        """
        Compute penalty for overlapping inner hexagons.
        Uses vertex-in-polygon and edge distance checks.
        """
        penalty = 0.0
        verts_list = []
        for i in range(n_hex):
            verts_list.append(_hex_vertices(centers[i][0], centers[i][1], phis[i]))
        
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                dc = np.linalg.norm(centers[i] - centers[j])
                if dc > 4.0 + 0.1:
                    continue
                
                overlap = False
                for v in v1:
                    if _point_in_hex(v, centers[j], phis[j]):
                        overlap = True
                        break
                if not overlap:
                    for v in v2:
                        if _point_in_hex(v, centers[i], phis[i]):
                            overlap = True
                            break
                
                if overlap:
                    penalty += weight
                else:
                    min_dist = float('inf')
                    for a in range(6):
                        p1 = v1[a]
                        p2 = v1[(a + 1) % 6]
                        for b in range(6):
                            p3 = v2[b]
                            p4 = v2[(b + 1) % 6]
                            d = _seg_seg_dist(p1, p2, p3, p4)
                            if d < min_dist:
                                min_dist = d
                    
                    if min_dist < 0.001:
                        penalty += weight * (0.001 - min_dist) ** 2
        
        return penalty
    
    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        
        side = compute_outer_side_exact(theta_out, phis)
        overlap_pen = _overlap_penalty(phis, weight=500.0)
        return side + overlap_pen
    
    # Initial guess: all hexagons aligned at 0 degrees, outer at 30 degrees
    x0 = np.zeros(n_hex + 1)
    x0[0] = np.deg2rad(30)
    
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Multi-start Powell optimization
    n_starts = 6
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            # All at 30 degrees
            x_start = np.full(n_hex + 1, np.deg2rad(30))
        elif i == 2:
            # Random perturbations
            x_start = x0 + np.random.uniform(-0.1, 0.1, n_hex + 1)
            x_start[0] = np.random.uniform(0, np.pi/3)
        elif i == 3:
            # Try 60-degree offset for inner
            x_start = x0.copy()
            x_start[1:] += np.deg2rad(60)
        else:
            x_start = np.random.uniform(-0.2, 0.2, n_hex + 1)
            x_start[0] = np.random.uniform(0, np.pi/3)
            
        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-10, 'xtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return _fallback_solution()
    
    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 8:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 2000, 'disp': False, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final refinement with exact objective only
    def objective_exact(params):
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side_exact(theta_out, phis)
    
    if time.time() < deadline - 3:
        try:
            res3 = minimize(objective_exact, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 1500, 'disp': False})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    # Verify no overlaps
    overlap_pen = _overlap_penalty(phis, weight=1.0)
    if overlap_pen > 0.01:
        return _fallback_solution()
    
    side = compute_outer_side_exact(theta_out, phis)
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    return inner, (0.0, 0.0), side + 1e-9, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: valid flower packing with all hexagons at 30 degrees.
    """
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = 2.0
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 4.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Compute exact side for this configuration
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    theta_out = np.deg2rad(30)
    normals = np.array([
        [np.cos(np.deg2rad(30 + k * 60) + theta_out),
         np.sin(np.deg2rad(30 + k * 60) + theta_out)]
        for k in range(6)
    ])
    
    max_proj = 0.0
    for i, (cx, cy) in enumerate(centers):
        phi = np.deg2rad(30)
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        verts = base_vertices @ rot.T + np.array([cx, cy])
        dots = verts @ normals.T
        max_proj = max(max_proj, np.max(dots))
    
    side = 2.0 * max_proj / np.sqrt(3)
    
    return inner, (0.0, 0.0), side + 1e-9, 30.0
