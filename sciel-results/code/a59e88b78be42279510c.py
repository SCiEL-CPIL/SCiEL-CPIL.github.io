"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _support_hexagon(cx, cy, s, angle_deg, direction):
    """Support function of a regular hexagon in a given direction (unit vector)."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    vx = s * np.cos(angles)
    vy = s * np.sin(angles)
    proj = vx * direction[0] + vy * direction[1]
    return cx * direction[0] + cy * direction[1] + np.max(proj)

def _hex_separated(p1, p2, tol=1e-9):
    """Check if two unit hexagons are separated (non-overlapping) using SAT."""
    x1, y1, a1 = p1
    x2, y2, a2 = p2
    
    # Vertices relative to centers
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    v1_rel = np.column_stack([np.cos(base_angles + np.radians(a1)), 
                               np.sin(base_angles + np.radians(a1))])
    v2_rel = np.column_stack([np.cos(base_angles + np.radians(a2)), 
                               np.sin(base_angles + np.radians(a2))])
    
    v1 = v1_rel + np.array([x1, y1])
    v2 = v2_rel + np.array([x2, y2])
    
    # Edge normals from both polygons (12 axes total)
    for v in [v1, v2]:
        for i in range(6):
            ex = v[(i+1)%6, 0] - v[i, 0]
            ey = v[(i+1)%6, 1] - v[i, 1]
            length = np.sqrt(ex*ex + ey*ey)
            if length < 1e-12:
                continue
            nx = -ey / length
            ny = ex / length
            
            proj1 = v1 @ np.array([nx, ny])
            proj2 = v2 @ np.array([nx, ny])
            
            if np.max(proj1) < np.min(proj2) + tol or np.max(proj2) < np.min(proj1) + tol:
                return True  # Separated
    
    return False  # Overlapping

def _polygons_overlap(p1, p2):
    """Check if two convex polygons overlap using SAT."""
    return not _hex_separated(p1, p2)

def _compute_min_side(inner, outer_center, outer_angle):
    """Compute minimum outer side length for given inner config and outer center/angle."""
    cx, cy = outer_center
    theta = np.radians(outer_angle)
    face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
    normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
    
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    
    min_s = 0.0
    for ix, iy, ia in inner:
        ia_rad = np.radians(ia)
        # Inner hex vertices relative to its center
        v_rel = np.column_stack([np.cos(base_angles + ia_rad), 
                                  np.sin(base_angles + ia_rad)])
        # Inner hex vertices in absolute coords
        v_abs = v_rel + np.array([ix, iy])
        
        for k in range(6):
            # (v_abs - center) . n_k <= s * sqrt(3)/2
            dots = (v_abs[:, 0] - cx) * normals[k, 0] + (v_abs[:, 1] - cy) * normals[k, 1]
            max_dot = np.max(dots)
            if max_dot > 0:
                s_needed = max_dot / (np.sqrt(3) / 2)
                min_s = max(min_s, s_needed)
    
    return min_s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin (10s buffer)
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    def make_hex_lattice(rows, cols, rotation_deg=0.0):
        """Create hexagonal lattice of unit hexagons."""
        spacing = 2 * np.sqrt(3)
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * spacing
                y = r * spacing * 1.5
                if r % 2 == 1:
                    x += spacing / 2
                pts.append([x, y, rotation_deg])
        return pts
    
    def center_config(config):
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx0 = (min(xs) + max(xs)) / 2
        cy0 = (min(ys) + max(ys)) / 2
        return [[p[0] - cx0, p[1] - cy0, p[2]] for p in config]
    
    def is_valid_config(inner_pts, outer_angle):
        """Check non-overlap for inner hexagons."""
        for i in range(N):
            for j in range(i+1, N):
                if not _hex_separated(inner_pts[i], inner_pts[j]):
                    return False
        return True
    
    def decode(x):
        """Decode parameter vector to inner points and outer angle."""
        inner_pts = []
        for i in range(N):
            ix = x[3*i]
            iy = x[3*i+1]
            ia = np.degrees(x[3*i+2])
            inner_pts.append([ix, iy, ia])
        outer_angle = np.degrees(x[3*N])
        return inner_pts, outer_angle
    
    # Generate diverse initial configurations
    configs = []
    configs.append(center_config(make_hex_lattice(3, 4)))
    configs.append(center_config(make_hex_lattice(4, 3)))
    configs.append(center_config(make_hex_lattice(2, 6)))
    configs.append(center_config(make_hex_lattice(6, 2)))
    
    # Add some perturbed configs for diversity
    np.random.seed(42)
    for base_config in configs[:2]:
        perturbed = []
        for p in base_config:
            dx = np.random.uniform(-0.5, 0.5)
            dy = np.random.uniform(-0.5, 0.5)
            da = np.random.uniform(-15, 15)
            perturbed.append([p[0] + dx, p[1] + dy, p[2] + da])
        configs.append(center_config(perturbed))
    
    # For each config, run optimization
    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
        
        # Initial outer side estimate
        s0 = _compute_min_side(config, (0.0, 0.0), 0.0)
        
        # Nelder-Mead optimization
        n_vars = 3 * N + 1
        x0 = []
        for p in config:
            x0.extend([p[0], p[1], np.radians(p[2])])
        x0.append(0.0)
        
        def objective_nm(x):
            if time.time() > deadline:
                return 1e10
            inner_pts, outer_angle = decode(x)
            
            # Check non-overlap with penalty
            for i in range(N):
                for j in range(i+1, N):
                    if not _hex_separated(inner_pts[i], inner_pts[j]):
                        # Compute overlap penalty
                        return 1e8 + s0 + (i * N + j) * 1e-3
            
            s = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
            return s
        
        try:
            result = minimize(objective_nm, x0, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7,
                                   'adaptive': True})
            
            if result.fun < best_score and result.fun < 1e7:
                inner_pts, outer_angle = decode(result.x)
                s_final = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
                
                if is_valid_config(inner_pts, outer_angle) and s_final < best_score:
                    best_score = s_final
                    best_inner = [list(p) for p in inner_pts]
                    best_center = (0.0, 0.0)
                    best_side = s_final
                    best_angle = outer_angle
                    
        except Exception:
            pass
        
        # Polish with SLSQP if we have a good starting point
        if best_inner is not None and time.time() < deadline - 15.0:
            x_polish = []
            for p in best_inner:
                x_polish.extend([p[0], p[1], np.radians(p[2])])
            x_polish.append(np.radians(best_angle))
            
            def objective_slsqp(x):
                if time.time() > deadline:
                    return 1e10
                inner_pts, outer_angle = decode(x)
                
                # Non-overlap penalty
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        if not _hex_separated(inner_pts[i], inner_pts[j]):
                            penalty += 1e5
                
                s = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
                return s + penalty
            
            try:
                result2 = minimize(objective_slsqp, x_polish, method='SLSQP',
                                 options={'maxiter': 500, 'ftol': 1e-9})
                
                if result2.fun < best_score and result2.fun < 1e7:
                    inner_pts, outer_angle = decode(result2.x)
                    s_final = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
                    
                    if is_valid_config(inner_pts, outer_angle) and s_final < best_score:
                        best_score = s_final
                        best_inner = [list(p) for p in inner_pts]
                        best_center = (0.0, 0.0)
                        best_side = s_final
                        best_angle = outer_angle
                        
            except Exception:
                pass
    
    # Final polish: optimize outer angle and center with current best inner config
    if best_inner is not None and time.time() < deadline - 5.0:
        def outer_objective(x):
            ox, oy, oangle = x
            inner_pts = best_inner
            s = _compute_min_side(inner_pts, (ox, oy), np.degrees(oangle))
            return s
        
        x0_outer = [0.0, 0.0, np.radians(best_angle)]
        try:
            result_outer = minimize(outer_objective, x0_outer, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result_outer.fun < best_score:
                ox, oy, oangle = result_outer.x
                s_final = _compute_min_side(best_inner, (ox, oy), np.degrees(oangle))
                
                if s_final < best_score:
                    best_score = s_final
                    best_center = (ox, oy)
                    best_side = s_final
                    best_angle = np.degrees(oangle)
        except Exception:
            pass
    
    # If we found something, return it
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback: return the seed
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
