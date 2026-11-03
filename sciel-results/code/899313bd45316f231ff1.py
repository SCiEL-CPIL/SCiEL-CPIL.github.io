"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns penetration depth if overlap, 0 otherwise."""
    # v1, v2 are (6, 2) arrays of vertices
    # We need to check separation along all edge normals of both polygons.
    min_penetration = 0.0
    
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            edge = p2 - p1
            # Normal to the edge
            nx = -edge[1]
            ny = edge[0]
            length = np.sqrt(nx*nx + ny*ny)
            if length < 1e-12:
                continue
            nx /= length
            ny /= length
            
            # Project both polygons onto the normal
            proj1 = v1 @ np.array([nx, ny])
            proj2 = v2 @ np.array([nx, ny])
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            # If max1 < min2 or max2 < min1, they are separated.
            # The penetration depth is the overlap of the intervals.
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                if overlap > min_penetration:
                    min_penetration = overlap
            else:
                # Separated in this direction, so no overlap in this axis.
                # But we need to check ALL axes. If ANY axis separates, no overlap.
                # Wait, SAT says: if there is ANY separating axis, polygons are disjoint.
                # So if overlap <= 0 for ANY axis, return 0.
                return 0.0

    return min_penetration

def _compute_min_side(inner_pts, outer_angle_deg):
    """
    Compute minimum outer side length for given inner config.
    Outer hexagon is centered at (0,0).
    inner_pts: list of (x, y, angle_deg)
    """
    theta = np.radians(outer_angle_deg)
    # Normals of the outer hexagon faces
    # The faces are at angles 30, 90, 150, 210, 270, 330 + theta
    face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
    normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
    
    min_s = 0.0
    for ix, iy, ia in inner_pts:
        v = _hex_vertices(ix, iy, 1.0, ia)
        for k in range(6):
            n = normals[k]
            # Max projection of inner hex vertices onto normal
            dots = v @ n
            max_dot = np.max(dots)
            # The constraint is max_dot <= s * apothem_outer
            # Apothem of outer hexagon with side s is s * sqrt(3) / 2
            # So s >= max_dot / (sqrt(3)/2)
            if max_dot > 0:
                s_needed = max_dot / (np.sqrt(3) / 2)
                if s_needed > min_s:
                    min_s = s_needed
    
    return min_s

def _objective(x, deadline):
    """
    Objective function for optimization.
    x: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
    """
    if time.time() > deadline:
        return 1e10

    inner_pts = []
    for i in range(N):
        ix = x[3*i]
        iy = x[3*i+1]
        ia = x[3*i+2] # radians
        inner_pts.append((ix, iy, np.degrees(ia)))
    
    outer_angle = np.degrees(x[3*N])
    
    # Calculate side length required to contain all
    s = _compute_min_side(inner_pts, outer_angle)
    
    # Calculate overlap penalty
    penalty = 0.0
    # Pre-compute vertices for all inner hexagons to reuse
    verts = []
    for ix, iy, ia_deg in inner_pts:
        v = _hex_vertices(ix, iy, 1.0, ia_deg)
        verts.append(v)
        
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            dx = inner_pts[i][0] - inner_pts[j][0]
            dy = inner_pts[i][1] - inner_pts[j][1]
            dist_sq = dx*dx + dy*dy
            # Max possible distance for overlap is 2 * circumradius = 2.0? 
            # Actually, if rotated, they can interlock further.
            # Max distance for potential overlap is 2 * 1.0 + 2 * 1.0 = 4.0? 
            # Center to center distance must be < 2 * inradius + 2 * inradius?
            # Inradius is sqrt(3)/2 ~ 0.866. Sum ~ 1.732.
            # If aligned, min dist is 2*sqrt(3) ~ 3.46.
            # So check dist < 4.0 is safe.
            if dist_sq < 16.0:
                pen = _polygons_overlap(verts[i], verts[j])
                penalty += pen * pen
    
    return s + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s buffer
    
    best_score = float('inf')
    best_inner = None
    best_angle = 0.0
    
    def decode(x):
        inner_pts = []
        for i in range(N):
            ix = x[3*i]
            iy = x[3*i+1]
            ia = np.degrees(x[3*i+2])
            inner_pts.append([ix, iy, ia])
        outer_angle = np.degrees(x[3*N])
        return inner_pts, outer_angle

    # Generate initial configurations
    
    # 1. Honeycomb 1+6+5
    def make_honeycomb_1_6_5():
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        # Ring 1
        for k in range(6):
            angle = k * 60
            rad = np.radians(angle)
            x = 2 * np.sqrt(3) * np.cos(rad)
            y = 2 * np.sqrt(3) * np.sin(rad)
            pts.append([x, y, 0.0])
        # Ring 2 (5 of them)
        # Positions in ring 2 are at distance 4*sqrt(3) from center?
        # Or adjacent to ring 1.
        # Let's just place them in the gaps or outside.
        # A standard hex lattice step is dx=sqrt(3), dy=1.5?
        # Let's use lattice coordinates.
        # Center (0,0)
        # Ring 1: (1,0), (1,1), (0,1), (-1,1), (-1,0), (0,-1) in axial?
        # Let's stick to Cartesian for simplicity in initialization.
        # Add 5 more around the ring 1.
        # Positions: 
        # Between Ring 1 hexes, further out.
        for k in range(5):
             angle = k * 60 + 30 # offset
             rad = np.radians(angle)
             # Distance from center to ring 2 centers is 2 * (2*sqrt(3)) = 4*sqrt(3)? No.
             # Distance between adjacent hex centers is 2*sqrt(3).
             # Ring 1 centers are at dist 2*sqrt(3).
             # Ring 2 centers adjacent to Ring 1 are at dist 4*sqrt(3) from center?
             # Let's just perturb the ring 1 points outward.
             x = 4 * np.sqrt(3) * np.cos(rad)
             y = 4 * np.sqrt(3) * np.sin(rad)
             pts.append([x, y, 0.0])
        return pts

    # 2. 2x6 Grid (Rectangular block)
    def make_grid_2x6():
        pts = []
        dx = 2 * np.sqrt(3)
        dy = 3.0 # Vertical distance between rows in hex lattice is 3 * inradius? 
        # Inradius = sqrt(3)/2.
        # Row offset y = 3 * sqrt(3)/2 = 1.5 * sqrt(3) ~ 2.598
        dy = 1.5 * np.sqrt(3)
        for r in range(2):
            for c in range(6):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2
                pts.append([x, y, 0.0])
        # Center the grid
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pts:
            p[0] -= cx
            p[1] -= cy
        return pts

    # 3. 3x4 Grid
    def make_grid_3x4():
        pts = []
        dx = 2 * np.sqrt(3)
        dy = 1.5 * np.sqrt(3)
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2
                pts.append([x, y, 0.0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pts:
            p[0] -= cx
            p[1] -= cy
        return pts

    # 4. 4x3 Grid (Rotated version of 3x4 essentially)
    def make_grid_4x3():
        pts = []
        dx = 2 * np.sqrt(3)
        dy = 1.5 * np.sqrt(3)
        for r in range(4):
            for c in range(3):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2
                pts.append([x, y, 0.0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pts:
            p[0] -= cx
            p[1] -= cy
        return pts

    initial_configs = [
        make_honeycomb_1_6_5(),
        make_grid_2x6(),
        make_grid_3x4(),
        make_grid_4x3()
    ]
    
    # Add random perturbations to break symmetry and explore local minima
    np.random.seed(42)
    for i in range(4):
        base = initial_configs[i % len(initial_configs)]
        perturbed = []
        for p in base:
            dx = np.random.uniform(-0.2, 0.2)
            dy = np.random.uniform(-0.2, 0.2)
            da = np.random.uniform(-10, 10)
            perturbed.append([p[0] + dx, p[1] + dy, p[2] + da])
        initial_configs.append(perturbed)

    for config in initial_configs:
        if time.time() > deadline - 10.0:
            break
        
        # Convert to optimization vector
        x0 = []
        for p in config:
            x0.extend([p[0], p[1], np.radians(p[2])])
        x0.append(0.0) # outer angle
        
        # Run optimization
        # Use Nelder-Mead for robustness with non-smooth penalties initially
        # Then SLSQP for refinement
        
        def obj(x):
            return _objective(x, deadline)
        
        try:
            # Phase 1: Nelder-Mead
            res1 = minimize(obj, x0, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            current_best = res1.fun
            current_x = res1.x
            
            # Phase 2: SLSQP for local refinement
            # SLSQP can handle the smooth parts better if we are close
            # But the penalty function is non-smooth (max(0, pen)^2).
            # Let's try COBYLA which is derivative-free and handles constraints/penalties well?
            # Or just run NM again with tighter tolerances.
            
            res2 = minimize(obj, current_x, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res2.fun < current_best:
                current_best = res2.fun
                current_x = res2.x
            
            # Extract and validate
            inner_pts, outer_angle = decode(current_x)
            
            # Final check for validity (no overlaps)
            valid = True
            verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner_pts]
            for i in range(N):
                for j in range(i+1, N):
                    if _polygons_overlap(verts[i], verts[j], tol=1e-7) > 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                s_final = _compute_min_side(inner_pts, outer_angle)
                if s_final < best_score:
                    best_score = s_final
                    best_inner = inner_pts
                    best_angle = outer_angle
                    
        except Exception:
            pass

    # If we have a best inner, try to optimize outer angle and center slightly
    # The problem statement says "outer hexagon center (x, y)".
    # My objective assumed center (0,0).
    # We can shift the inner points to center them, or optimize the outer center.
    # Shifting inner points is equivalent to shifting outer center in relative terms.
    # Since the objective is translation invariant for the "containment" part if we shift both,
    # but the "overlap" part depends on relative positions.
    # So optimizing outer center (ox, oy) is equivalent to shifting all inner points by (-ox, -oy).
    # We can just re-center the best_inner found.
    
    if best_inner is not None:
        # Re-center inner points to (0,0) to match the assumption in _compute_min_side
        xs = [p[0] for p in best_inner]
        ys = [p[1] for p in best_inner]
        cx = sum(xs) / N
        cy = sum(ys) / N
        for p in best_inner:
            p[0] -= cx
            p[1] -= cy
        
        # Now optimize outer angle with centered points
        def obj_angle(x):
            angle = x[0]
            return _compute_min_side(best_inner, np.degrees(angle))
        
        try:
            res_ang = minimize(obj_angle, [np.radians(best_angle)], method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res_ang.fun < best_score:
                best_angle = np.degrees(res_ang.x[0])
                best_score = res_ang.fun
        except Exception:
            pass

    if best_inner is not None:
        return best_inner, (0.0, 0.0), best_score, best_angle
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}")
