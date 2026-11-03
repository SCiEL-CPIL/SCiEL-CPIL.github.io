"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        inradius = side * math.sqrt(3) / 2.0
        
        for i in range(6):
            normal_angle = math.radians(30 + i * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            dist = lx * nx + ly * ny
            if dist > inradius + 1e-9:
                return False
        return True
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    def hexagons_overlap(cx1, cy1, a1, cx2, cy2, a2):
        """Check if two unit hexagons overlap. Returns True if they overlap."""
        dx = cx1 - cx2
        dy = cy1 - cy2
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 2.0 + 1e-9:
            return False
        if dist < 1e-6:
            return True
        
        verts1 = get_hex_vertices(cx1, cy1, 1.0, a1)
        verts2 = get_hex_vertices(cx2, cy2, 1.0, a2)
        
        for v in verts1:
            if point_in_hexagon(v[0], v[1], cx2, cy2, 1.0, a2):
                return True
        for v in verts2:
            if point_in_hexagon(v[0], v[1], cx1, cy1, 1.0, a1):
                return True
        
        for i in range(6):
            e1 = (verts1[i], verts1[(i+1)%6])
            for j in range(6):
                e2 = (verts2[j], verts2[(j+1)%6])
                if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                    return True
        return False
    
    def required_outer_side(centers, angles, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_side = 0.0
        outer_angle_rad = math.radians(outer_angle_deg)
        
        for i in range(len(centers)):
            cx, cy = centers[i]
            a = angles[i]
            verts = get_hex_vertices(cx, cy, 1.0, a)
            for (vx, vy) in verts:
                r = math.sqrt(vx*vx + vy*vy)
                theta = math.atan2(vy, vx)
                
                best_ratio = float('inf')
                for k in range(6):
                    normal_angle = outer_angle_rad + math.radians(30 + k * 60)
                    delta = theta - normal_angle
                    delta = ((delta + math.pi) % (2*math.pi)) - math.pi
                    if abs(delta) < math.pi/2 - 1e-9:
                        ratio = 1.0 / math.cos(delta)
                        if ratio < best_ratio:
                            best_ratio = ratio
                
                if best_ratio < float('inf'):
                    required_S = r / (math.sqrt(3)/2.0 * best_ratio)
                    if required_S > max_side:
                        max_side = required_S
        
        return max_side
    
    def make_initial_config():
        """Create a symmetric 3-4-3 configuration."""
        row_spacing = math.sqrt(3)
        col_spacing = 2.0
        
        centers = []
        angles = []
        
        # Row 0 (top): 3 hexagons
        y0 = row_spacing
        for i in range(3):
            x = (i - 1) * col_spacing
            centers.append((x, y0))
            angles.append(0.0)
        
        # Row 1 (middle): 4 hexagons
        y1 = 0.0
        for i in range(4):
            x = (i - 1.5) * col_spacing
            centers.append((x, y1))
            angles.append(0.0)
        
        # Row 2 (bottom): 3 hexagons
        y2 = -row_spacing
        for i in range(3):
            x = (i - 1) * col_spacing
            centers.append((x, y2))
            angles.append(0.0)
        
        return centers, angles
    
    def flatten(centers, angles):
        params = []
        for (cx, cy), a in zip(centers, angles):
            params.extend([cx, cy, a])
        return params
    
    def unflatten(params):
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        return centers, angles
    
    def obj_func(params):
        if time.time() > deadline:
            return 1e10
        
        centers, angles = unflatten(params)
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(centers[i][0], centers[i][1], angles[i],
                                     centers[j][0], centers[j][1], angles[j]):
                    return 1e10
        
        outer_side = required_outer_side(centers, angles, 0.0)
        return outer_side
    
    centers0, angles0 = make_initial_config()
    x0 = flatten(centers0, angles0)
    
    # Initial evaluation
    initial_side = required_outer_side(centers0, angles0, 0.0)
    best_side = initial_side
    best_params = x0.copy()
    
    # Verify initial config has no overlaps
    valid_initial = True
    for i in range(N):
        for j in range(i+1, N):
            if hexagons_overlap(centers0[i][0], centers0[i][1], angles0[i],
                                 centers0[j][0], centers0[j][1], angles0[j]):
                valid_initial = False
                break
        if not valid_initial:
            break
    
    if not valid_initial:
        # Adjust spacing
        centers_adj, angles_adj = make_initial_config()
        # Scale up
        for i in range(len(centers_adj)):
            centers_adj[i] = (centers_adj[i][0] * 1.1, centers_adj[i][1] * 1.1)
        x0 = flatten(centers_adj, angles_adj)
        best_params = x0.copy()
        best_side = required_outer_side(centers_adj, angles_adj, 0.0)
    
    # Try differential evolution
    bounds = [(-6, 6)] * (3 * N)
    
    try:
        result = differential_evolution(
            obj_func,
            bounds,
            maxiter=30,
            popsize=12,
            tol=0.05,
            seed=42,
            polish=True,
            time_limit=80
        )
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    except Exception as e:
        pass
    
    # Local refinement
    try:
        result2 = minimize(
            obj_func,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005}
        )
        if result2.fun < best_side:
            best_side = result2.fun
            best_params = result2.x
    except Exception as e:
        pass
    
    centers_best, angles_best = unflatten(best_params)
    
    # Final validation
    final_valid = True
    for i in range(N):
        for j in range(i+1, N):
            if hexagons_overlap(centers_best[i][0], centers_best[i][1], angles_best[i],
                                 centers_best[j][0], centers_best[j][1], angles_best[j]):
                final_valid = False
                break
        if not final_valid:
            break
    
    if not final_valid:
        centers_best, angles_best = centers0, angles0
        outer_side = required_outer_side(centers_best, angles_best, 0.0)
    else:
        outer_side = required_outer_side(centers_best, angles_best, 0.0)
    
    inner = []
    for (cx, cy), a in zip(centers_best, angles_best):
        inner.append([float(cx), float(cy), float(a)])
    
    return inner, (0.0, 0.0), float(outer_side), 0.0
# EVOLVE-BLOCK-END
