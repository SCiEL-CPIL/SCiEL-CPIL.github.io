"""Packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize, Bounds
import time

N = 11
DEADLINE = 140.0  # Safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_hexagon(p, center, side, angle_deg):
    """Check if point p is inside a regular hexagon."""
    # Transform to hexagon's local frame
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    # Regular hexagon with side s: |x| <= s, |y| <= s*sqrt(3)/2, |x|/s + |y|/(s*sqrt(3)/2) <= 2
    s = side
    return (np.abs(lx) <= s + 1e-10 and 
            np.abs(ly) <= s * np.sqrt(3) / 2 + 1e-10 and
            np.abs(lx) / s + np.abs(ly) / (s * np.sqrt(3) / 2) <= 2 + 1e-10)

def hexagon_distance(h1_center, h1_angle, h2_center, h2_angle, side=1.0):
    """Approximate minimum distance between two hexagons (center distance minus circumradii)."""
    d = np.linalg.norm(np.array(h1_center) - np.array(h2_center))
    # For hexagons, the inradius is side * sqrt(3)/2, circumradius is side
    # Minimum distance between boundaries is approximately d - 2*inradius for aligned hexagons
    # But for safety, use d - 2*side as a conservative lower bound on separation needed
    return d - 2 * side

def objective(params, outer_center, outer_side, outer_angle):
    """Objective: minimize outer_side + penalty for constraint violations."""
    n = len(params) // 3
    inner_centers = params[:2*n].reshape(n, 2)
    inner_angles = params[2*n:]
    
    # Penalty for inner hexagons overlapping with outer boundary
    outer_center_arr = np.array(outer_center)
    penalty_outer = 0.0
    for i in range(n):
        cx, cy = inner_centers[i]
        ang = inner_angles[i]
        # Check all 6 vertices of inner hexagon
        verts = hex_vertices(cx, cy, ang, 1.0)
        for v in verts:
            # Transform to outer hexagon local frame
            oa = np.radians(outer_angle)
            cos_a = np.cos(-oa)
            sin_a = np.sin(-oa)
            dx = v[0] - outer_center_arr[0]
            dy = v[1] - outer_center_arr[1]
            lx = cos_a * dx - sin_a * dy
            ly = sin_a * dx + cos_a * dy
            s = outer_side
            # Distance from hexagon boundary (negative if inside)
            dist = max(np.abs(lx) - s, np.abs(ly) - s * np.sqrt(3) / 2, 
                       np.abs(lx) / s + np.abs(ly) / (s * np.sqrt(3) / 2) - 2)
            if dist > 0:
                penalty_outer += dist ** 2
    
    # Penalty for inner hexagons overlapping each other
    penalty_overlap = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(inner_centers[i] - inner_centers[j])
            # Minimum center distance for non-overlapping unit hexagons is 2*sqrt(3) ≈ 3.464
            min_d = 2 * np.sqrt(3)
            if d < min_d:
                penalty_overlap += (min_d - d) ** 2
    
    return penalty_outer + penalty_overlap

def construct():
    start_time = time.time()
    
    # Initial hexagonal lattice arrangement
    # Central hexagon at origin, 6 around it, 4 more
    # Hexagonal grid spacing: sqrt(3) for adjacent hexagons
    sp = np.sqrt(3)
    
    # Mode 0: standard hexagonal lattice
    init_positions = [
        (0, 0),
        (sp, 0), (-sp, 0),
        (sp/2, sp*np.sqrt(3)/2), (-sp/2, sp*np.sqrt(3)/2),
        (sp/2, -sp*np.sqrt(3)/2), (-sp/2, -sp*np.sqrt(3)/2),
        (0, sp*np.sqrt(3)), (0, -sp*np.sqrt(3)),
        (sp*np.sqrt(3), 0), (-sp*np.sqrt(3), 0),
    ]
    
    # We have 11 positions, but let's use a more compact arrangement
    # Actually, for 11 hexagons, a good arrangement is:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Or a hexagonal cluster
    
    # Let's use a hexagonal number arrangement: 1 + 6 + 4 = 11
    # Center: 1
    # First ring: 6
    # Second ring partial: 4
    
    init_positions = [
        (0, 0),  # center
        (sp, 0), (-sp, 0),
        (sp/2, sp*np.sqrt(3)/2), (-sp/2, sp*np.sqrt(3)/2),
        (sp/2, -sp*np.sqrt(3)/2), (-sp/2, -sp*np.sqrt(3)/2),
        # 4 more in second ring
        (sp, sp*np.sqrt(3)), (-sp, -sp*np.sqrt(3)),
        (sp*np.sqrt(3)/2, sp*3*np.sqrt(3)/2), (-sp*np.sqrt(3)/2, -sp*3*np.sqrt(3)/2),
    ]
    
    # Verify we have 11
    assert len(init_positions) == N
    
    best_result = None
    best_score = float('inf')
    
    # Try different outer rotations
    outer_rotations = [0, 30, 60]
    
    for outer_angle_deg in outer_rotations:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Estimate required outer side length
        # Find max extent of inner hexagons
        max_extent = 0
        for cx, cy in init_positions:
            for ang in [0]:
                verts = hex_vertices(cx, cy, ang, 1.0)
                for v in verts:
                    dist = np.linalg.norm(v)
                    max_extent = max(max_extent, dist)
        
        # Outer hexagon must contain all vertices
        # For a hexagon with side s, the apothem is s*sqrt(3)/2, circumradius is s
        # We need s >= max_extent / (some factor depending on rotation)
        initial_outer_side = max_extent / 0.866 + 0.1  # Conservative estimate
        
        # Parameters: x, y, angle for each inner hexagon + outer side length
        # Let's fix outer center at origin for simplicity
        outer_center = (0.0, 0.0)
        
        # Initial parameters
        x0 = []
        for cx, cy in init_positions:
            x0.extend([cx, cy, 0.0])
        x0.append(initial_outer_side)
        
        n_params = 3 * N + 1
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
        bounds.append([1.0, 20.0])  # outer side length
        
        def obj_func(params):
            if time.time() - start_time > DEADLINE:
                return 1e10
            n_inner = len(params) - 1
            inner_params = params[:3*n_inner]
            outer_side = params[-1]
            
            # Penalty for constraints
            penalty = 0.0
            
            # Check inner hexagons against outer boundary
            for i in range(n_inner):
                cx = inner_params[i*3]
                cy = inner_params[i*3+1]
                ang = inner_params[i*3+2]
                verts = hex_vertices(cx, cy, ang, 1.0)
                for v in verts:
                    # Check if vertex is inside outer hexagon
                    oa = np.radians(outer_angle_deg)
                    cos_a = np.cos(-oa)
                    sin_a = np.sin(-oa)
                    dx = v[0] - outer_center[0]
                    dy = v[1] - outer_center[1]
                    lx = cos_a * dx - sin_a * dy
                    ly = sin_a * dx + cos_a * dy
                    s = outer_side
                    # If outside, add penalty
                    dist_out = max(np.abs(lx) - s, np.abs(ly) - s * np.sqrt(3) / 2, 
                                   np.abs(lx) / s + np.abs(ly) / (s * np.sqrt(3) / 2) - 2)
                    if dist_out > 0:
                        penalty += dist_out ** 2 * 1000
            
            # Check inner hexagons against each other
            for i in range(n_inner):
                for j in range(i+1, n_inner):
                    cx1 = inner_params[i*3]
                    cy1 = inner_params[i*3+1]
                    cx2 = inner_params[j*3]
                    cy2 = inner_params[j*3+1]
                    d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                    min_d = 2 * np.sqrt(3)
                    if d < min_d:
                        penalty += (min_d - d) ** 2 * 1000
            
            # Objective: minimize outer side + penalty
            return outer_side + penalty
        
        try:
            result = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < best_score:
                n_inner = N
                inner_params = result.x[:3*n_inner]
                outer_side = result.x[-1]
                
                # Extract inner hexagons
                inner = []
                for i in range(n_inner):
                    cx = inner_params[i*3]
                    cy = inner_params[i*3+1]
                    ang = inner_params[i*3+2]
                    inner.append([cx, cy, ang])
                
                # Verify validity
                valid = True
                for i in range(n_inner):
                    cx, cy, ang = inner[i]
                    verts = hex_vertices(cx, cy, ang, 1.0)
                    for v in verts:
                        oa = np.radians(outer_angle_deg)
                        cos_a = np.cos(-oa)
                        sin_a = np.sin(-oa)
                        dx = v[0] - outer_center[0]
                        dy = v[1] - outer_center[1]
                        lx = cos_a * dx - sin_a * dy
                        ly = sin_a * dx + cos_a * dy
                        s = outer_side
                        if (np.abs(lx) > s + 1e-6 or 
                            np.abs(ly) > s * np.sqrt(3) / 2 + 1e-6 or
                            np.abs(lx) / s + np.abs(ly) / (s * np.sqrt(3) / 2) > 2 + 1e-6):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    for i in range(n_inner):
                        for j in range(i+1, n_inner):
                            d = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                            if d < 2 * np.sqrt(3) - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = outer_side
                    if score < best_score:
                        best_score = score
                        best_result = (inner, outer_center, outer_side, outer_angle_deg)
                        
        except Exception as e:
            pass
    
    # If no valid result found, use a conservative default
    if best_result is None:
        # Fallback: place hexagons in a line with large outer hexagon
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # SLSQP polish on the best result
    if time.time() - start_time < DEADLINE - 20:
        inner, outer_center, outer_side, outer_angle_deg = best_result
        x0 = []
        for cx, cy, ang in inner:
            x0.extend([cx, cy, ang])
        x0.append(outer_side)
        
        bounds = []
        for i in range(N):
            bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
        bounds.append([1.0, 20.0])
        
        try:
            result = minimize(obj_func, x0, method='SLSQP', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-10})
            
            if result.fun < best_score:
                n_inner = N
                inner_params = result.x[:3*n_inner]
                outer_side = result.x[-1]
                
                inner = []
                for i in range(n_inner):
                    cx = inner_params[i*3]
                    cy = inner_params[i*3+1]
                    ang = inner_params[i*3+2]
                    inner.append([cx, cy, ang])
                
                # Verify
                valid = True
                for i in range(n_inner):
                    cx, cy, ang = inner[i]
                    verts = hex_vertices(cx, cy, ang, 1.0)
                    for v in verts:
                        oa = np.radians(outer_angle_deg)
                        cos_a = np.cos(-oa)
                        sin_a = np.sin(-oa)
                        dx = v[0] - outer_center[0]
                        dy = v[1] - outer_center[1]
                        lx = cos_a * dx - sin_a * dy
                        ly = sin_a * dx + cos_a * dy
                        s = outer_side
                        if (np.abs(lx) > s + 1e-5 or 
                            np.abs(ly) > s * np.sqrt(3) / 2 + 1e-5 or
                            np.abs(lx) / s + np.abs(ly) / (s * np.sqrt(3) / 2) > 2 + 1e-5):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    for i in range(n_inner):
                        for j in range(i+1, n_inner):
                            d = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                            if d < 2 * np.sqrt(3) - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    best_result = (inner, outer_center, outer_side, outer_angle_deg)
                    best_score = outer_side
                    
        except Exception as e:
            pass
    
    return best_result
