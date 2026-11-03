"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    sqrt3 = math.sqrt(3)
    s = 1.0  # inner hexagon side length

    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon with given center, side, and rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            vx = cx + side * math.cos(theta)
            vy = cy + side * math.sin(theta)
            verts.append((vx, vy))
        return verts

    def hexagon_contains_point(px, py, hx, hy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (hx,hy) with given side and angle."""
        # Transform to hexagon's local coordinate system
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # For a flat-top hexagon with side s centered at origin:
        # |lx| <= s
        # |ly| <= sqrt(3)/2 * s
        # |lx + ly/sqrt(3)| <= s
        # Actually, the constraints are:
        # For each of 6 edges, the point must be on the correct side
        # Edge normals for flat-top hex at angles 0, 60, 120, 180, 240, 300
        for i in range(6):
            theta = i * math.pi / 3
            val = lx * math.cos(theta) + ly * math.sin(theta)
            if val > side * sqrt3 / 2 + 1e-10:
                return False
        return True

    def min_outer_side(positions, angles, outer_angle_deg, center=None):
        """
        Find minimum side length of a regular hexagon (with given orientation)
        centered at 'center' that contains all inner hexagons.
        positions: list of (x, y)
        angles: list of angle_degrees
        outer_angle_deg: orientation of outer hexagon
        center: (cx, cy) or None to compute optimal center
        """
        if center is None:
            # Use centroid as initial center
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
        else:
            cx, cy = center
        
        # Get all vertices of all inner hexagons
        all_verts = []
        for (px, py), ang in zip(positions, angles):
            verts = hex_vertices(px, py, s, ang)
            all_verts.extend(verts)
        
        # For a regular hexagon with side L, centered at (cx,cy), rotated by outer_angle_deg:
        # A point (px, py) is inside if for all 6 edge normals n_i:
        #   (px-cx)*n_i.x + (py-cy)*n_i.y <= L * sqrt(3)/2
        # So L >= max over all points and all edges of [(px-cx)*n_i.x + (py-cy)*n_i.y] / (sqrt(3)/2)
        # L >= max_i max_p [ (p - center) . n_i ] / (sqrt(3)/2)
        # where n_i are unit normals to the 6 edges
        
        outer_rad = math.radians(outer_angle_deg)
        max_proj = 0.0
        for i in range(6):
            theta = outer_rad + i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            for (vx, vy) in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                if proj > max_proj:
                    max_proj = proj
        # Also need to consider the opposite direction (the hexagon is symmetric)
        # Actually, for a convex body, we need max of both +n and -n projections
        # The constraint is: for each edge normal n_i, max_p (p-center).n_i <= L*sqrt(3)/2
        # AND min_p (p-center).n_i >= -L*sqrt(3)/2
        # So L*sqrt(3)/2 >= max( max_p (p-center).n_i, -min_p (p-center).n_i )
        # = max( max_p (p-center).n_i, max_p (center-p).n_i )
        
        # Let me redo: for each of 6 directions (normals), we need:
        # L*sqrt(3)/2 >= max over all vertices of | (v - center) . n_i |
        # Wait, that's not right either. The hexagon is:
        # intersection over i of { p : (p-c).n_i <= L*sqrt(3)/2 }
        # So we need: for all i, for all v: (v-c).n_i <= L*sqrt(3)/2
        # So L >= max_i max_v (v-c).n_i / (sqrt(3)/2)
        
        # But we also need the hexagon to be centered optimally.
        # Let's just compute the required L for a given center.
        
        max_L = 0.0
        for i in range(6):
            theta = outer_rad + i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            for (vx, vy) in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                L_needed = proj / (sqrt3 / 2)
                if L_needed > max_L:
                    max_L = L_needed
        # Also check the negatives (the other side of each edge)
        for i in range(6):
            theta = outer_rad + i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            for (vx, vy) in all_verts:
                proj = (cx - vx) * nx + (cy - vy) * ny
                L_needed = proj / (sqrt3 / 2)
                if L_needed > max_L:
                    max_L = L_needed
        
        return max_L

    def check_disjoint(positions, angles, tol=1e-6):
        """Check if all pairs of inner hexagons are disjoint (no overlap)."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if any vertex of hex i is inside hex j, or vice versa
                verts_i = hex_vertices(positions[i][0], positions[i][1], s, angles[i])
                verts_j = hex_vertices(positions[j][0], positions[j][1], s, angles[j])
                for (vx, vy) in verts_i:
                    if hexagon_contains_point(vx, vy, positions[j][0], positions[j][1], s, angles[j]):
                        return False
                for (vx, vy) in verts_j:
                    if hexagon_contains_point(vx, vy, positions[i][0], positions[i][1], s, angles[i]):
                        return False
        return True

    # Strategy: Start with honeycomb arrangement, then use optimization to improve
    # Variables: 12 positions (24 vars) + 12 angles (12 vars) + outer center (2 vars) + outer angle (1 var) = 39 vars
    
    # Initial honeycomb arrangement
    init_positions = []
    init_angles = []
    for row in range(4):
        y = row * sqrt3
        for col in range(3):
            if row % 2 == 0:
                x = (col - 1) * 2.0
            else:
                x = (col - 1) * 2.0 + 1.0
            init_positions.append((x, y))
            init_angles.append(0.0)
    
    # Compute initial outer side
    init_center = (0.0, 0.0)
    init_outer_angle = 0.0
    init_side = min_outer_side(init_positions, init_angles, init_outer_angle, init_center)
    
    best_side = init_side
    best_positions = [list(p) for p in init_positions]
    best_angles = list(init_angles)
    best_center = init_center
    best_outer_angle = init_outer_angle
    
    # Define the optimization problem
    # x = [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, outer_angle]
    # 39 variables
    
    def objective(x):
        """Return the outer hexagon side length, or a large penalty if hexagons overlap."""
        positions = []
        angles = []
        for i in range(N):
            px = x[3*i]
            py = x[3*i + 1]
            ang = x[3*i + 2]
            positions.append((px, py))
            angles.append(ang)
        cx = x[36]
        cy = x[37]
        outer_ang = x[38]
        
        # Check for overlaps - use a fast bounding box check first
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance for non-overlap is roughly 2*s = 2 for aligned hexagons
                # But with rotation it can be less. Use a conservative bound.
                if dist < 1.5:
                    # Potential overlap, do detailed check
                    if not check_disjoint([positions[i], positions[j]], [angles[i], angles[j]]):
                        return 100.0 + dist  # penalty
        
        side = min_outer_side(positions, angles, outer_ang, (cx, cy))
        return side
    
    # Initial guess
    x0 = []
    for i in range(N):
        x0.append(init_positions[i][0])
        x0.append(init_positions[i][1])
        x0.append(init_angles[i])
    x0.append(init_center[0])
    x0.append(init_center[1])
    x0.append(init_outer_angle)
    
    # Try optimization with multiple random restarts
    np.random.seed(42)
    best_result = best_side
    
    # Try a few different approaches
    # Approach 1: Optimize only outer center and angle for the fixed honeycomb
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        
        # Perturb the honeycomb slightly
        if trial == 0:
            pos = [list(p) for p in init_positions]
            angs = list(init_angles)
        else:
            pos = []
            angs = []
            for i in range(N):
                pos.append([init_positions[i][0] + np.random.uniform(-0.5, 0.5),
                           init_positions[i][1] + np.random.uniform(-0.5, 0.5)])
                angs.append(np.random.uniform(-30, 30))
        
        # Check if perturbed arrangement is valid
        if not check_disjoint(pos, angs):
            continue
        
        # Optimize outer center and angle
        def obj_outer(x):
            cx, cy, oa = x
            side = min_outer_side(pos, angs, oa, (cx, cy))
            return side
        
        x0_outer = [init_center[0], init_center[1], init_outer_angle]
        res = minimize(obj_outer, x0_outer, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success or res.fun < best_result:
            cx, cy, oa = res.x
            side = res.fun
            if side < best_result:
                best_result = side
                best_positions = pos
                best_angles = angs
                best_center = (cx, cy)
                best_outer_angle = oa
    
    # Approach 2: Full optimization of all variables
    if time.time() < deadline - 30:
        # Use Nelder-Mead on the full 39-dim problem
        # This is expensive, so limit iterations
        def obj_full(x):
            positions = []
            angles = []
            for i in range(N):
                positions.append((x[3*i], x[3*i+1]))
                angles.append(x[3*i+2])
            cx, cy = x[36], x[37]
            oa = x[38]
            
            # Quick overlap check
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1.8:
                        if not check_disjoint([positions[i], positions[j]], [angles[i], angles[j]]):
                            return 100.0 + dist
            
            return min_outer_side(positions, angles, oa, (cx, cy))
        
        # Try from the best honeycomb
        x0_full = []
        for i in range(N):
            x0_full.append(best_positions[i][0])
            x0_full.append(best_positions[i][1])
            x0_full.append(best_angles[i])
        x0_full.append(best_center[0])
        x0_full.append(best_center[1])
        x0_full.append(best_outer_angle)
        
        try:
            res = minimize(obj_full, x0_full, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_result:
                # Extract the solution
                for i in range(N):
                    best_positions[i][0] = res.x[3*i]
                    best_positions[i][1] = res.x[3*i+1]
                    best_angles[i] = res.x[3*i+2]
                best_center = (res.x[36], res.x[37])
                best_outer_angle = res.x[38]
                best_result = res.fun
        except Exception:
            pass
    
    # Final validation: ensure the best solution is valid
    if not check_disjoint(best_positions, best_angles):
        # Fall back to honeycomb
        best_positions = [list(p) for p in init_positions]
        best_angles = list(init_angles)
        best_center = init_center
        best_outer_angle = init_outer_angle
        best_result = init_side
    
    # Build the inner list
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    
    # Add small margin for floating point
    final_side = best_result * 1.001
    
    return inner, best_center, final_side, best_outer_angle
# EVOLVE-BLOCK-END
