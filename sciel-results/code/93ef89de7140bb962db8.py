import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_center, outer_angle):
        """
        Calculate the minimum side length of a regular hexagon centered at outer_center 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            # Translate vertices relative to outer center
            verts_rel = verts - np.array(outer_center)
            projs = verts_rel @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(poly1, poly2, tol=1e-9):
        """
        Fast check for disjointness of two convex hexagons.
        Returns True if disjoint, False if overlapping.
        Uses Separating Axis Theorem (SAT).
        """
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 - tol or max_x2 < min_x1 - tol or max_y1 < min_y2 - tol or max_y2 < min_y1 - tol:
            return True

        # SAT: Check all edge normals of both polygons
        axes = []
        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            edge = p2 - p1
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
        for i in range(6):
            p1 = poly2[i]
            p2 = poly2[(i+1)%6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
        
        for axis in axes:
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - tol or max2 < min1 - tol:
                return True
        
        return False

    # Strategy:
    # 1. Generate a set of initial configurations (honeycomb lattice, random).
    # 2. For each, run SLSQP to minimize outer side length subject to non-overlap constraints.
    # 3. Polish the outer hexagon parameters (center, angle) for the best inner configuration.
    # 4. Return the best valid result.

    best_score = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = None
    best_side = None

    sqrt3 = math.sqrt(3)

    # Generate initial configurations
    inits = []
    
    # 1. Honeycomb lattice (standard touching)
    # Centers at distance sqrt(3) apart.
    # Basis: v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # This forms a triangular lattice of hex centers.
    # Let's pick 12 points close to origin.
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5 + j * 0.0 # v2 x is 0
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0) # Center it
    init_ang = np.zeros(N)
    inits.append((init_pos, init_ang))
    
    # 2. Random starts
    np.random.seed(42)
    for _ in range(5):
        # Random positions within a reasonable bounding box
        pos = np.random.uniform(-4, 4, (N, 2))
        # Simple rejection to avoid obvious overlaps? SLSQP will handle it, but start valid is better.
        # Let's just try a few random valid-ish starts
        ang = np.random.uniform(0, 60, N)
        inits.append((pos, ang))

    for idx, (init_pos, init_ang) in enumerate(inits):
        if time.time() > deadline - 20:
            break

        # Define objective and constraints for SLSQP
        # Variables: x0...x11 (x coords), y0...y11 (y coords), a0...a11 (angles in rad)
        # Total 36 variables.
        
        def objective(params):
            pos = params[:2*N].reshape(N, 2)
            ang = params[2*N:]
            # Outer hexagon is fixed at (0,0) and 0 deg for this stage?
            # No, we want to minimize the side length.
            # The side length depends on the max projection.
            # We can optimize the outer center and angle in a separate polish step.
            # For now, assume outer is centered at (0,0) and angle 0.
            # This might be suboptimal if the inner cluster is off-center or rotated.
            # However, due to symmetry, we can assume the optimal outer hex is centered at the centroid of the inner hexes?
            # Not necessarily. But let's try this first.
            
            # Calculate min side length
            side = calc_min_outer_side(pos, np.degrees(ang), [0.0, 0.0], 0.0)
            return side

        def constraints(params):
            pos = params[:2*N].reshape(N, 2)
            ang = params[2*N:]
            
            cons = []
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, np.degrees(ang[i])) for i in range(N)]
            
            # Non-overlap constraints
            # For SLSQP, constraints are g(x) >= 0.
            # We need a smooth measure of separation.
            # The minimum distance between two convex polygons is a valid constraint.
            # However, calculating exact min distance is expensive and non-smooth.
            # Alternative: Use SAT separation depth.
            # If they overlap, depth is negative. If separated, depth is positive.
            # We can define the constraint as the minimum separation distance along all SAT axes.
            # sep = min_axis (max_proj1 - min_proj2, max_proj2 - min_proj1)
            # This is the negative of the maximum penetration depth?
            # Actually, if we define d = min_axis (gap), then d >= 0 means disjoint.
            
            for i in range(N):
                for j in range(i+1, N):
                    p1 = polys[i]
                    p2 = polys[j]
                    
                    # Quick bounding box check
                    min_x1, min_y1 = np.min(p1, axis=0)
                    max_x1, max_y1 = np.max(p1, axis=0)
                    min_x2, min_y2 = np.min(p2, axis=0)
                    max_x2, max_y2 = np.max(p2, axis=0)
                    
                    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                        # Clearly separated, constraint is satisfied (large positive)
                        cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j, val=10.0: val})
                        continue

                    # Calculate min separation
                    min_sep = float('inf')
                    axes = []
                    for k in range(6):
                        edge = p1[(k+1)%6] - p1[k]
                        n = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-9: axes.append(n/norm)
                    for k in range(6):
                        edge = p2[(k+1)%6] - p2[k]
                        n = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-9: axes.append(n/norm)
                    
                    for axis in axes:
                        proj1 = np.dot(p1, axis)
                        proj2 = np.dot(p2, axis)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        # Gap between the two intervals [min1, max1] and [min2, max2]
                        # If they overlap, gap is negative (penetration depth)
                        # If they are apart, gap is positive.
                        # We want max(min1 - max2, min2 - max1) ? No.
                        # Distance between intervals is:
                        # If max1 < min2: min2 - max1
                        # If max2 < min1: min1 - max2
                        # Else: 0 (or negative if we define penetration)
                        
                        # Let's define penetration depth as:
                        # pen = min(max1, max2) - max(min1, min2)
                        # If pen < 0, they are separated by -pen.
                        # If pen >= 0, they overlap by pen.
                        # We want separation >= 0.
                        # sep = -pen = max(min1, min2) - min(max1, max2)
                        
                        sep = max(min1, min2) - min(max1, max2)
                        if sep < min_sep:
                            min_sep = sep
                    
                    # Constraint: min_sep >= 0
                    # To make it smooth-ish and handle the case where they are far apart,
                    # we can just return min_sep.
                    # If they are far apart, min_sep will be large positive.
                    cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j, val=min_sep: val})
            
            return cons

        x0 = np.concatenate([init_pos.flatten(), init_ang])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                new_pos = res.x[:2*N].reshape(N, 2)
                new_ang = res.x[2*N:]
                
                # Verify validity
                polys = [get_hex_verts(new_pos[i][0], new_pos[i][1], 1.0, np.degrees(new_ang[i])) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j], tol=1e-8):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # Polish outer hexagon
                    # Optimize outer_center (2) and outer_angle (1)
                    def outer_obj(outer_params):
                        oc = outer_params[:2]
                        oa = outer_params[2]
                        return calc_min_outer_side(new_pos, np.degrees(new_ang), oc, oa)
                    
                    # Initial guess: centroid of inner hexes, angle 0
                    centroid = np.mean(new_pos, axis=0)
                    x0_outer = np.array([centroid[0], centroid[1], 0.0])
                    
                    res_outer = minimize(
                        outer_obj, 
                        x0_outer, 
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    
                    best_outer_center = res_outer.x[:2]
                    best_outer_angle = res_outer.x[2]
                    side = res_outer.fun
                    
                    if side < best_score:
                        best_score = side
                        best_inner = new_pos
                        best_outer_center = res_outer.x[:2]
                        best_outer_angle = res_outer.x[2]
                        best_side = side
                        
        except Exception as e:
            pass

    # Fallback if no valid configuration was found
    if best_inner is None:
        # Use the simple honeycomb fallback
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        best_inner = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        angles = np.zeros(N)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0
        best_side = calc_min_outer_side(best_inner, angles, best_outer_center, best_outer_angle)

    final_positions = best_inner
    final_angles = np.zeros(N) # We didn't track final angles in best_inner, need to recover or store
    # Wait, best_inner is just positions. I need to store the angles too.
    # Let's re-run the polish for the best_inner? No, I lost the angles.
    # I should have stored best_angles.
    
    # Let's fix the logic: store best_angles as well.
    # Since I didn't, I will just assume angles are 0 for the fallback or re-optimize?
    # Actually, the SLSQP result `res.x` contained the angles.
    # I will modify the code to store `best_angles`.
    
    # Re-implementing the storage part in the loop above was ideal, but since I'm writing the final code now:
    # I will assume the best configuration found in the loop is valid and use it.
    # If the loop found nothing, fallback uses 0 angles.
    
    # Let's refine the code to properly store angles.
    # I will rewrite the loop logic in the final output to be correct.
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], 0.0]) # Placeholder, will be fixed below
    
    # To ensure correctness, I will wrap the whole thing in a function that properly tracks state.
    # The code above is a sketch. I will write the clean version below.
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle

# The code above has a flaw: it doesn't store the angles for the best solution.
# I will provide the corrected complete program below.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_center, outer_angle):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            verts_rel = verts - np.array(outer_center)
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(poly1, poly2, tol=1e-9):
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 - tol or max_x2 < min_x1 - tol or max_y1 < min_y2 - tol or max_y2 < min_y1 - tol:
            return True

        axes = []
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9: axes.append(n/norm)
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9: axes.append(n/norm)
        
        for axis in axes:
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - tol or max2 < min1 - tol:
                return True
        
        return False

    best_score = float('inf')
    best_inner_pos = None
    best_inner_ang = None
    best_outer_center = None
    best_outer_angle = None
    best_side = None

    sqrt3 = math.sqrt(3)
    inits = []
    
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append((init_pos, init_ang))
    
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-4, 4, (N, 2))
        ang = np.random.uniform(0, 60, N)
        inits.append((pos, ang))

    for idx, (init_pos, init_ang) in enumerate(inits):
        if time.time() > deadline - 20:
            break

        def objective(params):
            pos = params[:2*N].reshape(N, 2)
            ang = params[2*N:]
            return calc_min_outer_side(pos, np.degrees(ang), [0.0, 0.0], 0.0)

        def constraints(params):
            pos = params[:2*N].reshape(N, 2)
            ang = params[2*N:]
            
            cons = []
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, np.degrees(ang[i])) for i in range(N)]
            
            for i in range(N):
                for j in range(i+1, N):
                    p1 = polys[i]
                    p2 = polys[j]
                    
                    min_x1, min_y1 = np.min(p1, axis=0)
                    max_x1, max_y1 = np.max(p1, axis=0)
                    min_x2, min_y2 = np.min(p2, axis=0)
                    max_x2, max_y2 = np.max(p2, axis=0)
                    
                    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                        cons.append({'type': 'ineq', 'fun': lambda p, val=10.0: val})
                        continue

                    min_sep = float('inf')
                    axes = []
                    for k in range(6):
                        edge = p1[(k+1)%6] - p1[k]
                        n = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-9: axes.append(n/norm)
                    for k in range(6):
                        edge = p2[(k+1)%6] - p2[k]
                        n = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-9: axes.append(n/norm)
                    
                    for axis in axes:
                        proj1 = np.dot(p1, axis)
                        proj2 = np.dot(p2, axis)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        sep = max(min1, min2) - min(max1, max2)
                        if sep < min_sep:
                            min_sep = sep
                    
                    cons.append({'type': 'ineq', 'fun': lambda p, val=min_sep: val})
            
            return cons

        x0 = np.concatenate([init_pos.flatten(), init_ang])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                new_pos = res.x[:2*N].reshape(N, 2)
                new_ang = res.x[2*N:]
                
                polys = [get_hex_verts(new_pos[i][0], new_pos[i][1], 1.0, np.degrees(new_ang[i])) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j], tol=1e-8):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    def outer_obj(outer_params):
                        oc = outer_params[:2]
                        oa = outer_params[2]
                        return calc_min_outer_side(new_pos, np.degrees(new_ang), oc, oa)
                    
                    centroid = np.mean(new_pos, axis=0)
                    x0_outer = np.array([centroid[0], centroid[1], 0.0])
                    
                    res_outer = minimize(
                        outer_obj, 
                        x0_outer, 
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    
                    side = res_outer.fun
                    
                    if side < best_score:
                        best_score = side
                        best_inner_pos = new_pos
                        best_inner_ang = new_ang
                        best_outer_center = res_outer.x[:2]
                        best_outer_angle = res_outer.x[2]
                        best_side = side
                        
        except Exception as e:
            pass

    if best_inner_pos is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        best_inner_pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        best_inner_ang = np.zeros(N)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0
        best_side = calc_min_outer_side(best_inner_pos, best_inner_ang, best_outer_center, best_outer_angle)

    inner = []
    for i in range(N):
        inner.append([best_inner_pos[i][0], best_inner_pos[i][1], np.degrees(best_inner_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle
