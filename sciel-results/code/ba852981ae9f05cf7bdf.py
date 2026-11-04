import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    sqrt3 = math.sqrt(3)

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
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
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # For a regular hexagon with side S, the inradius (apothem) is S*sqrt(3)/2.
        # The maximum projection onto edge normals gives the required inradius.
        # So S = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def soft_min_outer_side(positions, angles, outer_angle=0.0, temperature=0.01):
        """
        Smooth surrogate for calc_min_outer_side using soft-max.
        This makes the objective differentiable and easier to optimize.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        all_projs = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            all_projs.append(projs.flatten())
        
        all_projs = np.concatenate(all_projs)  # (N*6*6,)
        
        # Soft-max to approximate max
        scaled = all_projs / temperature
        max_scaled = np.max(scaled)
        exp_vals = np.exp(scaled - max_scaled)
        soft_max = temperature * (max_scaled + np.log(np.sum(exp_vals)))
        
        return soft_max * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            # Edge 1
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            # Edge 2
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
                return True # Separating axis found, disjoint
                
        return False # No separating axis, overlapping

    def is_valid_config(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    # --- Part 1: Lattice-based Polyhex Search (Inspiration) ---
    
    def lattice_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return (x, y)
        
    def is_connected(coords):
        if not coords: return True
        coord_set = set(coords)
        start = next(iter(coord_set))
        visited = {start}
        stack = [start]
        while stack:
            q, r = stack.pop()
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            for nq, nr in neighbors:
                if (nq, nr) in coord_set and (nq, nr) not in visited:
                    visited.add((nq, nr))
                    stack.append((nq, nr))
        return len(visited) == len(coords)

    def normalize_shape(coords):
        xs = [lattice_to_cartesian(q, r)[0] for q, r in coords]
        ys = [lattice_to_cartesian(q, r)[1] for q, r in coords]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(xs)
        return [(lattice_to_cartesian(q, r)[0] - cx, lattice_to_cartesian(q, r)[1] - cy) for q, r in coords]

    def generate_random_polyhex(size):
        coords = {(0,0)}
        frontier = [(0,0)]
        while len(coords) < size:
            if not frontier:
                return None
            random.shuffle(frontier)
            q, r = frontier[0]
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            random.shuffle(neighbors)
            added = False
            for nq, nr in neighbors:
                if (nq, nr) not in coords:
                    coords.add((nq, nr))
                    frontier.append((nq, nr))
                    added = True
                    break
            if not added:
                return None
        if is_connected(list(coords)):
            return list(coords)
        return None

    candidate_shapes = []
    
    # Add some specific compact shapes
    shape_2x3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    if is_connected(shape_2x3):
        candidate_shapes.append(normalize_shape(shape_2x3))
        
    shape_3x2 = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
    if is_connected(shape_3x2):
        candidate_shapes.append(normalize_shape(shape_3x2))
        
    # Generate random polyhexes
    for _ in range(50):
        if time.time() > deadline - 60: break
        shape = generate_random_polyhex(N)
        if shape:
            candidate_shapes.append(normalize_shape(shape))
            
    # Deduplicate
    unique_shapes = []
    seen = set()
    for shape in candidate_shapes:
        key = tuple(sorted((round(x, 3), round(y, 3)) for x, y in shape))
        if key not in seen:
            seen.add(key)
            unique_shapes.append(shape)

    best_score = float('inf')
    best_params = None

    # Evaluate each shape with optimized global rotation
    for shape in unique_shapes:
        if time.time() > deadline - 20: break
        
        init_positions = np.array(shape)
        
        def obj_cluster_fixed_outer(params):
            global_rot = params[0]
            c = math.cos(global_rot)
            s = math.sin(global_rot)
            rot_mat = np.array([[c, -s], [s, c]])
            rotated_positions = init_positions @ rot_mat.T
            angles_deg = np.full(N, math.degrees(global_rot))
            return calc_min_outer_side(rotated_positions, angles_deg, 0.0)
            
        x0_rot = np.array([0.0])
        try:
            res = minimize(obj_cluster_fixed_outer, x0_rot, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                global_rot = res.x[0]
                c = math.cos(global_rot)
                s = math.sin(global_rot)
                rot_mat = np.array([[c, -s], [s, c]])
                final_positions = init_positions @ rot_mat.T
                final_angles = np.full(N, math.degrees(global_rot))
                
                if is_valid_config(final_positions, final_angles):
                    best_score = res.fun
                    params_list = final_positions.flatten().tolist() + final_angles.tolist()
                    best_params = np.array(params_list)
        except Exception:
            pass

    # --- Part 2: General Optimization Strategy ---
    if best_params is None:
        # Fallback to standard honeycomb lattice start
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        x0 = np.concatenate([init_positions.flatten(), init_angles])
        
        # Calculate initial score
        init_score = calc_min_outer_side(init_positions, init_angles, 0.0)
        if init_score < best_score:
            best_score = init_score
            best_params = x0
    else:
        x0 = best_params
        # Re-evaluate best_score to be sure
        pos = best_params[:2*N].reshape(N, 2)
        ang = best_params[2*N:]
        best_score = calc_min_outer_side(pos, ang, 0.0)

    # --- Part 3: SLSQP Optimization with Hard Constraints (Inspiration) ---
    # We use SLSQP to refine the configuration.
    # Variables: [x1, y1, ..., xN, yN, a1, ..., aN]
    # Constraints: Separation for all pairs (SAT based)
    
    def slsqp_objective(params):
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        return calc_min_outer_side(positions, angles, 0.0)

    def slsqp_constraints():
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def sep_constraint(params, i=i, j=j):
                    if time.time() > deadline - 5:
                        return 1.0
                    pos = params[:2*N].reshape(N, 2)
                    angles = params[2*N:]
                    p1 = pos[i]
                    p2 = pos[j]
                    a1 = angles[i]
                    a2 = angles[j]
                    
                    v1 = get_hex_verts(p1[0], p1[1], 1.0, a1)
                    v2 = get_hex_verts(p2[0], p2[1], 1.0, a2)
                    
                    # SAT check for separation
                    # We need a smooth-ish constraint function.
                    # The minimum separation distance is a valid inequality constraint >= 0.
                    # However, min separation is not differentiable everywhere.
                    # SLSQP uses finite differences, so it can handle non-smoothness to some extent,
                    # but it might struggle.
                    # A better constraint for SAT is:
                    # For each axis k, (max_proj1 - min_proj2) >= 0 OR (max_proj2 - min_proj1) >= 0.
                    # This is a disjunction, which SLSQP cannot handle directly.
                    # Instead, we rely on the fact that if they are disjoint, there exists an axis with gap > 0.
                    # We can define the constraint as the minimum gap over all axes, but if they overlap, all gaps are negative.
                    # So: min_gap >= -epsilon.
                    
                    min_gap = float('inf')
                    axes = []
                    for k in range(6):
                        edge1 = v1[(k+1)%6] - v1[k]
                        normal1 = np.array([-edge1[1], edge1[0]])
                        axes.append(normal1)
                        edge2 = v2[(k+1)%6] - v2[k]
                        normal2 = np.array([-edge2[1], edge2[0]])
                        axes.append(normal2)
                    
                    for axis in axes:
                        norm = np.linalg.norm(axis)
                        if norm < 1e-9: continue
                        axis = axis / norm
                        proj1 = v1 @ axis
                        proj2 = v2 @ axis
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        gap1 = max1 - min2
                        gap2 = max2 - min1
                        # If disjoint, at least one gap is positive.
                        # We want the constraint to be violated (negative) if they overlap.
                        # If they overlap, all gaps are negative.
                        # The "most violated" axis is the one with the largest negative gap? No.
                        # If they overlap, the gap is negative for all axes.
                        # We want the constraint function to be < 0 when overlapping.
                        # Let's use the minimum of the positive gaps? No.
                        # Standard SAT constraint for optimization is tricky.
                        # Alternative: Use a penalty in the objective for SLSQP? No, SLSQP needs hard constraints.
                        # Let's try: min_gap = min( max(0, max1-min2), max(0, max2-min1) )? No.
                        
                        # Actually, if we define g(x) = min_axis ( max1 - min2 ) and h(x) = min_axis ( max2 - min1 ).
                        # If disjoint, either min(g) > 0 or min(h) > 0 is not quite right.
                        # Disjoint means: Exists axis k such that (max1_k < min2_k) OR (max2_k < min1_k).
                        # This is an OR condition.
                        # SLSQP cannot handle OR constraints.
                        
                        # Therefore, SLSQP with hard SAT constraints is difficult to implement correctly for disjointness.
                        # The inspiration code might have had a similar issue or used a specific trick.
                        # Let's stick to Nelder-Mead with a penalty function, which is more robust for this type of non-convex, non-smooth constraint.
                        # The "inspiration" was SLSQP, but if it's hard to implement correctly, we might lose time.
                        # Let's use Nelder-Mead with a strong penalty, but improved.
                        
                        # For now, return a large positive value if they seem close/overlapping to act as a barrier.
                        # This is not a proper constraint.
                        # Let's abandon SLSQP for hard constraints and stick to the penalty method but refine it.
                        
                        # We'll return 0.0 here to satisfy the function signature, but we won't use SLSQP for this.
                        pass
                    return 0.0
                cons.append({'type': 'ineq', 'fun': sep_constraint})
        return cons

    # Since SLSQP constraints are tricky for SAT, we will proceed with Nelder-Mead as in the current program,
    # but we will add multiple restarts and a more robust penalty function.

    def objective(params):
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    # Calculate overlap depth to make penalty smooth-ish
                    # Use center distance minus sum of inradii as a proxy for overlap depth
                    d = np.linalg.norm(polys[i].mean(axis=0) - polys[j].mean(axis=0))
                    inradius_sum = 2.0 * math.sqrt(3) / 2.0  # Sum of inradii of two unit hexagons
                    overlap = max(0, inradius_sum - d)
                    penalty += 50.0 * overlap * overlap + 5.0
        if penalty > 0:
            return penalty + soft_min_outer_side(positions, angles, 0.0, temperature=0.05)
        return soft_min_outer_side(positions, angles, 0.0, temperature=0.01)

    # Run Nelder-Mead from the best starting point
    try:
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if res.fun < best_score:
            new_positions = res.x[:2*N].reshape(N, 2)
            new_angles = res.x[2*N:]
            if is_valid_config(new_positions, new_angles):
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass

    # Local refinement with multiple restarts
    if best_params is not None:
        current_best = best_score
        current_params = best_params
        
        for _ in range(15):
            if time.time() > deadline - 15:
                break
                
            noise_pos = np.random.normal(0, 0.05, 2*N)
            noise_ang = np.random.normal(0, 3, N) * math.pi / 180
            x_perturbed = current_params.copy()
            x_perturbed[:2*N] += noise_pos
            x_perturbed[2*N:] += noise_ang
            
            try:
                res = minimize(
                    objective, 
                    x_perturbed, 
                    method='Nelder-Mead', 
                    options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < current_best:
                    new_positions = res.x[:2*N].reshape(N, 2)
                    new_angles = res.x[2*N:]
                    if is_valid_config(new_positions, new_angles):
                        current_best = res.fun
                        current_params = res.x
            except Exception:
                continue

        best_score = current_best
        best_params = current_params
        
        # Final high-precision polish pass with exact objective
        if time.time() < deadline - 20:
            def exact_objective(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                return calc_min_outer_side(positions, angles, 0.0)
            
            try:
                res_polish = minimize(
                    exact_objective, 
                    best_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                new_positions = res_polish.x[:2*N].reshape(N, 2)
                new_angles = res_polish.x[2*N:]
                if is_valid_config(new_positions, new_angles):
                    polish_score = calc_min_outer_side(new_positions, new_angles, 0.0)
                    if polish_score < best_score:
                        best_score = polish_score
                        best_params = res_polish.x
            except Exception:
                pass

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
