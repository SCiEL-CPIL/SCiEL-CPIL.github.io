import math
import time
import numpy as np
from scipy.optimize import minimize
import random
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    sqrt3 = math.sqrt(3)

    def get_hex_verts(cx, cy, side, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle_rad=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle_rad that contains all inner hexagons.
        angles are in radians.
        """
        theta_rad = outer_angle_rad
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
        
        return max_proj * 2.0 / math.sqrt(3)

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
            if max1 < min2 or max2 < min1:
                return True # Separating axis found, disjoint
                
        return False # No separating axis, overlapping

    def is_valid_config(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

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
                    penalty += 10.0
        if penalty > 0:
            return penalty + calc_min_outer_side(positions, angles, 0.0)
        return calc_min_outer_side(positions, angles, 0.0)

    # --- Part 1: Lattice-based Polyhex Search (Inspiration) ---
    # Generate candidate polyhex shapes (connected sets of 12 hexagons on honeycomb lattice)
    
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

    # Evaluate each shape with optimized global rotation and outer rotation
    for shape in unique_shapes:
        if time.time() > deadline - 20: break
        
        init_positions = np.array(shape)
        init_angles = np.zeros(N)
        
        # The shape is centered. We can rotate the whole cluster.
        # We also allow the outer hexagon to rotate.
        # Variables: [global_rotation, outer_rotation]
        # Note: Inner hexagons are rigidly attached to the lattice, so they all rotate by the same amount.
        
        def obj_cluster(params):
            global_rot = params[0]
            outer_rot = params[1]
            
            # Rotate positions
            c = math.cos(global_rot)
            s = math.sin(global_rot)
            rot_mat = np.array([[c, -s], [s, c]])
            rotated_positions = init_positions @ rot_mat.T
            
            # Angles are all global_rot (in radians)
            angles_rad = np.full(N, global_rot)
            
            score = calc_min_outer_side(rotated_positions, angles_rad, outer_rot)
            return score
        
        x0 = np.array([0.0, 0.0])
        try:
            res = minimize(obj_cluster, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                # Check validity of this configuration
                global_rot = res.x[0]
                c = math.cos(global_rot)
                s = math.sin(global_rot)
                rot_mat = np.array([[c, -s], [s, c]])
                final_positions = init_positions @ rot_mat.T
                final_angles = np.full(N, math.degrees(global_rot))
                
                if is_valid_config(final_positions, final_angles):
                    best_score = res.fun
                    # Store params in the same format as the general optimizer: [x1, y1, ..., xN, yN, a1, ..., aN]
                    params_list = final_positions.flatten().tolist() + final_angles.tolist()
                    best_params = np.array(params_list)
        except Exception:
            pass

    # --- Part 2: General Optimization (Current Program Strategy) ---
    # If we found a good lattice-based start, use it. Otherwise use standard lattice.
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
        # Verify best_score is actually the calculated side length for best_params
        pos = best_params[:2*N].reshape(N, 2)
        ang = best_params[2*N:]
        # The best_score from lattice search was for specific outer rotation? 
        # Wait, calc_min_outer_side in obj_cluster used outer_rot.
        # But the final return interface fixes outer_angle to 0.0 in the current program's return logic?
        # "outer hexagon center (x, y), side length and rotation"
        # The task allows returning the outer rotation.
        # However, the current program returns 0.0 for outer rotation.
        # Let's stick to outer rotation 0 for consistency with the "current program" interface unless we change it.
        # If we allow outer rotation, we need to return it.
        # The prompt says: "Keep the function names and signatures of the seed program."
        # Seed program returns: inner, (0.0, 0.0), final_side, 0.0
        # So I must return outer rotation 0.0.
        # This means I should only consider configurations that fit in a hexagon with rotation 0.
        # My lattice search optimized over outer rotation. I need to re-evaluate the best lattice shape with outer rotation 0.
        
        # Let's re-run the lattice evaluation forcing outer rotation to 0.
        best_score_lattice = float('inf')
        best_params_lattice = None
        
        for shape in unique_shapes:
            if time.time() > deadline - 20: break
            init_positions = np.array(shape)
            
            def obj_cluster_fixed_outer(params):
                global_rot = params[0]
                c = math.cos(global_rot)
                s = math.sin(global_rot)
                rot_mat = np.array([[c, -s], [s, c]])
                rotated_positions = init_positions @ rot_mat.T
                angles_rad = np.full(N, global_rot)
                return calc_min_outer_side(rotated_positions, angles_rad, 0.0)
                
            x0_rot = np.array([0.0])
            try:
                res = minimize(obj_cluster_fixed_outer, x0_rot, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_score_lattice:
                    global_rot = res.x[0]
                    c = math.cos(global_rot)
                    s = math.sin(global_rot)
                    rot_mat = np.array([[c, -s], [s, c]])
                    final_positions = init_positions @ rot_mat.T
                    final_angles = np.full(N, math.degrees(global_rot))
                    if is_valid_config(final_positions, final_angles):
                        best_score_lattice = res.fun
                        params_list = final_positions.flatten().tolist() + final_angles.tolist()
                        best_params_lattice = np.array(params_list)
            except Exception:
                pass
        
        if best_params_lattice is not None and best_score_lattice < best_score:
            best_score = best_score_lattice
            best_params = best_params_lattice
            x0 = best_params
        else:
            # Use the general lattice start if lattice search didn't yield better
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
            init_score = calc_min_outer_side(init_positions, init_angles, 0.0)
            if init_score < best_score:
                best_score = init_score
                best_params = x0
                x0 = best_params

    # Now run the general Nelder-Mead optimization from the best starting point found so far
    # This allows individual rotations and translations to break the lattice rigidity if beneficial.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res.fun < best_score:
            new_positions = res.x[:2*N].reshape(N, 2)
            new_angles = res.x[2*N:]
            if is_valid_config(new_positions, new_angles):
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass

    # Local refinement on the best found
    if best_params is not None:
        current_best = best_score
        current_params = best_params
        
        for _ in range(10):
            if time.time() > deadline - 10:
                break
                
            noise_pos = np.random.normal(0, 0.1, 2*N)
            noise_ang = np.random.normal(0, 0.1, N)
            x_perturbed = current_params.copy()
            x_perturbed[:2*N] += noise_pos
            x_perturbed[2*N:] += noise_ang
            
            try:
                res = minimize(
                    objective, 
                    x_perturbed, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
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

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        # Convert radians back to degrees for the return interface
        angle_deg = math.degrees(final_angles[i])
        inner.append([final_positions[i][0], final_positions[i][1], angle_deg])
    
    return inner, (0.0, 0.0), final_side, 0.0
