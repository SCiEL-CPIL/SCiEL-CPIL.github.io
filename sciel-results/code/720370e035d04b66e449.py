import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    def sat_gap(poly1, poly2):
        """
        Compute the minimum gap (positive if separated, negative if overlapping)
        between two convex polygons using SAT.
        Returns the maximum separation (minimum gap) over all axes.
        If any axis has positive separation, they are disjoint.
        """
        min_gap = float('inf')
        
        # Axes from poly1
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap # Separated
        
        # Axes from poly2
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap # Separated
        
        return min_gap

    def is_valid_config(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective(params, penalty_weight=100.0):
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
        
        base_score = calc_min_outer_side(positions, angles, 0.0)
        return base_score + penalty

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
    for _ in range(30):
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

    # Evaluate each shape with optimized global rotation (outer rotation fixed to 0)
    for shape in unique_shapes:
        if time.time() > deadline - 20: break
        
        init_positions = np.array(shape)
        init_angles = np.zeros(N)
        
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
            if res.fun < best_score:
                global_rot = res.x[0]
                c = math.cos(global_rot)
                s = math.sin(global_rot)
                rot_mat = np.array([[c, -s], [s, c]])
                final_positions = init_positions @ rot_mat.T
                final_angles = np.full(N, global_rot)
                
                if is_valid_config(final_positions, final_angles):
                    best_score = res.fun
                    params_list = final_positions.flatten().tolist() + final_angles.tolist()
                    best_params = np.array(params_list)
        except Exception:
            pass

    # --- Part 2: General Optimization with Simulated Annealing ---
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

    # Now run Simulated Annealing from the best starting point found so far
    # This allows individual rotations and translations to break the lattice rigidity if beneficial.
    # We use a soft penalty for overlaps to guide the search.
    
    current_params = x0.copy()
    current_score = objective(current_params)
    
    # If initial config is invalid, we might need to start from a valid but larger config
    # But SA with penalty should handle it.
    
    if best_params is not None:
        current_best = best_score
        current_best_params = best_params.copy()
        
        T = 0.5  # initial temperature
        cooling_rate = 0.9995
        step_size = 0.3
        max_iter = 50000
        
        np.random.seed(42)
        random.seed(42)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Generate perturbation
            new_params = current_params.copy()
            
            # Randomly perturb 1-3 hexagons
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, N-1)
                # Perturb position
                new_params[2*idx] += np.random.randn() * step_size
                new_params[2*idx+1] += np.random.randn() * step_size
                # Perturb angle
                new_params[2*N + idx] += np.random.randn() * step_size * 0.5 # Radians
                
            new_score = objective(new_params)
            
            # Acceptance criterion
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
                # Check if valid and better than best
                if current_score < current_best:
                    new_positions = current_params[:2*N].reshape(N, 2)
                    new_angles = current_params[2*N:]
                    if is_valid_config(new_positions, new_angles):
                        current_best = calc_min_outer_side(new_positions, new_angles, 0.0)
                        current_best_params = current_params.copy()
            else:
                delta = new_score - current_score
                if T > 0:
                    acceptance_prob = math.exp(-delta / T)
                    if random.random() < acceptance_prob:
                        current_params = new_params
                        current_score = new_score
            
            # Cool down
            T *= cooling_rate
            
            # Reduce step size slowly
            if iteration % 500 == 0:
                step_size *= 0.95
        
        best_score = current_best
        best_params = current_best_params

    # Final local refinement using Nelder-Mead on the best found
    if best_params is not None:
        try:
            res = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_score:
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                if is_valid_config(new_positions, new_angles):
                    final_side = calc_min_outer_side(new_positions, new_angles, 0.0)
                    if final_side < best_score:
                        best_score = final_side
                        best_params = res.x
        except Exception:
            pass

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        # Convert radians back to degrees for the return interface
        angle_deg = math.degrees(final_angles[i])
        inner.append([final_positions[i][0], final_positions[i][1], angle_deg])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
