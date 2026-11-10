import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    # Constants
    sqrt3 = np.sqrt(3)

    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit hexagon at (cx, cy) with given angle."""
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def get_all_vertices(params):
        """
        params: (33,) array [x1, y1, a1, x2, y2, a2, ...]
        returns: (11, 6, 2) array of vertices
        """
        verts = np.zeros((11, 6, 2))
        for i in range(11):
            cx = params[i*3]
            cy = params[i*3 + 1]
            ang = params[i*3 + 2]
            verts[i] = get_hex_vertices(cx, cy, ang)
        return verts

    def sat_min_gap(v1, v2):
        """
        Compute minimum gap between two hexagons using SAT.
        Returns the minimum separation (positive = separated, negative = overlapping).
        """
        edges1 = np.roll(v1, -1, axis=0) - v1
        normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
        n1 = np.linalg.norm(normals1, axis=1, keepdims=True)
        normals1 /= n1
        
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
        n2 = np.linalg.norm(normals2, axis=1, keepdims=True)
        normals2 /= n2
        
        all_normals = np.vstack([normals1, normals2])
        
        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T
        
        min1 = np.min(proj1, axis=1)
        max1 = np.max(proj1, axis=1)
        min2 = np.min(proj2, axis=1)
        max2 = np.max(proj2, axis=1)
        
        gaps = np.maximum(min2 - max1, min1 - max2)
        return np.min(gaps)

    def exact_check(params):
        """
        Perform exact validation.
        Returns (is_valid, S, min_gap)
        """
        verts = get_all_vertices(params)
        
        # Compute required side length S
        # Outer hexagon centered at origin, angle 0
        # Vertices at angles 0, 60, ..., 300
        # Normals at 30, 90, ..., 330
        outer_normal_angles = np.deg2rad(np.arange(30, 360, 60))
        outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])
        
        flat_verts = verts.reshape(-1, 2)
        projections = flat_verts @ outer_normals.T
        max_proj = np.max(projections, axis=1)
        global_max = np.max(max_proj)
        S = 2.0 * global_max / sqrt3
        
        # Check separation with SAT
        min_gap = np.inf
        for i in range(11):
            v1 = verts[i]
            for j in range(i+1, 11):
                v2 = verts[j]
                gap = sat_min_gap(v1, v2)
                if gap < min_gap:
                    min_gap = gap
        
        if min_gap < 1e-6:
            return False, S, min_gap
            
        # Check edge lengths
        for i in range(11):
            for j in range(6):
                p1 = verts[i, j]
                p2 = verts[i, (j+1)%6]
                if abs(np.linalg.norm(p2-p1) - 1.0) > 1e-8:
                    return False, S, min_gap
                    
        return True, S, min_gap

    def compute_S(params):
        """Compute the side length of the outer hexagon needed to contain all inner hexagons."""
        verts = get_all_vertices(params)
        
        outer_normal_angles = np.deg2rad(np.arange(30, 360, 60))
        outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])
        
        flat_verts = verts.reshape(-1, 2)
        projections = flat_verts @ outer_normals.T
        max_proj = np.max(projections, axis=1)
        global_max = np.max(max_proj)
        return 2.0 * global_max / sqrt3

    def overlap_penalty(params, weight=1000.0):
        """
        Compute penalty for overlaps using SAT.
        """
        verts = get_all_vertices(params)
        penalty = 0.0
        for i in range(11):
            v1 = verts[i]
            for j in range(i+1, 11):
                v2 = verts[j]
                gap = sat_min_gap(v1, v2)
                if gap < 0:
                    penalty += weight * gap * gap
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        S = compute_S(params)
        pen = overlap_penalty(params)
        return S + pen

    # Initial Guess: Use a known good configuration
    # 11 hexagons: 1 center + 6 in ring + 4 more
    def get_initial_guess():
        p = np.zeros(33)
        # Center
        p[0:3] = [0, 0, 0]
        
        # Ring 1: 6 hexagons at distance sqrt(3)
        d1 = sqrt3
        for k in range(6):
            ang = np.deg2rad(k * 60)
            idx = (k+1)*3
            p[idx] = d1 * np.cos(ang)
            p[idx+1] = d1 * np.sin(ang)
            p[idx+2] = 0
        
        # Ring 2: 4 hexagons
        # Place them at positions that fit nicely
        # Using axial coordinates: (2,0), (0,2), (-2,0), (0,-2)
        # Convert to Cartesian: x = sqrt3*(q + r/2), y = 1.5*r
        positions_r2 = [
            (2*sqrt3, 0),
            (sqrt3, 3),
            (-2*sqrt3, 0),
            (-sqrt3, -3)
        ]
        
        for k in range(4):
            idx = (k+7)*3
            p[idx] = positions_r2[k][0]
            p[idx+1] = positions_r2[k][1]
            p[idx+2] = 0
            
        return p

    # Simulated Annealing
    def simulated_annealing(params0, max_iters=50000):
        """
        Perform simulated annealing to optimize the packing.
        """
        params = params0.copy()
        current_obj = objective(params)
        best_params = params.copy()
        best_obj = current_obj
        
        T = 0.1  # Initial temperature
        cooling_rate = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Generate perturbation
            new_params = params.copy()
            # Perturb positions and angles
            noise_scale = 0.05 * T
            new_params[0:33:3] += np.random.normal(0, noise_scale, 11)  # x
            new_params[1:33:3] += np.random.normal(0, noise_scale, 11)  # y
            new_params[2:33:3] += np.random.normal(0, noise_scale * 10, 11)  # angle
            
            new_obj = objective(new_params)
            
            # Accept or reject
            if new_obj < current_obj or np.random.random() < np.exp(-(new_obj - current_obj) / T):
                params = new_params
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_params = params.copy()
            
            T *= cooling_rate
            
            if T < 1e-6:
                break
                
        return best_params, best_obj

    # Get initial guess
    p0 = get_initial_guess()
    
    # Try multiple optimization strategies
    best_S = 1e10
    best_params = None
    best_valid = False

    # Strategy 1: Powell optimization
    if time.time() < deadline - 120:
        try:
            res = minimize(objective, p0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
            
            valid, S_val, gap_val = exact_check(res.x)
            if valid and S_val < best_S:
                best_S = S_val
                best_params = res.x.copy()
                best_valid = True
        except Exception as e:
            pass

    # Strategy 2: Simulated Annealing
    if time.time() < deadline - 60:
        try:
            sa_params, sa_obj = simulated_annealing(p0, max_iters=30000)
            if sa_obj < best_S:
                best_S = sa_obj
                best_params = sa_params.copy()
            
            valid, S_val, gap_val = exact_check(sa_params)
            if valid and S_val < best_S:
                best_S = S_val
                best_params = sa_params.copy()
                best_valid = True
        except Exception as e:
            pass

    # Strategy 3: Nelder-Mead on best candidate
    if best_params is not None and time.time() < deadline - 30:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
            
            valid, S_val, gap_val = exact_check(res.x)
            if valid and S_val < best_S:
                best_S = S_val
                best_params = res.x.copy()
                best_valid = True
        except Exception as e:
            pass

    # Fallback: Use initial guess if no valid solution found
    if best_params is None or not best_valid:
        p = p0
        valid, S_val, gap_val = exact_check(p)
        if valid:
            best_params = p
            best_S = S_val
            best_valid = True
        else:
            # Construct a very safe, large hexagon
            inner_hex_data = np.array(
                [
                    [0, 0, 0],
                    [-2.5, 0, 0],
                    [2.5, 0, 0],
                    [-1.25, 2.17, 0],
                    [1.25, 2.17, 0],
                    [-1.25, -2.17, 0],
                    [1.25, -2.17, 0],
                    [-3.75, 2.17, 0],
                    [3.75, 2.17, 0],
                    [-3.75, -2.17, 0],
                    [3.75, -2.17, 0],
                ]
            )
            outer_hex_data = np.array([0, 0, 0])
            outer_hex_side_length = 8.0
            return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Format output
    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
