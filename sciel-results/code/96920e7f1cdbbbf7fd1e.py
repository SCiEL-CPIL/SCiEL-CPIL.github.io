import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize, basinhopping

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return min_overlap

    def compute_min_side(hexes, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, orientation outer_angle)
        that contains all given hexagons.
        """
        # Normals for a hexagon rotated by outer_angle.
        # Base normals for angle 0 are at 30, 90, 150, 210, 270, 330 degrees.
        base_angles = np.radians([30, 90, 150, 210, 270, 330])
        normal_angles = base_angles + outer_angle
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def get_hexes_from_params(params):
        """
        params: [r1, theta1, alpha1, r2, theta2, alpha2, outer_angle]
        Returns list of 12 hexagons (6x2 arrays).
        """
        r1, th1, al1, r2, th2, al2 = params[:6]
        
        hexes = []
        # Orbit 1
        for k in range(6):
            angle = np.radians(60 * k)
            # Position
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            # Rotation: base rotation + angle of position
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
            
        # Orbit 2
        for k in range(6):
            angle = np.radians(60 * k)
            # Position
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            # Rotation
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
            
        return hexes

    def compute_objective(params):
        """
        Objective: min_side + penalty for overlaps.
        params: [r1, theta1, alpha1, r2, theta2, alpha2, outer_angle]
        """
        if time.time() > deadline:
            return 1e10
            
        outer_angle = params[6]
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side(hexes, outer_angle)
        
        penalty = 0.0
        # Check overlaps
        # Precompute bounding boxes for speed
        bboxes = []
        for h in hexes:
            min_box = h.min(axis=0)
            max_box = h.max(axis=0)
            bboxes.append((min_box, max_box))
            
        for i in range(12):
            for j in range(i + 1, 12):
                min1, max1 = bboxes[i]
                min2, max2 = bboxes[j]
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
                    
        return s_min + penalty

    def compute_smooth_objective(params):
        """
        Smooth objective for global search.
        """
        if time.time() > deadline:
            return 1e10
            
        outer_angle = params[6]
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side(hexes, outer_angle)
        
        penalty = 0.0
        bboxes = []
        for h in hexes:
            min_box = h.min(axis=0)
            max_box = h.max(axis=0)
            bboxes.append((min_box, max_box))
            
        for i in range(12):
            for j in range(i + 1, 12):
                min1, max1 = bboxes[i]
                min2, max2 = bboxes[j]
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > -0.1:
                    # Softplus penalty
                    penalty += np.log(1.0 + np.exp(10.0 * overlap)) / 10.0
                    
        return s_min + penalty * 5.0

    # --- Optimization ---
    
    # Bounds
    # r: 0 to 5
    # theta: 0 to 2pi
    # alpha: 0 to pi/3 (60 deg, due to hex symmetry)
    # outer_angle: 0 to pi/3 (60 deg symmetry of outer hexagon)
    bounds = [
        (0.5, 5.0),  # r1
        (0.0, 2*np.pi), # theta1
        (0.0, np.pi/3), # alpha1
        (0.5, 5.0),  # r2
        (0.0, 2*np.pi), # theta2
        (0.0, np.pi/3), # alpha2
        (0.0, np.pi/3), # outer_angle
    ]
    
    best_s = np.inf
    best_params = None
    
    # Initial guesses based on honeycomb-like arrangements
    # r, theta, alpha
    initial_guesses = [
        [2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [1.5, 0.5, 0.0, 2.5, -0.5, np.pi/6, 0.0],
        [2.5, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0],
        [2.0, 0.0, np.pi/6, 3.0, 0.0, np.pi/6, 0.0],
        [1.8, 0.3, 0.0, 2.2, -0.3, 0.0, 0.0],
    ]
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        try:
            val = compute_objective(guess)
            if val < best_s:
                best_s = val
                best_params = np.array(guess)
        except Exception:
            pass
    
    # Phase 1: Global search with Differential Evolution using smooth objective
    if time.time() < deadline - 20:
        try:
            res = differential_evolution(
                compute_smooth_objective,
                bounds,
                maxiter=300,
                popsize=20,
                tol=1e-5,
                seed=42,
                polish=False,
                updating='deferred',
                init='sobol'
            )
            
            # Convert smooth result to exact objective
            exact_val = compute_objective(res.x)
            if exact_val < best_s:
                best_s = exact_val
                best_params = res.x
        except Exception:
            pass

    # Phase 2: Local polishing with L-BFGS-B on smooth objective
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_local = minimize(
                compute_smooth_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-8}
            )
            
            exact_val = compute_objective(res_local.x)
            if exact_val < best_s:
                best_s = exact_val
                best_params = res_local.x
        except Exception:
            pass

    # Phase 3: Multi-start local optimization with exact objective
    if time.time() < deadline - 10:
        np.random.seed(123)
        n_restarts = 15
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization near best solution or from initial guesses
            if best_params is not None and restart < 8:
                p = best_params + np.random.normal(0, 0.3, size=7)
            else:
                p = np.array([
                    np.random.uniform(1.0, 3.5),
                    np.random.uniform(0, 2*np.pi),
                    np.random.uniform(0, np.pi/3),
                    np.random.uniform(1.5, 4.0),
                    np.random.uniform(0, 2*np.pi),
                    np.random.uniform(0, np.pi/3),
                    np.random.uniform(0, np.pi/3)
                ])
            
            # Clip to bounds
            for i in range(7):
                p[i] = np.clip(p[i], bounds[i][0], bounds[i][1])
            
            try:
                val = compute_objective(p)
                if val < best_s:
                    best_s = val
                    best_params = p.copy()
            except Exception:
                continue
            
            try:
                res_local = minimize(
                    compute_objective,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass

    # Phase 4: Fine-tuning with very small steps
    if best_params is not None and time.time() < deadline - 2:
        current_params = best_params.copy()
        step_size = 0.01
        for _ in range(20):
            if time.time() > deadline - 1:
                break
                
            improved = False
            for i in range(7):
                for direction in [-1, 1]:
                    new_params = current_params.copy()
                    new_params[i] += direction * step_size
                    new_params[i] = np.clip(new_params[i], bounds[i][0], bounds[i][1])
                    
                    try:
                        val = compute_objective(new_params)
                        if val < best_s:
                            best_s = val
                            best_params = new_params.copy()
                            current_params = new_params.copy()
                            improved = True
                    except Exception:
                        pass
            
            if not improved:
                step_size *= 0.5
                if step_size < 0.001:
                    break
    
    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple known configuration
        best_params = np.array([2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        best_s = compute_objective(best_params)
    
    r1, th1, al1, r2, th2, al2, outer_angle = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        
        # Orbit 1
        x1 = r1 * np.cos(th1 + angle)
        y1 = r1 * np.sin(th1 + angle)
        a1 = al1 + angle
        inner.append([float(x1), float(y1), float(np.degrees(a1))])
        
        # Orbit 2
        x2 = r2 * np.cos(th2 + angle)
        y2 = r2 * np.sin(th2 + angle)
        a2 = al2 + angle
        inner.append([float(x2), float(y2), float(np.degrees(a2))])
        
    outer_center = (0.0, 0.0)
    outer_angle_deg = float(np.degrees(outer_angle))
    outer_side = float(best_s) # best_s is the objective value, which is min_side + penalty. 
                               # If penalty is 0, it is the side length.
                               # We should recompute the side length to be sure.
    
    # Recompute side length for accuracy
    hexes = get_hexes_from_params(best_params)
    final_side = compute_min_side(hexes, outer_angle)
    
    return inner, outer_center, float(final_side), outer_angle_deg
