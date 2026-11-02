import numpy as np
import time
import math

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

    def compute_min_side(hexes):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons.
        """
        # The outer hexagon has fixed orientation (angle 0).
        # Its supporting lines are defined by the 6 outward normals of a hexagon with side 1, angle 0.
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # For a regular hexagon of side s, the distance from center to side is s * sqrt(3) / 2.
        # So max_proj = s * sqrt(3) / 2
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_objective_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate the 12 hexagons
        # Base hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # Compute minimum outer side length
        min_s = compute_min_side(hexes)
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
                    
        return min_s + penalty

    def compute_objective_smooth(params):
        """
        Smooth objective for gradient-based optimization.
        Uses softplus approximation for overlap penalty.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        min_s = compute_min_side(hexes)
        
        penalty = 0.0
        beta = 30.0  # Increased softness parameter for sharper penalty
        
        # Precompute bounding boxes for quick rejection
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
                if overlap > -0.05:
                    # Softplus: log(1 + exp(beta * overlap)) / beta
                    penalty += np.log(1.0 + np.exp(beta * overlap)) / beta
                    
        return min_s + penalty * 8.0

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Initial guesses based on honeycomb-like arrangements
    initial_guesses = [
        # Guess 1: Two hexagons in a ring
        [2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        # Guess 2: Offset arrangement
        [1.5, 0.5, 0.0, 2.5, -0.5, np.pi/6],
        # Guess 3: More spread out
        [2.5, 0.0, 0.0, 3.5, 0.0, 0.0],
        # Guess 4: Rotated
        [2.0, 0.0, np.pi/6, 3.0, 0.0, np.pi/6],
        # Guess 5: Compact
        [1.8, 0.3, 0.0, 2.2, -0.3, 0.0],
    ]
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        try:
            val = compute_objective_symmetric(guess)
            if val < best_s:
                best_s = val
                best_params = np.array(guess)
        except Exception:
            pass
    
    # Phase 1: Global search with Differential Evolution using smooth objective
    try:
        from scipy.optimize import differential_evolution, minimize
        
        res = differential_evolution(
            compute_objective_smooth,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False,
            updating='deferred',
            init='sobol'
        )
        
        # Convert smooth result to exact objective
        exact_val = compute_objective_symmetric(res.x)
        if exact_val < best_s:
            best_s = exact_val
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Phase 2: Local polishing with L-BFGS-B on smooth objective
    if best_params is not None:
        try:
            res_local = minimize(
                compute_objective_smooth,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-10}
            )
            
            exact_val = compute_objective_symmetric(res_local.x)
            if exact_val < best_s:
                best_s = exact_val
                best_params = res_local.x
        except Exception:
            pass
    
    # Phase 3: Multi-start local optimization
    if time.time() < deadline - 20:
        np.random.seed(123)
        for restart in range(30):
            if time.time() > deadline - 10:
                break
                
            # Random initialization near best solution or from initial guesses
            if best_params is not None and restart < 15:
                p = best_params + np.random.normal(0, 0.2, size=6)
            else:
                p = np.array([
                    np.random.uniform(1.0, 3.5),
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(0, np.pi/3),
                    np.random.uniform(1.5, 4.0),
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(0, np.pi/3)
                ])
            
            # Clip to bounds
            for i in range(6):
                p[i] = np.clip(p[i], bounds[i][0], bounds[i][1])
            
            try:
                val = compute_objective_symmetric(p)
                if val < best_s:
                    best_s = val
                    best_params = p.copy()
            except Exception:
                continue
            
            try:
                res_local = minimize(
                    compute_objective_smooth,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-15}
                )
                exact_val = compute_objective_symmetric(res_local.x)
                if exact_val < best_s:
                    best_s = exact_val
                    best_params = res_local.x
            except Exception:
                pass
    
    # Phase 4: Fine-tuning with very small steps
    if best_params is not None and time.time() < deadline - 5:
        current_params = best_params.copy()
        step_size = 0.005
        for _ in range(80):
            if time.time() > deadline - 3:
                break
                
            improved = False
            for i in range(6):
                for direction in [-1, 1]:
                    new_params = current_params.copy()
                    new_params[i] += direction * step_size
                    new_params[i] = np.clip(new_params[i], bounds[i][0], bounds[i][1])
                    
                    try:
                        val = compute_objective_symmetric(new_params)
                        if val < best_s:
                            best_s = val
                            best_params = new_params.copy()
                            current_params = new_params.copy()
                            improved = True
                    except Exception:
                        pass
            
            if not improved:
                step_size *= 0.5
                if step_size < 0.0005:
                    break
    
    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple known configuration
        best_params = np.array([2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        best_s = compute_objective_symmetric(best_params)
    
    x1, y1, a1, x2, y2, a2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        # Rotate (x1, y1)
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        # Rotate (x2, y2)
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([float(x1_rot), float(y1_rot), float(np.degrees(a1_rot))])
        inner.append([float(x2_rot), float(y2_rot), float(np.degrees(a2_rot))])
        
    # Compute the actual minimum outer side length
    base1 = hex_vertices(x1, y1, 1.0, a1)
    base2 = hex_vertices(x2, y2, 1.0, a2)
    
    hexes = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        rot = np.array([[c, -s_], [s_, c]])
        
        h1 = base1 @ rot.T
        h2 = base2 @ rot.T
        
        hexes.append(h1)
        hexes.append(h2)
    
    outer_side = compute_min_side(hexes)
    
    outer_center = (0.0, 0.0)
    outer_angle_deg = 0.0
    
    return inner, outer_center, float(outer_side), float(outer_angle_deg)
