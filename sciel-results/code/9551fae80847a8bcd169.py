import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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

    def smooth_max(values, temperature=0.01):
        """
        Smooth approximation of max using log-sum-exp.
        """
        if len(values) == 0:
            return 0.0
        v_max = np.max(values)
        if v_max == -np.inf:
            return -np.inf
        shifted = (values - v_max) / temperature
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        result = v_max + temperature * np.log(np.sum(np.exp(shifted)))
        return result

    def compute_min_side(hexes):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons.
        """
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_min_side_smooth(hexes, temperature=0.01):
        """
        Smooth version of compute_min_side using log-sum-exp.
        """
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Collect all projections
        all_projs = []
        for h in hexes:
            proj = h @ normals.T  # shape (6, 6)
            all_projs.append(proj)
        
        all_projs = np.concatenate(all_projs, axis=0)  # shape (72, 6)
        
        # For each of the 6 normals, compute smooth max of projections
        smooth_maxes = np.array([smooth_max(all_projs[:, i], temperature) for i in range(6)])
        
        # The side length is determined by the maximum of the 6 smooth maxes
        max_smooth = smooth_max(smooth_maxes, temperature)
        
        return max_smooth * 2.0 / np.sqrt(3.0)

    def compute_objective_symmetric(params, temperature=0.01, penalty_weight=50.0):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate the 12 hexagons
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
        
        # Compute minimum outer side length (smooth)
        min_s = compute_min_side_smooth(hexes, temperature)
        
        penalty = 0.0
        
        # Check overlaps using smooth approximation
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
                    # Smooth penalty: use softplus-like function
                    penalty += penalty_weight * (np.log1p(np.exp(overlap * 10)) / 10)
                    
        return min_s + penalty

    def compute_objective_exact(params):
        """
        Exact objective function for final evaluation.
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
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 100.0
                    
        return min_s + penalty

    # --- Optimization ---
    
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
    
    # Phase 1: Coarse global search with smooth objective
    try:
        res = differential_evolution(
            lambda p: compute_objective_symmetric(p, temperature=0.05, penalty_weight=20.0),
            bounds,
            maxiter=200,
            popsize=30,
            tol=1e-4,
            seed=42,
            polish=False
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
            
    except Exception as e:
        pass
    
    # Phase 2: Medium refinement with tighter smooth objective
    if best_params is not None and time.time() < deadline - 60:
        try:
            res2 = differential_evolution(
                lambda p: compute_objective_symmetric(p, temperature=0.01, penalty_weight=50.0),
                bounds,
                x0=best_params,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                seed=43,
                polish=False
            )
            
            if res2.fun < best_s:
                best_s = res2.fun
                best_params = res2.x.copy()
                
        except Exception as e:
            pass
    
    # Phase 3: Local polish with L-BFGS-B
    if best_params is not None and time.time() < deadline - 30:
        try:
            res_local = minimize(
                lambda p: compute_objective_symmetric(p, temperature=0.005, penalty_weight=100.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final exact evaluation and tiny adjustments
    if best_params is not None and time.time() < deadline - 10:
        # Try small perturbations around the best solution
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            perturbation = np.random.normal(0, 0.01, size=6)
            candidate = best_params + perturbation
            # Clip to bounds
            for i in range(6):
                candidate[i] = np.clip(candidate[i], bounds[i][0], bounds[i][1])
            
            val = compute_objective_exact(candidate)
            if val < best_s:
                best_s = val
                best_params = candidate.copy()
                
                # Try to polish further
                try:
                    res_fine = minimize(
                        lambda p: compute_objective_symmetric(p, temperature=0.001, penalty_weight=200.0),
                        best_params,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-16}
                    )
                    if res_fine.fun < best_s:
                        best_s = res_fine.fun
                        best_params = res_fine.x.copy()
                except Exception:
                    pass
    
    # Fallback if no good solution found
    if best_params is None:
        # Try several random starts
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            p = np.array([
                np.random.uniform(1.5, 3.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/6),
                np.random.uniform(2.5, 4.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/6)
            ])
            val = compute_objective_exact(p)
            if val < best_s:
                best_s = val
                best_params = p.copy()
                
                try:
                    res_local = minimize(
                        lambda p: compute_objective_symmetric(p, temperature=0.01, penalty_weight=50.0),
                        p,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200}
                    )
                    if res_local.fun < best_s:
                        best_s = res_local.fun
                        best_params = res_local.x.copy()
                except Exception:
                    pass
    
    x1, y1, a1, x2, y2, a2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
        inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
        
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
    
    return inner, outer_center, outer_side, outer_angle_deg
