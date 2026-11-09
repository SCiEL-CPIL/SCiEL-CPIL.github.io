# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Assume the optimal configuration has high symmetry (e.g., D3h, D6h, or C4v).
    2. Parameterize 14 points using symmetry groups:
       - Option A (D3h-like): 2 poles + 6 ring1 + 6 ring2
       - Option B (D6h-like): 2 poles + 12 ring (split or single)
       - Option C (C4v-like): 2 poles + 4 ring1 + 4 ring2 + 4 ring3
    3. Optimize the small set of parameters (radii, z-heights, angles) to maximize the metric.
    4. Use Differential Evolution for global search in the low-dimensional parameter space.
    5. Refine with Nelder-Mead for high precision.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin

    def evaluate_config(points):
        """
        Takes an array of shape (14, 3), centers, normalizes, and returns the metric.
        Returns (points_normalized, metric_value).
        """
        if not np.all(np.isfinite(points)):
            return None, 0.0
            
        pts = points.copy()
        # Center
        pts = pts - np.mean(pts, axis=0)
        
        # Check for coincident points (d_min = 0)
        dists = pdist(pts)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return None, 0.0
            
        d_min = np.min(dists)
        
        # Normalize to unit max distance
        pts_norm = pts / d_max
        
        ratio = d_min / d_max
        metric = ratio ** 2
        
        return pts_norm, metric

    def objective_de(params):
        """
        Objective function for Differential Evolution.
        Returns negative metric (to minimize).
        """
        if time.time() > deadline - 10:
            return 0.0
        
        try:
            # params structure depends on the symmetry model being tested
            # We will test multiple models and keep the best.
            # For simplicity in DE, we will define a specific model here.
            # Let's use a flexible model:
            # 2 poles on Z-axis: (0, 0, z1), (0, 0, -z1)
            # 6 points in Ring 1 at z=z2: radius r2, angles 0, 60, ..., 300
            # 6 points in Ring 2 at z=z3: radius r3, angles offset by phi, 0+phi, ...
            
            # This covers D3h if phi=0 or 30, and general D6h-like if z2=z3.
            # It also allows for chiral twists.
            
            z1 = params[0]
            z2 = params[1]
            r2 = params[2]
            z3 = params[3]
            r3 = params[4]
            phi = params[5] # offset for ring 2
            
            # Ensure radii are non-negative
            r2 = abs(r2)
            r3 = abs(r3)
            
            points = np.zeros((14, 3))
            
            # Poles
            points[0] = [0, 0, z1]
            points[1] = [0, 0, -z1]
            
            # Ring 1 (6 points)
            angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
            for i in range(6):
                points[2+i] = [r2 * np.cos(angles1[i]), r2 * np.sin(angles1[i]), z2]
                
            # Ring 2 (6 points)
            angles2 = angles1 + phi
            for i in range(6):
                points[8+i] = [r3 * np.cos(angles2[i]), r3 * np.sin(angles2[i]), z3]
                
            _, metric = evaluate_config(points)
            return -metric
        except Exception:
            return 0.0

    def objective_de_model_b(params):
        """
        Alternative Model: 
        2 poles
        4 points in Ring 1 (square, z=z2, radius r2)
        4 points in Ring 2 (square, z=z3, radius r3, offset by 45 deg)
        4 points in Ring 3 (square, z=z4, radius r4)
        Total: 2+4+4+4 = 14.
        This tests C4v symmetry.
        """
        if time.time() > deadline - 10:
            return 0.0
        
        try:
            z1 = params[0]
            z2 = params[1]
            r2 = params[2]
            z3 = params[3]
            r3 = params[4]
            z4 = params[5]
            r4 = params[6]
            
            r2 = abs(r2)
            r3 = abs(r3)
            r4 = abs(r4)
            
            points = np.zeros((14, 3))
            
            # Poles
            points[0] = [0, 0, z1]
            points[1] = [0, 0, -z1]
            
            # Ring 1 (4 points, square)
            angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
            for i in range(4):
                points[2+i] = [r2 * np.cos(angles1[i]), r2 * np.sin(angles1[i]), z2]
                
            # Ring 2 (4 points, square, offset 45 deg)
            angles2 = angles1 + np.pi/4
            for i in range(4):
                points[6+i] = [r3 * np.cos(angles2[i]), r3 * np.sin(angles2[i]), z3]
                
            # Ring 3 (4 points, square)
            angles3 = angles1
            for i in range(4):
                points[10+i] = [r4 * np.cos(angles3[i]), r4 * np.sin(angles3[i]), z4]
                
            _, metric = evaluate_config(points)
            return -metric
        except Exception:
            return 0.0

    best_points = None
    best_score = 0.0

    # --- Search Model A (D3h/D6h-like) ---
    if time.time() < deadline - 60:
        try:
            # Bounds: 
            # z1 (pole height): [0, 2]
            # z2, z3 (ring heights): [-2, 2]
            # r2, r3 (radii): [0, 2]
            # phi (offset): [0, pi/3] (symmetry period is 60 deg for 6-ring)
            bounds_a = [
                (0.0, 2.0),   # z1
                (-2.0, 2.0),  # z2
                (0.0, 2.0),   # r2
                (-2.0, 2.0),  # z3
                (0.0, 2.0),   # r3
                (0.0, np.pi/3) # phi
            ]
            
            res_a = differential_evolution(
                objective_de,
                bounds_a,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-10,
                polish=False,
                init='latinhypercube'
            )
            
            # Reconstruct points for best params
            params_best_a = res_a.x
            z1, z2, r2, z3, r3, phi = params_best_a
            r2 = abs(r2); r3 = abs(r3)
            
            pts_a = np.zeros((14, 3))
            pts_a[0] = [0, 0, z1]
            pts_a[1] = [0, 0, -z1]
            angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
            for i in range(6):
                pts_a[2+i] = [r2 * np.cos(angles1[i]), r2 * np.sin(angles1[i]), z2]
            angles2 = angles1 + phi
            for i in range(6):
                pts_a[8+i] = [r3 * np.cos(angles2[i]), r3 * np.sin(angles2[i]), z3]
                
            pts_norm_a, score_a = evaluate_config(pts_a)
            
            if score_a > best_score:
                best_score = score_a
                best_points = pts_norm_a
                
            # Refine with NM
            if time.time() < deadline - 30:
                # Wrap objective for NM to work with the same param structure
                def obj_nm_a(p):
                    return objective_de(p)
                
                res_nm_a = minimize(
                    obj_nm_a,
                    params_best_a,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                
                # Reconstruct points for refined params
                params_ref_a = res_nm_a.x
                z1, z2, r2, z3, r3, phi = params_ref_a
                r2 = abs(r2); r3 = abs(r3)
                
                pts_ref_a = np.zeros((14, 3))
                pts_ref_a[0] = [0, 0, z1]
                pts_ref_a[1] = [0, 0, -z1]
                angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
                for i in range(6):
                    pts_ref_a[2+i] = [r2 * np.cos(angles1[i]), r2 * np.sin(angles1[i]), z2]
                angles2 = angles1 + phi
                for i in range(6):
                    pts_ref_a[8+i] = [r3 * np.cos(angles2[i]), r3 * np.sin(angles2[i]), z3]
                    
                pts_norm_ref_a, score_ref_a = evaluate_config(pts_ref_a)
                
                if score_ref_a > best_score:
                    best_score = score_ref_a
                    best_points = pts_norm_ref_a
                    
        except Exception:
            pass

    # --- Search Model B (C4v-like) ---
    if time.time() < deadline - 60:
        try:
            bounds_b = [
                (0.0, 2.0),   # z1
                (-2.0, 2.0),  # z2
                (0.0, 2.0),   # r2
                (-2.0, 2.0),  # z3
                (0.0, 2.0),   # r3
                (-2.0, 2.0),  # z4
                (0.0, 2.0)    # r4
            ]
            
            res_b = differential_evolution(
                objective_de_model_b,
                bounds_b,
                seed=123,
                maxiter=200,
                popsize=20,
                tol=1e-10,
                polish=False,
                init='latinhypercube'
            )
            
            params_best_b = res_b.x
            z1, z2, r2, z3, r3, z4, r4 = params_best_b
            r2 = abs(r2); r3 = abs(r3); r4 = abs(r4)
            
            pts_b = np.zeros((14, 3))
            pts_b[0] = [0, 0, z1]
            pts_b[1] = [0, 0, -z1]
            angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
            for i in range(4):
                pts_b[2+i] = [r2 * np.cos(angles1[i]), r2 * np.sin(angles1[i]), z2]
            angles2 = angles1 + np.pi/4
            for i in range(4):
                pts_b[6+i] = [r3 * np.cos(angles2[i]), r3 * np.sin(angles2[i]), z3]
            angles3 = angles1
            for i in range(4):
                pts_b[10+i] = [r4 * np.cos(angles3[i]), r4 * np.sin(angles3[i]), z4]
                
            pts_norm_b, score_b = evaluate_config(pts_b)
            
            if score_b > best_score:
                best_score = score_b
                best_points = pts_norm_b
                
            # Refine with NM
            if time.time() < deadline - 30:
                def obj_nm_b(p):
                    return objective_de_model_b(p)
                
                res_nm_b = minimize(
                    obj_nm_b,
                    params_best_b,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                
                params_ref_b = res_nm_b.x
                z1, z2, r2, z3, r3, z4, r4 = params_ref_b
                r2 = abs(r2); r3 = abs(r3); r4 = abs(r4)
                
                pts_ref_b = np.zeros((14, 3))
                pts_ref_b[0] = [0, 0, z1]
                pts_ref_b[1] = [0, 0, -z1]
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                for i in range(4):
                    pts_ref_b[2+i] = [r2 * np.cos(angles1[i]), r2 * np.sin(angles1[i]), z2]
                angles2 = angles1 + np.pi/4
                for i in range(4):
                    pts_ref_b[6+i] = [r3 * np.cos(angles2[i]), r3 * np.sin(angles2[i]), z3]
                angles3 = angles1
                for i in range(4):
                    pts_ref_b[10+i] = [r4 * np.cos(angles3[i]), r4 * np.sin(angles3[i]), z4]
                    
                pts_norm_ref_b, score_ref_b = evaluate_config(pts_ref_b)
                
                if score_ref_b > best_score:
                    best_score = score_ref_b
                    best_points = pts_norm_ref_b
                    
        except Exception:
            pass

    # Fallback if no valid points found
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Simple fallback: Icosahedron + 2 poles (normalized)
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden],
            [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0],
            [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1],
            [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
        pole1 = np.array([0, 0, 1])
        pole2 = np.array([0, 0, -1])
        points = np.vstack([ico_verts, pole1, pole2])
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 1e-12:
            points = points / d_max
        best_points = points

    return best_points

# EVOLVE-BLOCK-END
