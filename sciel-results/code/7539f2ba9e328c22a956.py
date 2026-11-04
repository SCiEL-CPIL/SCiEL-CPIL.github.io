"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0

    # Hexagon geometry constants
    # Side length of inner hexagons = 1.0
    # Distance between centers of adjacent hexagons in a honeycomb lattice
    d = math.sqrt(3)

    # Generate centers for 1, 6, 12 rings (standard honeycomb packing)
    centers = [(0.0, 0.0)]

    # Ring 1: 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))

    # Ring 2: 12 neighbors
    # Positions are at distance 2*d from center? No.
    # In a hex grid, ring 2 has 12 hexagons.
    # They are located at (2,0), (1,1), (0,2), etc in axial coords.
    # Let's generate them properly.
    # Axial coords (q, r).
    # Center (0,0).
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) ? 
    # Standard axial to Cartesian:
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r

    ring2_centers = []
    # Iterate q, r such that max(|q|, |r|, |q+r|) == 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                # Cartesian
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))

    # ring2_centers should have 12 items
    # Verify
    if len(ring2_centers) != 12:
        # Fallback if generation is wrong, but logic is standard
        pass

    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    best_tx = 0.0
    best_ty = 0.0

    # Precompute vertices for a unit hexagon (pointy-top, vertices at 0, 60, ...)
    # Wait, the standard hexagon in these problems usually has vertices at 0, 60...
    # Let's check the seed. Seed uses angle 0.0.
    # If angle is 0, and we use standard cos/sin, vertex is at (1,0).
    # This is a "pointy-right" or "flat-top" depending on definition.
    # Let's stick to vertices at 0, 60, 120, 180, 240, 300 degrees.
    verts_rel = []
    for i in range(6):
        angle = i * math.pi / 3
        vx = math.cos(angle)
        vy = math.sin(angle)
        verts_rel.append((vx, vy))

    verts_rel = np.array(verts_rel)

    # Precompute all combinations of 4 from 12
    all_combinations = list(itertools.combinations(range(12), 4))

    # Precompute all possible sets of 11 centers? No, depends on combo.
    # But we can precompute the base 7 centers as array.
    base_centers = np.array(centers[:7])  # (7, 2)

    # Optimization function for a given set of centers (11, 2)
    # Objective: minimize S = (2/sqrt(3)) * max_k max_v |(v_rot + t) . n_k|
    # Parameters: (theta, tx, ty)
    
    def make_objective(centers_arr):
        # centers_arr: (11, 2)
        # Precompute all vertices for this configuration at theta=0, t=0
        # vertices = centers + verts_rel
        # Shape: (11, 6, 2)
        # Expand centers: (11, 1, 2) + (1, 6, 2)
        verts_all = centers_arr[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]  # (11, 6, 2)
        verts_all_flat = verts_all.reshape(-1, 2)  # (66, 2)

        def objective(params):
            theta, tx, ty = params
            c = math.cos(theta)
            s = math.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            
            # Rotate all vertices
            verts_rot = verts_all_flat @ rot.T  # (66, 2)
            
            # Translate
            verts_trans = verts_rot + np.array([tx, ty])
            
            # Normals of the outer hexagon (fixed orientation)
            # Normals at 0, 60, 120, 180, 240, 300 degrees
            # Note: The outer hexagon side length S is determined by the distance from center to facet.
            # If normals are n_k, the constraint is |p . n_k| <= S * sqrt(3)/2.
            # So S = (2/sqrt(3)) * max_k max_p |p . n_k|.
            
            # Precomputed normals: (6, 2)
            # n_k = (cos(k*pi/3), sin(k*pi/3))
            k_angles = np.arange(6) * math.pi / 3
            normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)  # (6, 2)
            
            # Dots: (66, 6)
            dots = verts_trans @ normals.T
            abs_dots = np.abs(dots)
            
            # Max over points for each normal
            max_per_normal = np.max(abs_dots, axis=0)  # (6,)
            
            # Max over normals
            max_proj = np.max(max_per_normal)
            
            S = (2.0 / math.sqrt(3)) * max_proj
            return S

        return objective

    # Loop over combinations
    for i, combo in enumerate(all_combinations):
        if time.time() > deadline - safety_margin:
            break

        # Construct centers
        combo_centers = np.array([ring2_centers[idx] for idx in combo])
        current_centers = np.vstack([base_centers, combo_centers])  # (11, 2)

        objective = make_objective(current_centers)

        # Initial guess
        x0 = [0.0, 0.0, 0.0]
        
        # Try a few different initial angles to avoid local minima?
        # Or just use Nelder-Mead which is robust.
        # Let's try initial theta = 0, but also maybe 10 degrees?
        # Nelder-Mead is sensitive to initial simplex.
        # We can try multiple starts if time permits.
        
        # For now, single start at (0,0,0)
        # But we can slightly perturb to help if stuck?
        # Nelder-Mead handles this usually.
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000, 'maxfev': 2000})
            if res.success:
                theta, tx, ty = res.x
                S = res.fun
                if S < best_S:
                    best_S = S
                    best_config = current_centers.copy()
                    best_theta = theta
                    best_tx = tx
                    best_ty = ty
        except Exception:
            pass

    # If no valid config found (shouldn't happen), fallback
    if best_config is None:
        # Just return a large valid config
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Final refinement?
    # We already used Nelder-Mead with tight tolerances.
    # We can do a tiny grid search around the best point if time permits.
    if time.time() < deadline - safety_margin:
        # Reconstruct objective for best config
        objective_best = make_objective(best_config)
        
        # Small perturbation search
        # theta +/- 0.01, tx +/- 0.01, ty +/- 0.01
        # Just a few random samples?
        # Or a finer Nelder-Mead restart?
        # Let's try a few random restarts around best solution
        for _ in range(50):
            if time.time() > deadline - safety_margin:
                break
            noise = 0.01
            x0_new = [best_theta + np.random.uniform(-noise, noise),
                      best_tx + np.random.uniform(-noise, noise),
                      best_ty + np.random.uniform(-noise, noise)]
            try:
                res = minimize(objective_best, x0_new, method='Nelder-Mead',
                               options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500, 'maxfev': 1000})
                if res.success and res.fun < best_S:
                    best_S = res.fun
                    best_theta, best_tx, best_ty = res.x
            except Exception:
                pass

    # Format output
    # inner: list of [x, y, angle]
    # The inner hexagons are at best_config (centers) relative to the local cluster frame.
    # But we rotated the cluster by best_theta and translated by (best_tx, best_ty).
    # So the final position of center i is:
    # p_i_final = R(best_theta) * p_i_initial + (best_tx, best_ty)
    
    c = math.cos(best_theta)
    s = math.sin(best_theta)
    rot_mat = np.array([[c, -s], [s, c]])
    
    inner = []
    for i in range(len(best_config)):
        cx, cy = best_config[i]
        # Rotate
        x_rot = c * cx - s * cy
        y_rot = s * cx + c * cy
        # Translate
        x_final = x_rot + best_tx
        y_final = y_rot + best_ty
        
        # The inner hexagon orientation is also rotated by best_theta
        # Angle in degrees
        angle_deg = math.degrees(best_theta)
        
        inner.append([x_final, y_final, angle_deg])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
