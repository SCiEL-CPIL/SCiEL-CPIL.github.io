import numpy as np
import time
import itertools
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    eps = 1e-12
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    # Iterate all 15 combinations of 4 outer hexagons from 6 possible positions
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))
    
    best_score = float('inf')
    best_config = None
    best_theta_rel = 0.0
    best_centers = None
    best_outer_center = (0.0, 0.0)
    
    for combo in combinations:
        # Build centers
        centers = np.array([[0.0, 0.0]])
        
        # Inner ring (6) at 30, 90, ...
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
        # Outer ring (4) at chosen angles
        # Using radius 3.0 for the "corner" hexagons of the second shell
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
        def compute_side_rigid(theta_rel, c=centers):
            theta_out = theta_rel
            normals = np.array([
                [np.cos(np.deg2rad(30 + k * 60) + theta_out),
                 np.sin(np.deg2rad(30 + k * 60) + theta_out)]
                for k in range(6)
            ])
            
            # Compute projections of all vertices of all inner hexagons
            # We need to consider all vertices of all inner hexagons
            # There are 11 inner hexagons, each with 6 vertices.
            # Total 66 points.
            
            # Optimization: Pre-compute vertices for each inner hexagon?
            # No, theta changes normals, so we just dot product.
            
            # Stack all vertices
            # centers has shape (11, 2)
            # base_vertices has shape (6, 2)
            # We need all pairs (center_i, vertex_j)
            
            # Efficient way:
            # For each inner hexagon i, vertices are c_i + base_vertices
            # Projections: (c_i + base_vertices) @ normals.T
            # Shape: (11, 6, 6) -> (11, 6) if we take max/min over vertices?
            # No, we need global max/min over all points.
            
            # Let's compute max and min for each normal direction k
            max_projs = np.zeros(6)
            min_projs = np.zeros(6)
            
            for i in range(len(c)):
                verts = base_vertices + c[i]
                dots = verts @ normals.T  # Shape (6, 6)
                # dots[j, k] is projection of vertex j of hexagon i on normal k
                max_projs = np.maximum(max_projs, np.max(dots, axis=0))
                min_projs = np.minimum(min_projs, np.min(dots, axis=0))
            
            widths = max_projs - min_projs
            max_width = np.max(widths)
            side = max_width / np.sqrt(3)
            
            # Find optimal center
            # Constraints:
            # min_k + side*sqrt(3)/2 <= C . n_k <= max_k - side*sqrt(3)/2
            # Let L_k = min_k + side*sqrt(3)/2
            # Let R_k = max_k - side*sqrt(3)/2
            # We need C such that L_k <= C . n_k <= R_k for all k.
            
            L = min_projs + side * np.sqrt(3) / 2
            R = max_projs - side * np.sqrt(3) / 2
            
            # Find a point C satisfying these.
            # Since the problem is 2D, we can find the intersection of the half-planes.
            # Or, note that for the tightest constraint k*, L_k* == R_k*.
            # So C . n_k* is fixed.
            # Let's find the tightest constraint.
            tightest_idx = np.argmax(widths)
            n_tight = normals[tightest_idx]
            c_tight = L[tightest_idx] # == R[tightest_idx]
            
            # C lies on the line n_tight . C = c_tight.
            # Parametrize C:
            # n_tight = (nx, ny).
            # If nx != 0, C = (c_tight/nx - ny/nx * y, y).
            # Or use a basis vector orthogonal to n_tight.
            # v_perp = (-ny, nx).
            # C = (c_tight / ||n_tight||^2) * n_tight + t * v_perp
            # Since n_tight is unit vector, ||n_tight||^2 = 1.
            # C0 = c_tight * n_tight
            # C(t) = C0 + t * v_perp
            
            C0 = c_tight * n_tight
            v_perp = np.array([-n_tight[1], n_tight[0]])
            
            # Check other constraints for C(t)
            # L_k <= (C0 + t*v_perp) . n_k <= R_k
            # L_k <= C0 . n_k + t * (v_perp . n_k) <= R_k
            
            # Let A_k = v_perp . n_k
            # Let B_k = C0 . n_k
            # L_k - B_k <= t * A_k <= R_k - B_k
            
            # We need to find t such that all these hold.
            # This is a 1D interval intersection problem.
            t_min = -np.inf
            t_max = np.inf
            
            for k in range(6):
                if k == tightest_idx:
                    continue
                A = np.dot(v_perp, normals[k])
                B = np.dot(C0, normals[k])
                Lk = L[k]
                Rk = R[k]
                
                if np.abs(A) < 1e-9:
                    # If A is 0, then B must be in [Lk, Rk]
                    if B < Lk - 1e-9 or B > Rk + 1e-9:
                        return float('inf'), (0,0) # Infeasible? Should not happen if side is correct
                else:
                    # Lk - B <= t * A <= Rk - B
                    low = Lk - B
                    high = Rk - B
                    
                    if A > 0:
                        t_low = low / A
                        t_high = high / A
                    else:
                        t_low = high / A
                        t_high = low / A
                        
                    t_min = max(t_min, t_low)
                    t_max = min(t_max, t_high)
                    
                    if t_min > t_max + 1e-9:
                        return float('inf'), (0,0)
            
            # Pick midpoint of [t_min, t_max]
            if t_min == -np.inf:
                t_val = 0.0
            elif t_max == np.inf:
                t_val = 0.0
            else:
                t_val = (t_min + t_max) / 2.0
                
            C_opt = C0 + t_val * v_perp
            return side, (C_opt[0], C_opt[1])

        # High-resolution grid search to find good starting point
        thetas = np.linspace(0, np.pi/6, 200)
        local_best_score = float('inf')
        local_best_theta = 0.0
        local_best_center = (0.0, 0.0)
        
        for t in thetas:
            if time.time() > deadline - 5:
                break
            s, c = compute_side_rigid(t)
            if s < local_best_score:
                local_best_score = s
                local_best_theta = t
                local_best_center = c
        
        # Local refinement using minimize_scalar
        if time.time() < deadline - 2 and local_best_score < float('inf'):
            step = (np.pi/6) / 200
            bounds = (max(0, local_best_theta - 5*step), min(np.pi/6, local_best_theta + 5*step))
            
            def obj(t):
                s, _ = compute_side_rigid(t)
                return s
            
            res = minimize_scalar(obj, bounds=bounds, method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < local_best_score:
                local_best_score = res.fun
                local_best_theta = res.x
                _, local_best_center = compute_side_rigid(local_best_theta)

        if local_best_score < best_score:
            best_score = local_best_score
            best_config = combo
            best_theta_rel = local_best_theta
            best_centers = centers
            best_outer_center = local_best_center

    # Fallback
    if best_centers is None:
        combo = combinations[0]
        centers = np.array([[0.0, 0.0]])
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
        # Re-use compute_side_rigid logic? Just call it.
        # Need to define it again or make it external.
        # For simplicity, just return a valid but maybe suboptimal result.
        # Or just use the first combo's result if we stored it.
        # Let's just calculate it simply.
        def compute_side_rigid_fallback(theta_rel, c=centers):
            theta_out = theta_rel
            normals = np.array([
                [np.cos(np.deg2rad(30 + k * 60) + theta_out),
                 np.sin(np.deg2rad(30 + k * 60) + theta_out)]
                for k in range(6)
            ])
            max_proj = 0.0
            min_proj = 0.0
            # Init min to inf
            min_proj = float('inf')
            for i in range(len(c)):
                verts = base_vertices + c[i]
                dots = verts @ normals.T
                max_proj = max(max_proj, np.max(np.abs(dots)))
                # This fallback assumes center 0,0
            return 2.0 * max_proj / np.sqrt(3)
            
        best_score = compute_side_rigid_fallback(0.0)
        best_theta_rel = 0.0
        best_centers = centers
        best_outer_center = (0.0, 0.0)

    inner = []
    for i in range(len(best_centers)):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])

    side = best_score
    theta_out_deg = np.rad2deg(best_theta_rel)

    return inner, best_outer_center, side, theta_out_deg

if __name__ == "__main__":
    construct()
