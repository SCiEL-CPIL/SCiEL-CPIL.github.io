import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    eps = 1e-12
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    # Define the 3 distinct symmetric cases for the 4 outer hexagons
    # Case 0: Empty gaps at 240, 300 (Adjacent). Current program's choice.
    # Case 1: Empty gaps at 120, 300 (Opposite).
    # Case 2: Empty gaps at 0, 120 (Separated by 1).
    
    cases = [
        [0, 60, 120, 180], # Empty 240, 300
        [60, 120, 180, 240], # Empty 0, 300 -> Wait, 0 and 300 are adjacent? No.
                             # 0 and 300 are 60 deg apart.
                             # Let's use explicit angle lists for the 4 chosen.
    ]
    
    # Let's define the 6 possible gap angles: 0, 60, 120, 180, 240, 300.
    # We pick 4.
    # Symmetry classes of 4-subsets of 6-cycle:
    # 1. 4 consecutive (2 adjacent empty). e.g. 0,60,120,180. Empty 240,300.
    # 2. 3 consecutive, 1 separated. e.g. 0,60,120,240. Empty 180,300.
    #    180 and 300 are separated by 120 deg.
    # 3. 2 pairs of consecutive. e.g. 0,60,180,240. Empty 120,300.
    #    120 and 300 are opposite.
    
    # Let's generate the 3 unique sets of angles.
    angle_sets = [
        [0, 60, 120, 180],   # Adjacent empty
        [0, 60, 180, 240],   # Opposite empty (120, 300 empty)
        [0, 60, 120, 240]    # Separated empty (180, 300 empty) - Wait, 180 and 300 are 120 apart.
    ]
    
    # Actually, let's just iterate all 15 combinations to be safe, it's cheap.
    import itertools
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))
    
    best_score = float('inf')
    best_config = None
    best_theta_rel = 0.0
    best_centers = None
    
    for combo in combinations:
        # Build centers
        centers = np.array([[0.0, 0.0]])
        
        # Inner ring (6) at 30, 90, ...
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
        # Outer ring (4) at chosen angles
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
        n_hex = len(centers)
        
        def compute_side_rigid(theta_rel, c=centers):
            theta_out = theta_rel
            normals = np.array([
                [np.cos(np.deg2rad(30 + k * 60) + theta_out),
                 np.sin(np.deg2rad(30 + k * 60) + theta_out)]
                for k in range(6)
            ])
            
            max_proj = 0.0
            for i in range(len(c)):
                verts = base_vertices + c[i]
                dots = verts @ normals.T
                max_proj = max(max_proj, np.max(np.abs(dots)))
            
            return 2.0 * max_proj / np.sqrt(3)

        # Coarse search
        thetas = np.linspace(0, np.pi/6, 200)
        local_best_score = float('inf')
        local_best_theta = 0.0
        for t in thetas:
            if time.time() > deadline - 5:
                break
            s = compute_side_rigid(t)
            if s < local_best_score:
                local_best_score = s
                local_best_theta = t
        
        # Refine
        if time.time() < deadline - 2:
            step = (np.pi/6) / 200
            bounds = (max(0, local_best_theta - 5*step), min(np.pi/6, local_best_theta + 5*step))
            try:
                res = minimize_scalar(compute_side_rigid, bounds=bounds, method='bounded', 
                                      options={'xatol': 1e-10, 'maxiter': 100})
                if res.fun < local_best_score:
                    local_best_score = res.fun
                    local_best_theta = res.x
            except Exception:
                pass

        if local_best_score < best_score:
            best_score = local_best_score
            best_config = combo
            best_theta_rel = local_best_theta
            best_centers = centers

    # Final polish on the best config
    if time.time() < deadline - 1:
        def compute_side_final(theta_rel):
            theta_out = theta_rel
            normals = np.array([
                [np.cos(np.deg2rad(30 + k * 60) + theta_out),
                 np.sin(np.deg2rad(30 + k * 60) + theta_out)]
                for k in range(6)
            ])
            max_proj = 0.0
            for i in range(len(best_centers)):
                verts = base_vertices + best_centers[i]
                dots = verts @ normals.T
                max_proj = max(max_proj, np.max(np.abs(dots)))
            return 2.0 * max_proj / np.sqrt(3)

        bounds_tight = (max(0, best_theta_rel - 1e-6), min(np.pi/6, best_theta_rel + 1e-6))
        try:
            res = minimize_scalar(compute_side_final, bounds=bounds_tight, method='bounded', 
                                  options={'xatol': 1e-14, 'maxiter': 1000})
            if res.fun < best_score:
                best_score = res.fun
                best_theta_rel = res.x
        except Exception:
            pass

    inner = []
    for i in range(len(best_centers)):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])

    side = best_score
    theta_out_deg = np.rad2deg(best_theta_rel)

    return inner, (0.0, 0.0), side, theta_out_deg

if __name__ == "__main__":
    construct()
