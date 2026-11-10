# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = 500.0  # Safety margin before 600s limit
    
    n = 11
    # Angle for inner hexagons (30 degrees to allow tighter packing in hex lattice)
    inner_angle = 30.0
    
    # Precompute unit hexagon vertices for angle 30
    # Vertices of unit hex at angle 0: (1,0), (0.5, sqrt(3)/2), ...
    # Rotated by 30:
    angles_rad = np.deg2rad(30 + 60 * np.arange(6))
    v_unit = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
    
    # Outer hexagon normals (for containment check)
    # Outer hex angle 0. Vertices at 0, 60...
    # Normals to edges are at 30, 90, 150, 210, 270, 330
    outer_normals_rad = np.deg2rad(30 + 60 * np.arange(6))
    n_outer = np.column_stack([np.cos(outer_normals_rad), np.sin(outer_normals_rad)])
    
    # Function to calculate required outer side length S for a given set of centers
    def calc_required_S(centers):
        # centers: (11, 2)
        # For each inner hex, find max projection of its vertices onto outer normals
        max_proj = 0.0
        for i in range(n):
            c = centers[i]
            # Vertices of inner hex i
            verts = c + v_unit
            # Project onto outer normals
            # projections[6, 6] -> proj[k, j] = dot(n_k, v_j)
            projs = verts @ n_outer.T
            # Max absolute projection for this hex
            max_abs = np.max(np.abs(projs))
            if max_abs > max_proj:
                max_proj = max_abs
        
        # S * sqrt(3)/2 >= max_proj  =>  S >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)

    # Function to calculate overlap penalty
    # Constraint: max_k |d . n'_k| > sqrt(3)
    # n'_k are normals of rotated hex (angle 30).
    # Normals at 0, 60, 120...
    inner_normals_rad = np.deg2rad(0 + 60 * np.arange(6))
    n_inner = np.column_stack([np.cos(inner_normals_rad), np.sin(inner_normals_rad)])
    
    def calc_overlap_penalty(centers):
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                # Project d onto inner normals
                projs = np.abs(d @ n_inner.T)
                M = np.max(projs)
                # We require M > sqrt(3)
                if M < np.sqrt(3.0):
                    # Quadratic penalty for violation
                    violation = np.sqrt(3.0) - M
                    penalty += violation * violation
        return penalty

    # Objective function
    def objective(params):
        if time.time() > deadline:
            return np.inf
        centers = params.reshape(n, 2)
        S_req = calc_required_S(centers)
        pen = calc_overlap_penalty(centers)
        # Weight for penalty
        w = 100.0
        return S_req + w * pen

    # Initial guess: Hexagonal lattice
    # Spacing slightly more than sqrt(3)
    spacing = 1.74
    # 1 center, 6 neighbors, 4 more
    # Center
    pos = [(0.0, 0.0)]
    # Neighbors at distance spacing, angles 0, 60, ...
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pos.append((spacing * np.cos(angle), spacing * np.sin(angle)))
    # 4 more: pick from second ring
    # Second ring radius is 2 * spacing? No, distance from center to second ring nodes in hex lattice is sqrt(3)*spacing?
    # Let's just place them roughly
    # Nodes at (2*spacing, 0), etc are too far.
    # In hex lattice, neighbors of neighbors.
    # Let's try positions at radius ~ 1.74 * 1.5?
    # Or just use a generic grid and let optimizer fix it.
    # Better initial guess:
    # 1 center
    # 6 at dist 1.74
    # 4 at dist ~ 2.5, angles 30, 90, 150, 270?
    # Let's use a symmetric guess
    r2 = 2.5
    angles2 = np.deg2rad(30 + 90 * np.arange(4))
    for angle in angles2:
        pos.append((r2 * np.cos(angle), r2 * np.sin(angle)))
        
    x0 = np.array(pos).flatten()
    
    best_result = None
    best_score = np.inf
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
        # Try a few more restarts if time permits
        if time.time() < deadline - 50:
            for restart in range(5):
                if time.time() > deadline:
                    break
                # Add noise
                noise = np.random.normal(0, 0.5, x0.shape)
                x0_perturbed = x0 + noise
                res2 = minimize(objective, x0_perturbed, method='Powell', options={'maxiter': 500, 'disp': False})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_result = res2.x
    except Exception as e:
        pass

    if best_result is None:
        # Fallback to seed
        centers = np.array([
            [0, 0], [-2.5, 0], [2.5, 0],
            [-1.25, 2.17], [1.25, 2.17],
            [-1.25, -2.17], [1.25, -2.17],
            [-3.75, 2.17], [3.75, 2.17],
            [-3.75, -2.17], [3.75, -2.17]
        ])
        S = 8.0
    else:
        centers = best_result.reshape(n, 2)
        S = calc_required_S(centers)
        
        # Check validity and adjust if necessary
        # If penalty is high, we might have overlaps.
        # We should verify and potentially expand S or move hexes?
        # For now, we trust the optimizer minimized the combined objective.
        # However, we must ensure strict separation.
        # Let's check the min gap.
        
        min_gap = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = centers[i] - centers[j]
                projs = np.abs(d @ n_inner.T)
                M = np.max(projs)
                gap = M - np.sqrt(3.0)
                if gap < min_gap:
                    min_gap = gap
        
        if min_gap < 1e-6:
            # Scale up positions slightly to ensure gap
            scale = 1.0 + (1e-5 - min_gap) / (np.max(np.linalg.norm(centers, axis=1)) + 1e-9)
            centers = centers * scale
            S = calc_required_S(centers)

    # Construct output arrays
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = inner_angle

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(S)

# EVOLVE-BLOCK-END
