import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    def get_vertices(cx, cy, angle_deg):
        angles = np.deg2rad(np.arange(0, 360, 60) + angle_deg)
        vx = np.cos(angles)
        vy = np.sin(angles)
        return np.column_stack([cx + vx, cy + vy])

    def min_gap(v1, v2):
        # SAT for two convex polygons
        # v1, v2 are (6,2)
        edges1 = np.roll(v1, -1, axis=0) - v1
        normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
        normals1 /= np.linalg.norm(normals1, axis=1, keepdims=True)
        
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
        normals2 /= np.linalg.norm(normals2, axis=1, keepdims=True)
        
        all_normals = np.vstack([normals1, normals2])
        
        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T
        
        min1 = np.min(proj1, axis=1)
        max1 = np.max(proj1, axis=1)
        min2 = np.min(proj2, axis=1)
        max2 = np.max(proj2, axis=1)
        
        gaps = np.maximum(min2 - max1, min1 - max2)
        return np.min(gaps)

    def compute_S(params):
        # params: (11, 3) -> (x, y, angle)
        xy = params[:, :2]
        angles = params[:, 2]
        
        # Compute vertices for all
        verts = np.zeros((11, 6, 2))
        for i in range(11):
            verts[i] = get_vertices(xy[i, 0], xy[i, 1], angles[i])
            
        # Bounding hexagon side S
        # The outer hexagon is centered at (0,0) with angle 0 (aligned with axes)
        # We need to find the smallest S such that all vertices are inside.
        # The condition is: for each of the 6 directions, the max projection <= S * sqrt(3)/2
        # Directions are 0, 60, 120, 180, 240, 300 degrees.
        # But the hexagon is defined by its sides.
        # The apothem is S * sqrt(3)/2.
        # The projection of a point (x,y) onto the normal of side k is x*cos(theta) + y*sin(theta).
        # theta for sides are 30, 90, 150, 210, 270, 330.
        side_angles = np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(side_angles), np.sin(side_angles)])
        
        flat_verts = verts.reshape(-1, 2)
        projections = flat_verts @ normals.T
        max_proj = np.max(projections, axis=1)
        
        # S * sqrt(3)/2 >= max_proj
        S = 2.0 * np.max(max_proj) / np.sqrt(3)
        
        # Check separation
        min_gap_val = np.inf
        for i in range(11):
            for j in range(i + 1, 11):
                gap = min_gap(verts[i], verts[j])
                if gap < min_gap_val:
                    min_gap_val = gap
        
        penalty = 0.0
        if min_gap_val < 1e-6:
            penalty = (1e-6 - min_gap_val) ** 2 * 10000.0
            
        return S + penalty

    # Initial guess
    np.random.seed(42)
    params0 = np.zeros((11, 3))
    # 1 center
    params0[0] = [0, 0, 0]
    # 6 ring 1
    for k in range(6):
        ang = np.deg2rad(k * 60)
        params0[1+k] = [np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang), 0]
    # 4 ring 2
    for k in range(4):
        ang = np.deg2rad(k * 90 + 45)
        params0[7+k] = [2*np.sqrt(3)*np.cos(ang), 2*np.sqrt(3)*np.sin(ang), 0]

    bounds = [(-5, 5), (-5, 5), (0, 360)] * 11

    best_params = params0.copy()
    best_S = compute_S(best_params)

    # Optimization
    try:
        res = differential_evolution(
            compute_S, bounds, seed=42, maxiter=500, popsize=20,
            tol=1e-4, polish=False
        )
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    if time.time() < deadline - 30:
        try:
            res_nm = minimize(
                compute_S, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_nm.fun < best_S:
                best_S = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass

    # Final validation and return
    xy = best_params[:, :2]
    angles = best_params[:, 2]
    inner_data = best_params
    
    # Calculate S for return
    verts = np.zeros((11, 6, 2))
    for i in range(11):
        verts[i] = get_vertices(xy[i, 0], xy[i, 1], angles[i])
        
    side_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(side_angles), np.sin(side_angles)])
    flat_verts = verts.reshape(-1, 2)
    projections = flat_verts @ normals.T
    max_proj = np.max(projections, axis=1)
    S = 2.0 * np.max(max_proj) / np.sqrt(3)

    outer_data = np.array([0.0, 0.0, 0.0])
    
    return inner_data, outer_data, float(S)
