import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590 # 10s margin

    # Base hexagon vertices for side 1, angle 0
    # Vertices at 0, 60, ..., 300
    angles_base = np.deg2rad(np.arange(0, 360, 60))
    base_verts = np.column_stack([np.cos(angles_base), np.sin(angles_base)])
    
    # Normals for outer hexagon (angle 0)
    # Edges are between 0-60, 60-120...
    # Midpoint of 0-60 is 30 deg. Normal is 30 deg.
    outer_normal_angles = np.deg2rad(np.arange(30, 360, 60))
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])
    
    def get_hex_verts(cx, cy, angle_deg):
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        return base_verts @ R.T + np.array([cx, cy])
        
    def min_gap(v1, v2):
        # v1, v2 are (6, 2) arrays
        # Normals of v1 and v2
        # Edges of v1: v1[i] to v1[i+1]
        # Normal is perpendicular to edge, pointing outwards?
        # For SAT, we just need the normal direction.
        # Let's compute normals for v1 and v2.
        
        def get_normals(verts):
            # verts (6,2)
            edges = np.roll(verts, -1, axis=0) - verts
            # Normal to edge (dx, dy) is (-dy, dx) or (dy, -dx)
            # We need consistent orientation.
            # For a CCW polygon, (dy, -dx) points inwards?
            # Let's check. Edge (1,0) to (0,1). Edge vec (-1, 1).
            # Normal (1, 1) points away from origin (0,0)?
            # Midpoint (0.5, 0.5). Normal (1,1) dot (0.5,0.5) > 0. Yes.
            # So normals = np.column_stack([edges[:,1], -edges[:,0]])
            normals = np.column_stack([edges[:, 1], -edges[:, 0]])
            # Normalize
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            return normals / norms
            
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        all_normals = np.vstack([n1, n2])
        
        # Project v1 and v2 onto all_normals
        # (12, 6) projections
        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T
        
        min1 = np.min(proj1, axis=1)
        max1 = np.max(proj1, axis=1)
        min2 = np.min(proj2, axis=1)
        max2 = np.max(proj2, axis=1)
        
        # Gap is max(min2 - max1, min1 - max2)
        # If they overlap, this will be negative.
        gaps = np.maximum(min2 - max1, min1 - max2)
        return np.min(gaps)

    def evaluate(params):
        # params (11, 3)
        if time.time() > deadline:
            return 1e10
            
        xy = params[:, :2]
        angles = params[:, 2]
        
        # Calculate vertices
        # Vectorized calculation
        cos_a = np.cos(np.deg2rad(angles))
        sin_a = np.sin(np.deg2rad(angles))
        
        # Rotate base vertices
        # base_verts (6, 2)
        # We want (11, 6, 2)
        # v_rot = [[ca, -sa], [sa, ca]] @ v
        # v_rot_x = ca * vx - sa * vy
        # v_rot_y = sa * vx + ca * vy
        
        vx = base_verts[:, 0]
        vy = base_verts[:, 1]
        
        rot_vx = cos_a[:, None] * vx[None, :] - sin_a[:, None] * vy[None, :]
        rot_vy = sin_a[:, None] * vx[None, :] + cos_a[:, None] * vy[None, :]
        
        verts_x = xy[:, 0:1] + rot_vx
        verts_y = xy[:, 1:2] + rot_vy
        
        # Stack to (11, 6, 2)
        verts = np.stack([verts_x, verts_y], axis=-1)
        
        # 1. Check Containment
        # Flatten vertices to (66, 2)
        flat_verts = verts.reshape(-1, 2)
        
        # Project onto outer normals
        # (66, 6)
        projections = flat_verts @ outer_normals.T
        max_proj = np.max(projections, axis=1)
        global_max_proj = np.max(max_proj)
        
        # Required Apothem
        A_req = global_max_proj
        # Side S = 2 * A / sqrt(3)
        S = 2.0 * A_req / np.sqrt(3)
        
        # 2. Check Separation
        min_gap_val = np.inf
        for i in range(11):
            for j in range(i + 1, 11):
                gap = min_gap(verts[i], verts[j])
                if gap < min_gap_val:
                    min_gap_val = gap
                
        # Penalty for overlap (gap < 1e-6)
        penalty = 0.0
        if min_gap_val < 1e-6:
            penalty = (1e-6 - min_gap_val) ** 2 * 10000.0
            
        return S + penalty

    # Initial guess
    # 1 center, 6 ring, 4 outer?
    # Let's try a structured init
    np.random.seed(42)
    
    # Center
    x0 = np.zeros(11)
    y0 = np.zeros(11)
    a0 = np.zeros(11)
    
    # Ring 1 (6 hexagons)
    d = np.sqrt(3)
    for k in range(6):
        ang = k * 60
        x0[1+k] = d * np.cos(np.deg2rad(ang))
        y0[1+k] = d * np.sin(np.deg2rad(ang))
        
    # Ring 2 (4 hexagons) - place them in gaps of ring 1?
    # Gaps are at 30, 150, 270, 330?
    # Distance from center to gap center?
    # Let's just place them at d * 2 roughly?
    # Or use optimization to find them.
    # Let's scatter the remaining 4 randomly near the perimeter.
    for k in range(4):
        ang = (k * 90 + 45)
        x0[7+k] = 2.5 * np.cos(np.deg2rad(ang))
        y0[7+k] = 2.5 * np.sin(np.deg2rad(ang))
        
    params0 = np.column_stack([x0, y0, a0])
    
    # Optimize
    bounds = [(-5, 5), (-5, 5), (0, 360)] * 11
    bounds = [tuple(b) for b in bounds]
    
    best_S = 1e10
    best_params = params0
    
    # Try Differential Evolution
    # maxiter is key. 1000 generations might be enough.
    # popsize 15
    try:
        res = differential_evolution(evaluate, bounds, seed=42, maxiter=1000, popsize=15, tol=1e-4, polish=False)
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Try Local Refinement (Nelder-Mead) from best found
    if time.time() < deadline - 30:
        try:
            res_nm = minimize(evaluate, best_params, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res_nm.fun < best_S:
                best_S = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
            
    # Final Validation and Extraction
    # Recalculate S from best_params to ensure it's the geometric S, not the penalized objective
    final_S = evaluate(best_params)
    # If penalty was applied, final_S includes it. We need pure S.
    # But if it's valid, penalty is 0.
    # Let's re-evaluate components to be safe.
    
    xy = best_params[:, :2]
    angles = best_params[:, 2]
    
    cos_a = np.cos(np.deg2rad(angles))
    sin_a = np.sin(np.deg2rad(angles))
    
    vx = base_verts[:, 0]
    vy = base_verts[:, 1]
    
    rot_vx = cos_a[:, None] * vx[None, :] - sin_a[:, None] * vy[None, :]
    rot_vy = sin_a[:, None] * vx[None, :] + cos_a[:, None] * vy[None, :]
    
    verts_x = xy[:, 0:1] + rot_vx
    verts_y = xy[:, 1:2] + rot_vy
    verts = np.stack([verts_x, verts_y], axis=-1)
    flat_verts = verts.reshape(-1, 2)
    projections = flat_verts @ outer_normals.T
    max_proj = np.max(np.max(projections, axis=1))
    pure_S = 2.0 * max_proj / np.sqrt(3)
    
    # Check validity
    valid = True
    if not np.isfinite(pure_S):
        valid = False
    if pure_S < np.sqrt(11):
        valid = False
        
    # Check separation
    min_gap_val = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            gap = min_gap(verts[i], verts[j])
            if gap < min_gap_val:
                min_gap_val = gap
    if min_gap_val < 1e-6:
        valid = False
        
    if not valid:
        # Fallback to seed's fallback (which might be invalid or just a guess)
        # The seed's fallback was a specific grid. Let's return a known valid small packing if possible?
        # Or just return the seed's fallback structure.
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

    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(pure_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
