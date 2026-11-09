import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    # Unit hexagon vertices (angle 0)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([[np.cos(a), np.sin(a)] for a in v_angles])
    
    # 3 edge normals (perpendicular to edges)
    # The edges are at angles 30, 90, 150 from the normal?
    # Normal to edge connecting v0 and v1.
    # v0=(1,0), v1=(0.5, 0.866). Midpoint (0.75, 0.433). Normal direction (1, 0.577) -> 30 deg.
    n_angles = np.deg2rad([30, 90, 150])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    def compute_enclosing_side(pts):
        """Compute minimum side length of enclosing hexagon centered at origin."""
        # pts shape (11, 2)
        # Vertices of all hexagons
        verts = pts[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        verts_flat = verts.reshape(-1, 2)
        
        max_proj = 0.0
        for k in range(3):
            n_k = normals[k]
            projs = np.abs(verts_flat @ n_k)
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
        # Apothem A = max_proj. Side S = 2 * A / sqrt(3)
        return (2.0 / np.sqrt(3.0)) * max_proj

    def separation_penalty(pts):
        """Calculate penalty for overlapping hexagons."""
        penalty = 0.0
        # Required separation on at least one normal is sqrt(3) + 1e-6
        # Actually, the condition is: max_k |v . n_k| > sqrt(3) + 1e-6
        req_sep = np.sqrt(3) + 1e-6
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                # Project on 3 normals
                proj_v = np.abs(v @ normals.T)
                max_proj = np.max(proj_v)
                gap = max_proj - req_sep
                if gap < 0:
                    penalty += gap * gap
        return penalty

    def objective(x):
        pts = x.reshape(11, 2)
        side = compute_enclosing_side(pts)
        pen = separation_penalty(pts)
        return side + 1000.0 * pen

    # Initial guess
    # 1 center, 6 ring, 4 outer
    pts_init = np.zeros((11, 2))
    # Center
    pts_init[0] = [0.0, 0.0]
    # Ring 1
    d1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pts_init[k+1] = [d1 * np.cos(angle), d1 * np.sin(angle)]
    # Ring 2 (4 hexagons in valleys)
    # Valleys at 30, 90, 150, 210, 270, 330
    # We pick 4. Let's try 30, 90, 150, 210 for now, or symmetric 30, 150, 270, 30? No.
    # Let's try 30, 150, 270, 30 is bad.
    # Let's try 30, 90, 150, 210.
    angles_outer = [30, 90, 150, 210]
    d2 = 3.0
    for idx, angle_deg in enumerate(angles_outer):
        angle = np.deg2rad(angle_deg)
        pts_init[7+idx] = [d2 * np.cos(angle), d2 * np.sin(angle)]
        
    x_init = pts_init.flatten()
    
    best_x = x_init
    best_val = objective(x_init)
    
    # Run Nelder-Mead
    # Use multiple starts if time permits
    n_starts = 3
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
            
        if i == 0:
            x_start = x_init
        else:
            # Perturb the initial guess
            noise = np.random.normal(0, 0.1, 22)
            x_start = x_init + noise
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception as e:
            pass

    pts_final = best_x.reshape(11, 2)
    
    # Final refinement
    # Check validity
    def is_valid(pts):
        # Check separation
        req_sep = np.sqrt(3) + 1e-6
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                if np.max(proj_v) <= req_sep:
                    return False
        return True

    if not is_valid(pts_final):
        # Try to fix with a small local step
        # Or just accept if penalty is very small
        pass

    final_side = compute_enclosing_side(pts_final)
    
    # Construct output
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, final_side
