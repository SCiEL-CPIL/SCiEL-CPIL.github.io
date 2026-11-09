import numpy as np
import time
from scipy.optimize import differential_evolution

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    # Unit hexagon vertices (angle 0)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([[np.cos(a), np.sin(a)] for a in v_angles])
    
    # 3 edge normals (perpendicular to edges)
    n_angles = np.deg2rad([30, 90, 150])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    def compute_enclosing_side(pts):
        """Compute minimum side length of enclosing hexagon centered at origin."""
        # pts shape (11, 2)
        # Vertices of all hexagons: pts[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        # Shape (11, 6, 2)
        verts = pts[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        # Flatten to (66, 2)
        verts_flat = verts.reshape(-1, 2)
        
        max_proj = 0.0
        for k in range(3):
            n_k = normals[k]
            projs = np.abs(verts_flat @ n_k)
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
        return (2.0 / np.sqrt(3.0)) * max_proj

    def separation_penalty(pts):
        """Calculate penalty for overlapping hexagons."""
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                # Project on 3 normals
                proj_v = np.abs(v @ normals.T)
                max_proj = np.max(proj_v)
                # Required separation is sqrt(3) + 1e-6
                gap = max_proj - (np.sqrt(3) + 1e-6)
                if gap < 0:
                    penalty += gap * gap
        return penalty

    def objective(x):
        pts = x.reshape(11, 2)
        side = compute_enclosing_side(pts)
        pen = separation_penalty(pts)
        return side + 1000.0 * pen

    # Bounds for each coordinate
    # Initial guess side is ~4. Centers should be within [-4, 4]
    bounds = [(-5.0, 5.0)] * 22
    
    # Initial population size and other DE params
    # We run DE once, maybe with a few restarts if time permits
    
    best_x = None
    best_val = np.inf
    
    # We can try a few random initializations or just rely on DE's global nature
    # DE is stochastic, so we can run it once with a large population
    
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=200, popsize=15, tol=1e-8, polish=False)
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE failed or took too long, we might need a fallback, but let's assume it works.
    # Check time
    if time.time() > deadline - 10:
        if best_x is None:
            # Fallback to simple grid
            pts = np.array([[0, 0]])
            d = np.sqrt(3)
            for k in range(6):
                angle = np.deg2rad(60 * k)
                pts = np.vstack([pts, [[d * np.cos(angle), d * np.sin(angle)]]])
            for angle_deg in [0, 120, 180, 300]:
                angle = np.deg2rad(angle_deg)
                pts = np.vstack([pts, [[2 * d * np.cos(angle), 2 * d * np.sin(angle)]]])
            best_x = pts.flatten()
            best_val = objective(best_x)
        else:
            pass # Use best_x

    pts_final = best_x.reshape(11, 2)
    
    # Refine with local optimization (Nelder-Mead) to squeeze out last bits of error
    # This helps satisfy the strict 1e-6 gap constraint
    def objective_local(x):
        pts = x.reshape(11, 2)
        side = compute_enclosing_side(pts)
        pen = separation_penalty(pts)
        return side + 1000.0 * pen

    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        res_local = minimize(objective_local, best_x, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res_local.fun < best_val:
            best_x = res_local.x
            best_val = res_local.fun
            pts_final = best_x.reshape(11, 2)

    final_side = compute_enclosing_side(pts_final)
    
    # Verify validity and adjust side if necessary to ensure containment
    # The computed side is the exact minimum enclosing side for the current positions.
    # So containment is guaranteed by definition of compute_enclosing_side.
    # We just need to ensure separation is valid.
    
    # Check separation again
    valid_sep = True
    for i in range(11):
        for j in range(i + 1, 11):
            v = pts_final[i] - pts_final[j]
            proj_v = np.abs(v @ normals.T)
            if np.max(proj_v) <= np.sqrt(3) + 1e-6:
                valid_sep = False
                break
        if not valid_sep:
            break
            
    if not valid_sep:
        # If separation is still violated, we might need to increase side slightly?
        # No, separation is between inner hexagons.
        # If they overlap, the packing is invalid.
        # We should try to fix it by pushing them apart or accepting a slightly larger side
        # but the side is determined by the outer boundary.
        # If inner hexagons overlap, we must move them.
        # Simple repair: push apart
        for _ in range(50):
            moved = False
            for i in range(11):
                for j in range(i + 1, 11):
                    v = pts_final[i] - pts_final[j]
                    proj_v = np.abs(v @ normals.T)
                    max_proj = np.max(proj_v)
                    if max_proj <= np.sqrt(3) + 1e-6:
                        moved = True
                        min_idx = np.argmin(proj_v)
                        n = normals[min_idx]
                        dot = np.dot(v, n)
                        if dot > 0:
                            pts_final[i] += 0.01 * n
                            pts_final[j] -= 0.01 * n
                        else:
                            pts_final[i] -= 0.01 * n
                            pts_final[j] += 0.01 * n
            if not moved:
                break
        final_side = compute_enclosing_side(pts_final)

    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, final_side
