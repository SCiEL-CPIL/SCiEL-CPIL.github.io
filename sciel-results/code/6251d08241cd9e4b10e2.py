import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 600s limit

    # Hexagon vertices for unit hexagon (side 1, angle 0)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([[np.cos(a), np.sin(a)] for a in v_angles])
    
    # The 3 edge normals (perpendicular to edges) at 30, 90, 150 degrees
    n_angles = np.deg2rad([30, 90, 150])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    # Width of unit hexagon along each normal direction
    # For normal at 30 deg: max projection = cos(0)*cos(30) + sin(0)*sin(30) = sqrt(3)/2
    # Actually let's compute: vertex at angle 0 projects to cos(30)=sqrt(3)/2
    # vertex at 60: cos(60)*cos(30)+sin(60)*sin(30) = 0.5*0.866 + 0.866*0.5 = 0.866
    # So width = 2 * sqrt(3)/2 = sqrt(3)
    hex_width = np.sqrt(3)
    
    def compute_enclosing_side(pts):
        """Compute minimum side length of enclosing hexagon centered at origin."""
        # For each of the 3 normals, find max |projection| of all vertices of all hexagons
        max_proj = 0.0
        for k in range(3):
            n = normals[k]
            # Project all hexagon vertices onto normal
            # For each inner hex at pts[i], its vertices are pts[i] + hex_verts
            # Project onto n: (pts[i] + hex_verts[v]) dot n
            # = pts[i] dot n + hex_verts[v] dot n
            pts_proj = pts @ n  # (11,)
            verts_proj = hex_verts @ n  # (6,)
            # All combinations: pts_proj[:,None] + verts_proj[None,:]
            all_projs = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
            max_abs = np.max(np.abs(all_projs))
            if max_abs > max_proj:
                max_proj = max_abs
        # Side length S relates to max projection: S * (sqrt(3)/2) = max_proj
        # Because the apothem (distance from center to edge) of regular hexagon with side S is S*sqrt(3)/2
        # And the max projection of vertices onto a normal perpendicular to an edge equals the apothem + S/2... 
        # Actually for a regular hexagon with side S, vertices at distance S from center.
        # The projection of a vertex onto the normal perpendicular to an edge:
        # If edge is perpendicular to normal, the vertices adjacent to that edge project to S*cos(30)=S*sqrt(3)/2
        # The opposite vertex projects to S. Wait, let me reconsider.
        # 
        # For outer hexagon with side S, vertices at distance S from center at angles 0,60,...
        # Normal at 30 deg: vertex at 0 projects to S*cos(30)=S*sqrt(3)/2
        # vertex at 60: S*cos(60)*cos(30)+S*sin(60)*sin(30) = S*(0.5*0.866+0.866*0.5)=S*0.866
        # vertex at 120: S*cos(120)*cos(30)+S*sin(120)*sin(30) = S*(-0.5*0.866+0.866*0.5)=0
        # So max projection = S*sqrt(3)/2
        # Therefore S = max_proj * 2/sqrt(3)
        return max_proj * 2.0 / np.sqrt(3.0)
    
    def check_separation(pts, tol=1e-6):
        """Check that all pairs of hexagons are separated by > tol on at least one normal."""
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj = np.abs(v @ normals.T)  # (3,)
                gaps = proj - hex_width
                if np.max(gaps) <= tol:
                    return False
        return True
    
    def check_boundary(pts, S, tol=1e-6):
        """Check all inner hex vertices are inside outer hexagon."""
        outer_verts = S * hex_verts
        for i in range(11):
            for v in range(6):
                p = pts[i] + hex_verts[v]
                for e in range(6):
                    p1 = outer_verts[e]
                    p2 = outer_verts[(e + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -tol:
                        return False
        return True
    
    def objective(x):
        pts = x.reshape(11, 2)
        S = compute_enclosing_side(pts)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj = np.abs(v @ normals.T)
                gaps = proj - hex_width
                min_gap = np.min(gaps)
                if min_gap < 0:
                    penalty += (-min_gap) ** 2
        
        # Penalty for boundary violation
        bnd_penalty = 0.0
        outer_verts = S * hex_verts
        for i in range(11):
            for v in range(6):
                p = pts[i] + hex_verts[v]
                for e in range(6):
                    p1 = outer_verts[e]
                    p2 = outer_verts[(e + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < 0:
                        bnd_penalty += (-cross) ** 2
        
        return S + 100.0 * penalty + 100.0 * bnd_penalty
    
    # Initial configuration: center + 6 ring + 4 outer
    d = np.sqrt(3)
    pts_init = np.zeros((11, 2))
    pts_init[0] = [0, 0]
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pts_init[k + 1] = [d * np.cos(angle), d * np.sin(angle)]
    # 4 more at distance 2*d at angles 0, 120, 180, 300
    for k, ang_deg in enumerate([0, 120, 180, 300]):
        angle = np.deg2rad(ang_deg)
        pts_init[7 + k] = [2 * d * np.cos(angle), 2 * d * np.sin(angle)]
    
    x0 = pts_init.flatten()
    
    best_x = x0.copy()
    best_S = compute_enclosing_side(pts_init)
    best_obj = objective(x0)
    
    # Try multiple optimization approaches
    np.random.seed(12345)
    
    # Powell optimization
    if time.time() < deadline - 100:
        res = minimize(objective, x0, method='Powell', 
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
            best_S = compute_enclosing_side(res.x.reshape(11, 2))
    
    # Nelder-Mead with perturbations
    for restart in range(5):
        if time.time() > deadline - 80:
            break
        if restart == 0:
            x_start = best_x.copy()
        else:
            x_start = best_x + np.random.normal(0, 0.02, best_x.shape)
        
        res = minimize(objective, x_start, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
            best_S = compute_enclosing_side(res.x.reshape(11, 2))
    
    pts_final = best_x.reshape(11, 2)
    final_S = compute_enclosing_side(pts_final)
    
    # Post-processing: fix any small violations
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        valid = True
        # Check and fix separations
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts_final[i] - pts_final[j]
                proj = np.abs(v @ normals.T)
                gaps = proj - hex_width
                if np.max(gaps) <= 1e-6:
                    valid = False
                    min_idx = np.argmin(proj)
                    n = normals[min_idx]
                    dot = np.dot(v, n)
                    if dot > 0:
                        pts_final[i] += 0.005 * n
                        pts_final[j] -= 0.005 * n
                    else:
                        pts_final[i] -= 0.005 * n
                        pts_final[j] += 0.005 * n
        
        if valid:
            break
    
    # Recompute side after fixes
    final_S = compute_enclosing_side(pts_final)
    
    # Ensure boundary validity - expand if needed
    if not check_boundary(pts_final, final_S):
        for factor in [1.001, 1.005, 1.01, 1.02]:
            test_S = final_S * factor
            if check_boundary(pts_final, test_S):
                final_S = test_S
                break
    
    # Final validation
    if not check_separation(pts_final):
        # Try to fix
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            fixed = True
            for i in range(11):
                for j in range(i + 1, 11):
                    v = pts_final[i] - pts_final[j]
                    proj = np.abs(v @ normals.T)
                    gaps = proj - hex_width
                    if np.max(gaps) <= 1e-6:
                        fixed = False
                        min_idx = np.argmin(proj)
                        n = normals[min_idx]
                        dot = np.dot(v, n)
                        if dot > 0:
                            pts_final[i] += 0.002 * n
                            pts_final[j] -= 0.002 * n
                        else:
                            pts_final[i] -= 0.002 * n
                            pts_final[j] += 0.002 * n
            if fixed:
                break
        final_S = compute_enclosing_side(pts_final)
    
    # Build output
    inner = np.zeros((11, 3))
    inner[:, 0] = pts_final[:, 0]
    inner[:, 1] = pts_final[:, 1]
    inner[:, 2] = 0.0
    
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, final_S
