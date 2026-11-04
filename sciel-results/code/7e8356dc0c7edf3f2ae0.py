import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a flower packing (1 center, 6 ring 1, 4 ring 2) with optimized rotations.
    Vectorized objective + epigraph formulation + adaptive smooth surrogate.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    # Precompute: for each inner hexagon i, store its center and base vertex projections
    # centers shape: (n_hex, 2)
    # base_vertices shape: (6, 2)
    
    def _compute_all_projections(theta_out, phis):
        """
        Vectorized computation of all vertex projections onto outer hexagon normals.
        Returns array of shape (n_hex, 6, 6) where [i, v, k] is projection of
        vertex v of hexagon i onto normal k.
        """
        # Outer normals: (6, 2)
        angles_out = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles_out), np.sin(angles_out)])  # (6, 2)
        
        # For each inner hexagon i with rotation phi_i:
        #   rotated_verts[i] = base_vertices @ R(phi_i).T + centers[i]
        #   proj[i, v, k] = rotated_verts[i, v] . normals[k]
        
        # phis: (n_hex,)
        c = np.cos(phis)  # (n_hex,)
        s = np.sin(phis)  # (n_hex,)
        
        # R(phi_i).T for each i: (n_hex, 2, 2)
        # R(phi) = [[c, -s], [s, c]], so R(phi).T = [[c, s], [-s, c]]
        rotT = np.zeros((n_hex, 2, 2))
        rotT[:, 0, 0] = c
        rotT[:, 0, 1] = s
        rotT[:, 1, 0] = -s
        rotT[:, 1, 1] = c
        
        # rotated_verts[i, v, :] = base_vertices[v, :] @ rotT[i] + centers[i, :]
        # base_vertices: (6, 2), rotT: (n_hex, 2, 2)
        # For each i: base_vertices @ rotT[i] gives (6, 2)
        # Use einsum: (6,2) @ (n,2,2) -> (n,6,2)
        rotated_verts = np.einsum('vd,ndj->nvj', base_vertices, rotT) + centers[:, np.newaxis, :]
        
        # proj[i, v, k] = rotated_verts[i, v, :] . normals[k, :]
        # rotated_verts: (n, 6, 2), normals: (6, 2)
        # proj: (n, 6, 6)
        proj = np.einsum('nvk,km->nvm', rotated_verts, normals)
        
        return proj, normals

    def compute_outer_side(theta_out, phis):
        """
        Exact outer hexagon side length.
        """
        proj, _ = _compute_all_projections(theta_out, phis)
        max_proj = np.max(proj)
        return 2.0 * max_proj / np.sqrt(3)

    def objective_smooth(params, beta=50.0):
        """
        Smooth objective using log-sum-exp approximation for the max projection.
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        
        proj, _ = _compute_all_projections(theta_out, phis)
        all_projs = proj.flatten()
        
        # Soft max via log-sum-exp with numerical stability
        m = np.max(all_projs)
        soft_max = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (all_projs - m))))
        
        return 2.0 * soft_max / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    def objective_epigraph(params):
        """
        Epigraph formulation: minimize t subject to all projections <= t.
        params: [t, theta_out, phi_0, ..., phi_10]
        Returns t + penalty for violations.
        """
        t = params[0]
        theta_out = params[1]
        phis = params[2:]
        
        proj, _ = _compute_all_projections(theta_out, phis)
        max_proj = np.max(proj)
        
        # Penalty for exceeding t
        penalty = np.maximum(0.0, max_proj - t) ** 2
        return t + penalty

    # Initial guess: align inner hexagon face normals radially outward
    x0 = np.zeros(n_hex + 1)
    x0[0] = 0.0  # outer rotation
    for i in range(n_hex):
        cx, cy = centers[i]
        if abs(cx) < 1e-12 and abs(cy) < 1e-12:
            x0[i+1] = 0.0
        else:
            alpha = np.arctan2(cy, cx)
            # Align a face normal to point radially outward
            # Base hexagon has face normals at 30, 90, 150, 210, 270, 330 degrees
            # We want one face normal to align with alpha
            x0[i+1] = alpha - np.deg2rad(30)
            
    # Alternative initial guess: all aligned
    x0_aligned = np.zeros(n_hex + 1)
    x0_aligned[1:] = np.deg2rad(30.0)  # All inner hexagons rotated 30 deg
    
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Multi-start with smooth surrogate, then exact refinement
    n_starts = 12
    for i in range(n_starts):
        if time.time() > deadline - 25:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            x_start = x0_aligned.copy()
        elif i == 2:
            # Outer rotation at 30 degrees
            x_start = x0.copy()
            x_start[0] = np.deg2rad(30.0)
        else:
            # Random perturbations around x0
            x_start = x0 + np.random.uniform(-0.15, 0.15, n_hex + 1)
            x_start[0] = np.random.uniform(-np.pi/6, np.pi/6)
            
        # Step 1: Smooth optimization with L-BFGS-B
        try:
            res_smooth = minimize(objective_smooth, x_start, method='L-BFGS-B',
                                  options={'maxiter': 300, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8})
            x_start = res_smooth.x
        except Exception:
            pass
        
        # Step 2: Exact optimization with Powell
        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 800, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Step 3: Try epigraph formulation from this starting point
        if time.time() < deadline - 15:
            try:
                t_init = compute_outer_side(x_start[0], x_start[1:])
                x_epigraph = np.concatenate([[t_init + 0.01], x_start])
                res_ep = minimize(objective_epigraph, x_epigraph, method='L-BFGS-B',
                                  options={'maxiter': 200, 'disp': False, 'ftol': 1e-10})
                # Extract the actual side length from the optimized params
                t_val = res_ep.x[0]
                theta_val = res_ep.x[1]
                phis_val = res_ep.x[2:]
                side_val = compute_outer_side(theta_val, phis_val)
                if side_val < best_score:
                    best_score = side_val
                    best_params = np.concatenate([[theta_val], phis_val])
            except Exception:
                pass
    
    if best_params is None:
        return _fallback_solution()
    
    # Phase 2: Nelder-Mead polish on exact objective
    if time.time() < deadline - 8:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 3000, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3:
        try:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000, 'disp': False})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
            
    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    side = best_score + 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    side = 4.0 + 1e-7
    return inner, (0.0, 0.0), side, 30.0
