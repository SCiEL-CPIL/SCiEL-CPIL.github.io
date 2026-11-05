import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # ---- Fixed inner layout: "3x3 grid" pattern ----
    # This is often tighter than the flower pattern for 9 hexagons
    eps = 1e-10
    centers = []
    # 3x3 grid in a hexagonal lattice
    # Distances: adjacent hexagons in same row/col are sqrt(3) apart?
    # Side 1 hexagon width (pointy) is 2, height is sqrt(3).
    # In a 3x3 block:
    # Let's place them in a standard hex lattice.
    # (0,0), (1,0), (2,0) -> x: 0, sqrt(3), 2*sqrt(3)
    # (0.5, 0.5) offset?
    # Actually, let's just define relative positions.
    # Hex 0: (0,0)
    # Hex 1: (sqrt(3), 0)
    # Hex 2: (2*sqrt(3), 0)
    # Hex 3: (sqrt(3)/2, 1.5)  # y offset is 1.5 (3/2 * 1)
    # Hex 4: (3*sqrt(3)/2, 1.5)
    # Hex 5: (5*sqrt(3)/2, 1.5)
    # Hex 6: (0, 3)
    # Hex 7: (sqrt(3), 3)
    # Hex 8: (2*sqrt(3), 3)
    
    s3 = np.sqrt(3)
    grid_centers = [
        (0.0, 0.0),
        (s3, 0.0),
        (2*s3, 0.0),
        (s3/2, 1.5),
        (3*s3/2, 1.5),
        (5*s3/2, 1.5),
        (0.0, 3.0),
        (s3, 3.0),
        (2*s3, 3.0)
    ]
    
    # Center the grid at origin for symmetry
    cx = np.mean([c[0] for c in grid_centers])
    cy = np.mean([c[1] for c in grid_centers])
    centers = np.array([[c[0] - cx, c[1] - cy] for c in grid_centers])
    
    n_hex = len(centers)
    assert n_hex == 9

    # Base vertices of a unit hexagon, pointy-right (angle 0)
    base_vertices = np.array(
        [[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)]
    )

    def get_all_vertices(params):
        """
        params: (n_hex * 3,) array [x1, y1, phi1, x2, y2, phi2, ...]
        Returns: (n_hex * 6, 2) array of vertices.
        """
        out = np.empty((n_hex * 6, 2))
        for i in range(n_hex):
            x, y, phi = params[i*3], params[i*3+1], params[i*3+2]
            c, s = np.cos(phi), np.sin(phi)
            R = np.array([[c, -s], [s, c]])
            out[i * 6:(i + 1) * 6] = base_vertices @ R.T + np.array([x, y])
        return out

    def compute_outer_side_for_angle(theta_out, verts):
        """
        Compute side length of outer hexagon with rotation theta_out 
        containing all vertices in 'verts'.
        Outer hexagon is centered at origin.
        """
        angles = np.deg2rad(30) + np.arange(6) * np.deg2rad(60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        dots = verts @ normals.T
        max_proj = np.max(dots, axis=0)
        
        apothem = float(np.max(max_proj))
        return 2.0 * apothem / np.sqrt(3)

    def objective_smooth(params, beta=100.0):
        """
        Smooth objective function for gradient-based optimization.
        Uses log-sum-exp to smooth the max over projections.
        """
        if time.time() > deadline:
            return float('inf')
            
        verts = get_all_vertices(params)
        
        # We need to minimize the outer side length.
        # The outer side length S(theta) = 2/sqrt(3) * max_k (max_v (v . n_k(theta)))
        # This is non-smooth.
        # Approximate max_v (v . n_k) with log-sum-exp.
        # Then minimize over theta? No, theta is also a variable or we can optimize it separately.
        # Let's include theta in the parameters? Or just optimize positions/rotations for a fixed theta?
        # The optimal theta for a symmetric grid is likely 0 or 30.
        # Let's fix theta to 0 for the smooth optimization, then polish with theta search.
        
        theta = 0.0
        angles = np.deg2rad(30) + np.arange(6) * np.deg2rad(60) + theta
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        dots = verts @ normals.T # (M, 6)
        
        # Smooth max over vertices for each normal
        # max_i (dots[i, k]) approx (1/beta) * log(sum(exp(beta * dots[:, k])))
        max_dots_smooth = np.zeros(6)
        for k in range(6):
            col = dots[:, k]
            max_val = np.max(col)
            max_dots_smooth[k] = max_val + (1.0/beta) * np.log(np.sum(np.exp(beta * (col - max_val))))
            
        # The outer apothem is the max of these 6 smooth maxes.
        # We also smooth the max over the 6 faces.
        apothem_smooth = np.max(max_dots_smooth) # This is still non-smooth wrt faces, but faces are discrete.
        # Actually, the max over faces is the "support function". 
        # To make it fully differentiable, we might need to smooth the max over k as well, 
        # or just rely on the fact that near the optimum, the active set is stable.
        
        # Let's just return the smooth apothem.
        return 2.0 * apothem_smooth / np.sqrt(3)

    def objective_exact(params):
        """
        Exact objective function.
        """
        if time.time() > deadline:
            return float('inf')
            
        verts = get_all_vertices(params)
        
        # Find best theta_out
        best_S = float('inf')
        n_coarse = 12
        for i in range(n_coarse):
            theta = i * (np.pi / 3) / n_coarse
            S = compute_outer_side_for_angle(theta, verts)
            if S < best_S:
                best_S = S
        
        return best_S

    # Initial guess: Grid positions, 0 rotation
    x0 = np.zeros(n_hex * 3)
    for i in range(n_hex):
        x0[i*3] = centers[i][0]
        x0[i*3+1] = centers[i][1]
        x0[i*3+2] = 0.0
        
    best_score = objective_exact(x0)
    best_params = x0.copy()
    
    # Gradient-based optimization using L-BFGS-B
    # We use the smooth objective for gradients
    try:
        # Use a moderate beta for stability
        res = minimize(objective_smooth, x0, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False},
                       args=(100.0,))
        if res.fun < best_score:
            # Verify with exact objective
            exact_score = objective_exact(res.x)
            if exact_score < best_score:
                best_score = exact_score
                best_params = res.x.copy()
    except Exception:
        pass

    # Polish with Nelder-Mead on the exact objective
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective_exact, best_params, method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Final fine-tuning of outer angle
    final_verts = get_all_vertices(best_params)
    best_theta_out = 0.0
    min_S = float('inf')
    
    # Coarse
    for i in range(12):
        theta = i * (np.pi / 3) / 12
        S = compute_outer_side_for_angle(theta, final_verts)
        if S < min_S:
            min_S = S
            best_theta_out = theta
            
    # Fine
    lower = best_theta_out - (np.pi / 3) / 12
    upper = best_theta_out + (np.pi / 3) / 12
    def S_of_theta(theta):
        return compute_outer_side_for_angle(theta, final_verts)
    try:
        res_final = minimize_scalar(S_of_theta, bounds=(lower, upper), method='bounded', 
                                    options={'xatol': 1e-10, 'disp': False})
        if res_final.success:
            if res_final.fun < min_S:
                min_S = res_final.fun
                best_theta_out = res_final.x
    except Exception:
        pass
        
    if min_S < best_score:
        best_score = min_S

    # Construct return values
    inner = []
    for i in range(n_hex):
        x = best_params[i*3]
        y = best_params[i*3+1]
        angle_deg = float(np.rad2deg(best_params[i*3+2]))
        inner.append([float(x), float(y), angle_deg])

    # Add tiny epsilon to side length to ensure validity if on boundary
    side = best_score + 1e-9
    return inner, (0.0, 0.0), float(side), float(np.rad2deg(best_theta_out))

if __name__ == "__main__":
    construct()
