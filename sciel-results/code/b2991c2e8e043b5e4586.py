import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
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
    
    def compute_outer_side(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        Uses projection onto the 6 face normals of the outer hexagon.
        """
        # Outer hexagon face normals: at angles 30 + k*60 + theta_out
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            max_proj = max(max_proj, np.max(dots))
        
        # For a regular hexagon with side S, the apothem (center to face) is S * sqrt(3) / 2
        # So S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)
    
    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)
    
    def smooth_objective(params, t=100.0):
        """
        Smooth approximation using log-sum-exp for the max projection.
        """
        theta_out = params[0]
        phis = params[1:]
        
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        all_dots = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T  # shape (6, 6)
            all_dots.append(dots)
        
        all_dots = np.vstack(all_dots)  # shape (66, 6)
        
        # For each normal direction, take max over all vertices, then smooth max over 6 directions
        max_per_normal = np.max(all_dots, axis=0)  # shape (6,)
        
        # Log-sum-exp approximation of max over the 6 normals
        m = np.max(max_per_normal)
        log_sum_exp = m + np.log(np.sum(np.exp(max_per_normal - m)))
        smooth_max = log_sum_exp / t
        
        return 2.0 * smooth_max / np.sqrt(3)
    
    # Initial guess: all rotations 0, outer rotation 0
    x0 = np.zeros(n_hex + 1)
    
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Multi-start Powell optimization
    n_starts = 8
    for i in range(n_starts):
        if time.time() > deadline - 25:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            x_start = np.array([np.pi/6] + [0.0] * n_hex)
        elif i == 2:
            x_start = np.array([0.0] + [np.pi/6] * n_hex)
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)
        
        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return _fallback_solution()
    
    # Phase 2: Nelder-Mead polish with smooth objective
    if time.time() < deadline - 10:
        try:
            res2 = minimize(smooth_objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000, 'disp': False})
            true_score = objective(res2.x)
            if true_score < best_score:
                best_score = true_score
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 5:
        for iteration in range(50):
            if time.time() > deadline - 3:
                break
            
            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 3:
                    break
                
                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val
                
                # Golden section search around current value
                for delta in [1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6,
                             1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3,
                             1e-2, -1e-2]:
                    test_val = current_val + delta
                    if test_val < 0 or test_val > np.pi/3:
                        continue
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue
                
                if best_local < best_score - 1e-15:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True
            
            if not improved:
                break
    
    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    side = best_score + 1e-9  # Tiny margin for floating point safety
    
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
    
    # Compute side length analytically
    # With all hexagons at 30 deg (flat-top), outer at 30 deg
    # The critical constraint is the outermost vertices projected onto face normals
    theta_out = np.deg2rad(30)
    normals = np.array([
        [np.cos(np.deg2rad(30 + k * 60) + theta_out),
         np.sin(np.deg2rad(30 + k * 60) + theta_out)]
        for k in range(6)
    ])
    
    base_v = np.array([[np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)])
    phi = np.deg2rad(30)
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    local_verts = base_v @ rot.T
    
    max_proj = 0.0
    for x, y in centers:
        verts = local_verts + np.array([x, y])
        dots = verts @ normals.T
        max_proj = max(max_proj, np.max(dots))
    
    side = 2.0 * max_proj / np.sqrt(3) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
