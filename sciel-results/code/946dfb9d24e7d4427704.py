import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Generate initial positions for 12 unit hexagons
    # 3 columns, 4 rows in a hex grid
    pos = []
    for j in range(4):
        for i in range(3):
            x = i * np.sqrt(3) + (j % 2) * np.sqrt(3) / 2
            y = j * 1.5
            pos.append([x, y])
    pos = np.array(pos)
    # Center the cluster at origin
    pos -= pos.mean(axis=0)

    # Precompute unit hexagon vertices (relative to center)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    verts_rel = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Precompute normals for containment check
    # Normals are at angles phi + pi/6 + k * pi/3
    normal_angles_base = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    normals_base = np.column_stack([np.cos(normal_angles_base), np.sin(normal_angles_base)])

    def objective(vars):
        tx, ty, theta_c = vars[0], vars[1], vars[2]
        alphas = vars[3:15]
        cx, cy, S, phi_o = vars[15], vars[16], vars[17], vars[18]
        
        if S <= 0:
            return 1e6

        # Global transform
        c_tc = np.cos(theta_c)
        s_tc = np.sin(theta_c)
        R = np.array([[c_tc, -s_tc],
                      [s_tc, c_tc]])
        centers = pos @ R.T + np.array([tx, ty])
        
        # Generate all 72 points
        points_list = []
        for k in range(12):
            ang = theta_c + alphas[k]
            c_a = np.cos(ang)
            s_a = np.sin(ang)
            R_k = np.array([[c_a, -s_a],
                            [s_a, c_a]])
            verts = R_k @ verts_rel.T
            pts = centers[k] + verts
            points_list.append(pts)
            
        points = np.vstack(points_list)
        
        # Check containment in outer hexagon
        # Outer hexagon: center (cx, cy), side S, rotation phi_o
        # Apothem a = S * sqrt(3) / 2
        apothem = S * np.sqrt(3) / 2
        
        # Rotate normals
        c_po = np.cos(phi_o)
        s_po = np.sin(phi_o)
        R_p = np.array([[c_po, -s_po],
                        [s_po, c_po]])
        normals = R_p @ normals_base.T
        
        diff = points - np.array([cx, cy])
        dots = diff @ normals.T # shape (72, 6)
        violations = np.maximum(0, dots - apothem)
        penalty = np.sum(violations ** 2)
        
        return S + 100.0 * penalty

    # 2. Initial guess
    # Estimate S
    # Cluster width ~ 5.2, height ~ 4.5
    # Hexagon side S roughly covers this.
    # S=4.0 is a reasonable guess.
    x0 = np.array([0.0, 0.0, 0.0] + [0.0]*12 + [0.0, 0.0, 4.0, 0.0])
    
    best_result = None
    best_score = float('inf')
    
    # 3. Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6})
        if time.time() > deadline:
            raise TimeoutError
        if res.fun < best_score:
            best_result = res
            best_score = res.fun
    except Exception:
        pass

    # Try a second run with different initial rotation if time permits
    if time.time() < deadline - 10.0:
        x0_2 = np.array([0.0, 0.0, np.pi/6] + [0.0]*12 + [0.0, 0.0, 4.0, 0.0])
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6})
            if time.time() > deadline:
                raise TimeoutError
            if res2.fun < best_score:
                best_result = res2
                best_score = res2.fun
        except Exception:
            pass

    # 4. Extract best
    if best_result is not None and best_result.fun < float('inf'):
        tx, ty, theta_c = best_result.x[0], best_result.x[1], best_result.x[2]
        alphas = best_result.x[3:15]
        cx, cy, S, phi_o = best_result.x[15], best_result.x[16], best_result.x[17], best_result.x[18]
        
        c_tc = np.cos(theta_c)
        s_tc = np.sin(theta_c)
        R = np.array([[c_tc, -s_tc],
                      [s_tc, c_tc]])
        final_centers = pos @ R.T + np.array([tx, ty])
        
        inner = []
        for k in range(12):
            ang_rad = theta_c + alphas[k]
            inner.append([final_centers[k][0], final_centers[k][1], np.degrees(ang_rad)])
            
        return inner, (cx, cy), S, np.degrees(phi_o)
    else:
        # Fallback
        inner = [[0.0, 0.0, 0.0] for _ in range(12)]
        return inner, (0.0, 0.0), 10.0, 0.0
