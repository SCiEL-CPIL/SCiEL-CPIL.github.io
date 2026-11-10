import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600  # seconds
    SQRT3 = np.sqrt(3)
    MIN_DIST = SQRT3 + 1e-5
    
    # Initial positions: Hexagonal lattice
    def hex_lattice_points(n):
        points = []
        max_qr = 5
        for q in range(-max_qr, max_qr + 1):
            for r in range(-max_qr, max_qr + 1):
                x = SQRT3 * (q + r / 2.0)
                y = 1.5 * r
                dist = np.sqrt(x*x + y*y)
                points.append((dist, q, r, x, y))
        
        points.sort(key=lambda t: t[0])
        
        seen = set()
        selected = []
        for dist, q, r, x, y in points:
            if (q, r) not in seen:
                seen.add((q, r))
                selected.append((x, y))
                if len(selected) == n:
                    break
        return np.array(selected)

    init_positions = hex_lattice_points(11)
    x0 = init_positions[:, 0]
    y0 = init_positions[:, 1]
    S0 = 5.0
    A0 = 0.0
    
    init_vars = np.concatenate([x0, y0, [S0, A0]])
    
    # Precompute vertex offsets for inner hexagons (angle 0)
    angles_in = np.deg2rad(np.arange(0, 360, 60))
    vx_off = np.cos(angles_in)
    vy_off = np.sin(angles_in)
    
    def objective(vars):
        if time.time() - start_time > deadline - 5:
            return 1e9
        S = vars[22]
        return S

    def constraints(vars):
        x = vars[:11]
        y = vars[11:22]
        S = vars[22]
        A = vars[23]
        
        cons = []
        
        # Distance constraints
        for i in range(11):
            for j in range(i+1, 11):
                def con_dist(v, i=i, j=j):
                    dx = v[i] - v[j]
                    dy = v[i+11] - v[j+11]
                    return dx*dx + dy*dy - MIN_DIST**2
                cons.append({'type': 'ineq', 'fun': con_dist})
                
        # Containment constraints
        # Normal angles for outer hexagon
        normal_angles = np.deg2rad(A + 30 + np.arange(0, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        H = S * SQRT3 / 2.0
        
        # For each inner hex vertex, dot(v, n) - H <= 0
        # v = (x_i + vx_off, y_i + vy_off)
        for i in range(11):
            cx = x[i]
            cy = y[i]
            vx_i = cx + vx_off
            vy_i = cy + vy_off
            
            # dots shape (6, 6) -> vertices x normals
            # We need max over vertices for each normal <= H
            # So for each normal k, max_v (dot(v, n_k)) - H <= 0
            # This is not linear.
            # However, dot(v, n) is linear in v.
            # The constraint "all vertices inside" is equivalent to
            # max_vertex (dot(v, n_k)) <= H for all k.
            # max of linear functions is convex.
            # Minimizing S subject to convex constraints is a convex optimization problem?
            # SLSQP handles non-linear constraints.
            
            # We can add constraints for each vertex and normal?
            # dot(v_j, n_k) - H <= 0 for all j, k.
            # This is 36 constraints per hexagon.
            # 11 * 36 = 396 constraints.
            
            for k in range(6):
                nk_x = nx[k]
                nk_y = ny[k]
                for j in range(6):
                    def con_contain(v, i=i, j=j, k=k, nk_x=nk_x, nk_y=nk_y, S=S):
                        # v is the variable vector
                        # x_i = v[i], y_i = v[i+11]
                        # vertex x = v[i] + vx_off[j]
                        # vertex y = v[i+11] + vy_off[j]
                        vx_v = v[i] + vx_off[j]
                        vy_v = v[i+11] + vy_off[j]
                        dot = vx_v * nk_x + vy_v * nk_y
                        H_val = v[22] * SQRT3 / 2.0
                        return H_val - dot # We want dot <= H, so H - dot >= 0
                    cons.append({'type': 'ineq', 'fun': con_contain})
                    
        return cons

    bounds = [(-10, 10)] * 22
    bounds.append((3.31, 10.0))
    bounds.append((-30, 30))
    
    res = minimize(objective, init_vars, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
    
    if res.success:
        final_vars = res.x
    else:
        final_vars = init_vars
        
    x_final = final_vars[:11]
    y_final = final_vars[11:22]
    S_final = final_vars[22]
    A_final = final_vars[23]
    
    # Safety check: calculate required S
    def calc_required_S(x, y, A):
        angles = np.deg2rad(np.arange(0, 360, 60))
        vx = np.cos(angles)
        vy = np.sin(angles)
        normal_angles = np.deg2rad(A + 30 + np.arange(0, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        max_proj = 0.0
        for i in range(11):
            cx = x[i]
            cy = y[i]
            vx_i = cx + vx
            vy_i = cy + vy
            dots = np.outer(vx_i, nx) + np.outer(vy_i, ny)
            max_dot = np.max(dots)
            if max_dot > max_proj:
                max_proj = max_dot
        return 2.0 * max_proj / SQRT3

    S_calc = calc_required_S(x_final, y_final, A_final)
    S_final = max(S_final, S_calc, np.sqrt(11))
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = x_final
    inner_hex_data[:, 1] = y_final
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, A_final])
    
    return inner_hex_data, outer_hex_data, float(S_final)
