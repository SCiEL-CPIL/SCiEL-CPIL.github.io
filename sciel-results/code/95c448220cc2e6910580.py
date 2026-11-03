import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Pre-compute unit hexagon vertices for theta=0
    # Vertices at angles 0, 60, ...
    angles = np.arange(6) * np.pi / 3
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def get_rotated_verts(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return unit_verts @ R.T

    def objective(vars):
        return vars[26] # S

    def constraints(vars):
        centers = vars[:24].reshape(12, 2)
        theta = vars[24]
        phi = vars[25]
        S = vars[26]
        
        cons = []
        rot_verts = get_rotated_verts(theta)
        
        # 1. Non-overlap
        for i in range(12):
            for j in range(i + 1, 12):
                c_i = centers[i]
                c_j = centers[j]
                diff = c_i - c_j
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    angle_diff = 0
                else:
                    angle_diff = np.arctan2(diff[1], diff[0])
                
                alpha = angle_diff - theta
                alpha_mod = alpha % (np.pi / 3)
                beta = alpha_mod - np.pi / 6
                required_dist = np.sqrt(3) / np.cos(beta)
                
                # g = required_dist - dist <= 0
                def make_overlap_con(i, j, r_req):
                    def fun(v):
                        ci = v[:24].reshape(12, 2)[i]
                        cj = v[:24].reshape(12, 2)[j]
                        d = np.linalg.norm(ci - cj)
                        return r_req - d
                    return fun
                cons.append({'type': 'ineq', 'fun': make_overlap_con(i, j, required_dist)})

        # 2. Containment
        # Outer hexagon normals
        outer_angles = np.arange(6) * np.pi / 3 + phi
        normals = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
        R_in_outer = S * np.sqrt(3) / 2
        
        for i in range(12):
            c_i = centers[i]
            # Vertices of inner hexagon i
            verts = c_i + rot_verts
            
            for k in range(6):
                n_k = normals[k]
                for m in range(6):
                    v_m = verts[m]
                    # g = v_m dot n_k - R_in_outer <= 0
                    def make_contain_con(i, k, m, n_k, r_in):
                        def fun(v):
                            ci = v[:24].reshape(12, 2)[i]
                            th = v[24]
                            # Recalculate vertex? 
                            # To be fast, we can assume theta is fixed during the constraint eval?
                            # No, theta is a variable.
                            # We must recalculate the rotated vertex.
                            # This is expensive.
                            # Alternative: Linearize?
                            # Or just accept the cost. 12*6*6 = 432 constraints.
                            # Each constraint eval involves trig.
                            # 500 iterations * 432 * trig might be slow but < 120s?
                            # Let's try to optimize by passing theta and phi from v.
                            c_t = np.cos(th)
                            s_t = np.sin(th)
                            R_t = np.array([[c_t, -s_t], [s_t, c_t]])
                            v_off = unit_verts[m] @ R_t.T
                            v_full = ci + v_off
                            return v_full @ n_k - v[26] * np.sqrt(3) / 2
                        return fun
                    cons.append({'type': 'ineq', 'fun': make_contain_con(i, k, m, n_k, R_in_outer)})

        return cons

    # Initial guess
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 3/2])
    lattice_points = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            p = a * u + b * v
            lattice_points.append(p)
    lattice_points.sort(key=lambda p: np.linalg.norm(p))
    init_points = lattice_points[:12]
    
    x0 = np.array([coord for p in init_points for coord in p])
    x0 = np.append(x0, [0.0, 0.0, 4.0])
    
    bounds = [(None, None)] * 24 + [(0, 2*np.pi), (0, 2*np.pi), (1.0, 10.0)]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300, 'ftol': 1e-5})
    
    if res.success:
        best_vars = res.x
        centers = best_vars[:24].reshape(12, 2)
        theta = best_vars[24]
        phi = best_vars[25]
        S = best_vars[26]
    else:
        centers = np.array(init_points)
        theta = 0.0
        phi = 0.0
        S = 4.0
        
    inner = [[float(c[0]), float(c[1]), float(np.degrees(theta))] for c in centers]
    center = [0.0, 0.0]
    side = float(S)
    angle = float(np.degrees(phi))
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
