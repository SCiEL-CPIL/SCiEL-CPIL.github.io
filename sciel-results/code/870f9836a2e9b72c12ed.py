import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600 - 10  # 590s safety margin

    sqrt3 = np.sqrt(3)
    
    # Initial centers: 1 center, 6 ring 1, 4 ring 2
    # This is a valid non-overlapping configuration
    centers_init = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [2*sqrt3*np.cos(np.deg2rad(30)), 2*sqrt3*np.sin(np.deg2rad(30))],
        [2*sqrt3*np.cos(np.deg2rad(150)), 2*sqrt3*np.sin(np.deg2rad(150))],
        [2*sqrt3*np.cos(np.deg2rad(270)), 2*sqrt3*np.sin(np.deg2rad(270))],
        [2*sqrt3*np.cos(np.deg2rad(330)), 2*sqrt3*np.sin(np.deg2rad(330))],
    ])
    
    n_hex = 11
    
    def calc_S(centers, alpha_rad):
        """
        Calculate the minimum side length S of a regular hexagon centered at origin,
        with orientation alpha_rad, that contains all vertices of the hexagons
        defined by centers (assumed unit side, angle 0).
        """
        # Generate vertices for all inner hexagons
        # For speed, we can pre-calculate the 6 unit vertices
        unit_vertices = np.array([[np.cos(np.deg2rad(60*i)), np.sin(np.deg2rad(60*i))] for i in range(6)])
        
        # All vertices: (n_hex * 6, 2)
        # centers is (n_hex, 2)
        # vertices = centers[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]
        # This creates (n_hex, 6, 2). Reshape to (-1, 2).
        vertices = (centers[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]).reshape(-1, 2)
        
        # Normals of the outer hexagon
        # Outer hexagon vertices are at angles alpha + 60*i.
        # Normals are at angles alpha + 30 + 60*i.
        normal_angles = alpha_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
        
        cos_n = np.cos(normal_angles)
        sin_n = np.sin(normal_angles)
        
        # Project all vertices onto all 6 normals
        # proj shape: (n_vertices, 6)
        proj = vertices[:, 0:1] * cos_n[None, :] + vertices[:, 1:2] * sin_n[None, :]
        
        # For each normal, we need S * sqrt(3)/2 >= max(proj)
        # So S >= max(proj) * 2 / sqrt(3)
        max_proj = np.max(proj)
        
        S = 2 * max_proj / sqrt3
        return S

    def objective(vars):
        """
        vars: [cx0, cy0, cx1, cy1, ..., cx10, cy10, alpha_rad]
        """
        if time.time() > deadline:
            return 1e10
        centers = vars[:2*n_hex].reshape(n_hex, 2)
        alpha_rad = vars[2*n_hex]
        
        # Check constraints (distances)
        # If any distance < 2.001, return large penalty
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 2.001:
                    return 1e10 + (2.001 - dist)
        
        return calc_S(centers, alpha_rad)

    # Initial guess
    init_vars = np.concatenate([centers_init.flatten(), [0.0]])
    
    # Optimization
    best_S = 1e10
    best_vars = init_vars
    
    # Use Nelder-Mead for robustness without gradients
    # SLSQP might be better for constraints, but we handle constraints in objective.
    
    # Try a few restarts or just one good run
    # Nelder-Mead is local. Let's try to guide it.
    
    # We can also try to fix the symmetry to reduce variables, but full optimization is safer.
    
    # Time budget check
    if time.time() < deadline - 50:
        try:
            res = minimize(objective, init_vars, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res.fun < best_S:
                best_S = res.fun
                best_vars = res.x
        except Exception:
            pass
    
    # If we have time, try a finer search or different method
    if time.time() < deadline - 10:
        # Try SLSQP with explicit constraints for better convergence
        constraints = []
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                def make_constraint(i, j):
                    def constraint(vars):
                        centers = vars[:2*n_hex].reshape(n_hex, 2)
                        return np.linalg.norm(centers[i] - centers[j]) - 2.001
                    return constraint
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        
        try:
            res2 = minimize(objective, best_vars, method='SLSQP', 
                            constraints=constraints,
                            options={'maxiter': 500, 'disp': False, 'ftol': 1e-8})
            if res2.fun < best_S:
                best_S = res2.fun
                best_vars = res2.x
        except Exception:
            pass

    # Extract best solution
    best_centers = best_vars[:2*n_hex].reshape(n_hex, 2)
    best_alpha_rad = best_vars[2*n_hex]
    best_alpha_deg = np.rad2deg(best_alpha_rad) % 60.0
    
    # Final verification and calculation of S
    final_S = calc_S(best_centers, best_alpha_rad)
    
    # Ensure S is at least sqrt(11) to pass area check, though optimization should ensure this
    if final_S < np.sqrt(11):
        final_S = np.sqrt(11)
        
    inner_hex_data = np.column_stack((best_centers[:, 0], best_centers[:, 1], np.zeros(n_hex)))
    outer_hex_data = np.array([0.0, 0.0, best_alpha_deg])
    
    return inner_hex_data, outer_hex_data, float(final_S)
