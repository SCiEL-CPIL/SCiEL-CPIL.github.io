import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600 - 30  # Safety margin

    # Precompute base normals for unit hexagon (flat-top orientation)
    base_angles = np.deg2rad(np.arange(30, 360, 60))
    base_normals = np.column_stack((np.cos(base_angles), np.sin(base_angles)))  # (6, 2)

    def hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ))

    def get_normals(a_deg):
        """Get the 6 edge normals for a hexagon with given angle."""
        a_rad = np.deg2rad(a_deg)
        c, s = np.cos(a_rad), np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        return base_normals @ R.T

    def is_separated(v1, v2, n1, n2):
        """
        Check if two hexagons are separated.
        v1, v2: (6, 2) vertex arrays
        n1, n2: (6, 2) normal arrays for each hexagon
        """
        for i in range(6):
            n = n1[i]
            p1 = v1 @ n
            p2 = v2 @ n
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        for i in range(6):
            n = n2[i]
            p1 = v1 @ n
            p2 = v2 @ n
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        return False

    def get_min_side(positions, angles):
        """
        Compute the minimum side length of the outer hexagon.
        positions: (11, 2)
        angles: (11,)
        """
        outer_normals = base_normals  # (6, 2)
        
        max_proj = 0.0
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(projs))
        
        S = 2 * max_proj / np.sqrt(3)
        return S

    def smooth_max(x, t=0.01):
        """Smooth approximation of max using log-sum-exp."""
        x = np.asarray(x)
        return t * np.log(np.sum(np.exp(x / t)))

    def get_min_side_smooth(positions, angles, t=0.01):
        """
        Smooth version of get_min_side for better gradient flow.
        """
        outer_normals = base_normals  # (6, 2)
        
        all_projs = []
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T  # (6, 6)
            all_projs.append(projs)
        
        all_projs = np.concatenate(all_projs, axis=0)  # (66, 6)
        # For each outer normal, take smooth max over all inner vertices
        max_projs = np.array([smooth_max(all_projs[:, k], t) for k in range(6)])
        max_proj = smooth_max(max_projs, t)
        
        S = 2 * max_proj / np.sqrt(3)
        return S

    def containment_penalty(positions, angles, S):
        """
        Penalty for inner hexagons not being contained in outer hexagon of side S.
        """
        outer_normals = base_normals  # (6, 2)
        apothem = S * np.sqrt(3) / 2
        penalty = 0.0
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T  # (6, 6)
            violations = np.maximum(0, projs - apothem)
            penalty += np.sum(violations ** 2)
        return penalty

    def objective_smooth(x, t=0.01):
        if time.time() > deadline:
            return 1e10
        
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        
        S = get_min_side_smooth(positions, angles, t)
        
        penalty = 0.0
        for i in range(11):
            v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            n1 = get_normals(angles[i])
            for j in range(i+1, 11):
                v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                n2 = get_normals(angles[j])
                if not is_separated(v1, v2, n1, n2):
                    d = np.linalg.norm(positions[i] - positions[j])
                    penalty += max(0, 2.0 - d) ** 2
        
        pen = penalty + 0.1 * containment_penalty(positions, angles, S)
        
        return S + 100 * pen

    def objective_exact(x):
        if time.time() > deadline:
            return 1e10
        
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        
        S = get_min_side(positions, angles)
        
        penalty = 0.0
        for i in range(11):
            v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            n1 = get_normals(angles[i])
            for j in range(i+1, 11):
                v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                n2 = get_normals(angles[j])
                if not is_separated(v1, v2, n1, n2):
                    d = np.linalg.norm(positions[i] - positions[j])
                    penalty += max(0, 2.0 - d) ** 2
        
        pen = penalty + 0.1 * containment_penalty(positions, angles, S)
        
        return S + 100 * pen

    def validate(positions, angles):
        """Check if the configuration is valid (no overlaps, contained)."""
        S = get_min_side(positions, angles)
        if S < np.sqrt(11) - 1e-9:
            return False, S
        
        for i in range(11):
            v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            n1 = get_normals(angles[i])
            for j in range(i+1, 11):
                v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                n2 = get_normals(angles[j])
                if not is_separated(v1, v2, n1, n2):
                    return False, S
        
        # Check containment
        outer_normals = base_normals
        apothem = S * np.sqrt(3) / 2
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T
            if np.any(projs > apothem + 1e-6):
                return False, S
        
        return True, S

    # Initial guess
    sqrt3 = np.sqrt(3)
    init_pos = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    init_angles = np.zeros(11)
    
    x0 = np.concatenate([init_pos.flatten(), init_angles])
    
    best_x = x0.copy()
    best_val = objective_exact(x0)
    best_valid, best_S = validate(best_x[:22].reshape(11, 2), best_x[22:33])
    
    # Phase 1: Coarse optimization with smooth objective
    t_values = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for t in t_values:
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(objective_smooth, best_x, args=(t,), method='Nelder-Mead',
                              options={'maxiter': 15000, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_val:
                best_x = result.x
                best_val = result.fun
                valid, S = validate(best_x[:22].reshape(11, 2), best_x[22:33])
                if valid:
                    best_valid = True
                    best_S = S
        except Exception:
            pass
    
    # Phase 2: Multiple restarts from perturbed best-so-far
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 90:
            break
        
        # Perturb the best solution
        rng = np.random.default_rng(r * 12345 + 678)
        x_perturbed = best_x + rng.normal(0, 0.1, size=best_x.shape)
        
        try:
            result = minimize(objective_smooth, x_perturbed, args=(0.01,), method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_val:
                best_x = result.x
                best_val = result.fun
                valid, S = validate(best_x[:22].reshape(11, 2), best_x[22:33])
                if valid:
                    best_valid = True
                    best_S = S
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with exact objective
    if time.time() < deadline - 30:
        try:
            result2 = minimize(objective_exact, best_x, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-9})
            if result2.fun < best_val:
                best_x = result2.x
                best_val = result2.fun
        except Exception:
            pass
    
    # Extract best result
    positions = best_x[:22].reshape(11, 2)
    angles = best_x[22:33]
    
    # Final validation
    S = get_min_side(positions, angles)
    S = max(S, np.sqrt(11))
    
    valid = True
    for i in range(11):
        v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        n1 = get_normals(angles[i])
        for j in range(i+1, 11):
            v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
            n2 = get_normals(angles[j])
            if not is_separated(v1, v2, n1, n2):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = np.column_stack((positions, angles))
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = S

    return inner_hex_data, outer_hex_data, outer_hex_side_length
