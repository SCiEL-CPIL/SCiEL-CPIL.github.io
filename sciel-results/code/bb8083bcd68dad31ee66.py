# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses a symmetric 6-fold configuration with LP-based optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    SQRT3 = np.sqrt(3.0)
    
    # Hexagon vertices function
    def hex_vertices(cx, cy, s, a_rad):
        angles = a_rad + np.radians(60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    # Compute minimum side length of enclosing hexagon given inner hex positions/angles
    def enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle_deg):
        ocx, ocy = outer_center
        oa = np.radians(outer_angle_deg)
        
        all_verts = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            a_rad = np.radians(inner_angles[i])
            v = hex_vertices(cx, cy, 1.0, a_rad)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        all_verts_shifted = all_verts - np.array([ocx, ocy])
        
        min_s = 0.0
        for i in range(6):
            a_i = oa + np.radians(60 * i)
            a_ip1 = oa + np.radians(60 * (i + 1))
            e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            crosses = all_verts_shifted[:, 0] * e_unit[1] - all_verts_shifted[:, 1] * e_unit[0]
            min_cross = np.min(crosses)
            required_s = -2 * min_cross / SQRT3
            if required_s > min_s:
                min_s = required_s
        
        return max(min_s, 0.0)
    
    # Build symmetric configuration from parameters (r1, r2, angle_offset)
    # r1: radius for inner ring (6 hexagons around center)
    # r2: radius for outer ring (4 hexagons)
    # angle_offset: rotation offset for the entire configuration
    def build_symmetric_config(r1, r2, angle_offset_deg):
        """
        Returns 11 inner hex positions and angles in a symmetric arrangement.
        Hex 0: center
        Hex 1-6: inner ring at radius r1, angles 0, 60, 120, 180, 240, 300
        Hex 7-10: outer ring at radius r2, angles 30, 150, 210, 330
        """
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        
        # Center hexagon
        positions[0] = [0, 0]
        angles[0] = 0
        
        # Inner ring: 6 hexagons
        for i in range(6):
            theta = np.radians(60 * i + angle_offset_deg)
            positions[i+1] = [r1 * np.cos(theta), r1 * np.sin(theta)]
            angles[i+1] = angle_offset_deg  # same orientation as outer hex
        
        # Outer ring: 4 hexagons at 30, 150, 210, 330 degrees
        outer_angles = [30, 150, 210, 330]
        for i, theta_deg in enumerate(outer_angles):
            theta = np.radians(theta_deg + angle_offset_deg)
            positions[i+7] = [r2 * np.cos(theta), r2 * np.sin(theta)]
            angles[i+7] = angle_offset_deg
        
        return positions, angles
    
    # Compute minimum gap between two hexagons
    def min_gap_between_hexagons(v1, v2):
        min_gap = np.inf
        for vi in [v1, v2]:
            for k in range(6):
                e = vi[(k+1)%6] - vi[k]
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-15:
                    n = n / norm
                proj1 = v1 @ n
                proj2 = v2 @ n
                gap = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                if gap < min_gap:
                    min_gap = gap
        return min_gap
    
    # Objective: minimize outer hex side for symmetric config
    def objective_symmetric(params):
        if time.time() > deadline:
            return 1e10
        if not np.all(np.isfinite(params)):
            return 1e10
        
        r1, r2, angle_offset = params
        
        # Ensure r1, r2 are positive
        r1 = max(r1, 0.5)
        r2 = max(r2, 0.5)
        
        positions, angles = build_symmetric_config(r1, r2, angle_offset)
        
        # Check separation between all pairs
        inner_verts = np.zeros((11, 6, 2))
        for i in range(11):
            inner_verts[i] = hex_vertices(positions[i, 0], positions[i, 1], 1.0, np.radians(angles[i]))
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                gap = min_gap_between_hexagons(inner_verts[i], inner_verts[j])
                if gap < 1e-6:
                    penalty += (1e-6 - gap) ** 2
        
        # Compute enclosing hexagon side
        outer_center = np.array([0.0, 0.0])
        outer_angle = angle_offset
        s = enclosing_hex_side(positions, angles, outer_center, outer_angle)
        s = max(s, np.sqrt(11))
        
        return s + 1000 * penalty
    
    # Multi-start optimization over the 3-parameter symmetric space
    np.random.seed(42)
    
    best_params = np.array([2.0, 4.0, 0.0])  # initial guess
    best_val = objective_symmetric(best_params)
    
    # Try multiple starting points
    start_points = [
        [2.0, 4.0, 0.0],
        [2.0, 4.0, 30.0],
        [2.0, 4.0, 15.0],
        [1.8, 3.8, 0.0],
        [2.2, 4.2, 0.0],
        [1.5, 3.5, 0.0],
        [2.5, 4.5, 0.0],
        [2.0, 4.0, 45.0],
        [1.9, 3.9, 10.0],
        [2.1, 4.1, 20.0],
    ]
    
    # Add random starts
    for _ in range(20):
        r1 = np.random.uniform(1.5, 2.5)
        r2 = np.random.uniform(3.0, 5.0)
        ao = np.random.uniform(-30, 30)
        start_points.append([r1, r2, ao])
    
    for sp in start_points:
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(objective_symmetric, sp, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Refine with COBYLA for higher precision
    if time.time() < deadline - 30:
        try:
            result2 = minimize(objective_symmetric, best_params, method='COBYLA',
                              options={'maxiter': 5000, 'rhobeg': 1e-8, 'tol': 1e-14})
            
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Additional refinement passes
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        try:
            x_perturbed = best_params + np.random.randn(3) * 0.01
            result3 = minimize(objective_symmetric, x_perturbed, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14})
            
            if result3.fun < best_val:
                best_val = result3.fun
                best_params = result3.x.copy()
        except Exception:
            pass
    
    # Extract best solution
    r1, r2, angle_offset = best_params
    r1 = max(r1, 0.5)
    r2 = max(r2, 0.5)
    
    positions, angles = build_symmetric_config(r1, r2, angle_offset)
    
    # Compute actual enclosing side
    outer_center = np.array([0.0, 0.0])
    s = enclosing_hex_side(positions, angles, outer_center, angle_offset)
    s = max(s, np.sqrt(11))
    
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, angle_offset])
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
