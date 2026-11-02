import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define a compact cluster of 11 unit hexagons in axial coordinates
    # This is a 3x3 block plus 2 extra on top
    axial_coords = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # We need to find the minimum S such that all centers are inside a hexagon of side S.
    # The hexagon is centered at (cx, cy) and rotated by angle theta.
    # The constraint for each center (xi, yi) is:
    # max_k ( dot( (xi, yi) - (cx, cy), n_k(theta) ) ) <= S * APOTHEM
    # where n_k(theta) are the 6 outward normals of the unit hexagon rotated by theta.
    # n_k(theta) = (cos(theta + pi/6 + k*pi/3), sin(theta + pi/6 + k*pi/3))
    
    # For a fixed theta, the optimal (cx, cy, S) can be found.
    # Actually, for fixed theta, S is determined by the max projection.
    # cx, cy should be the center of the bounding box of the projections?
    # No, the hexagon is symmetric. The optimal center is the center of the minimum enclosing hexagon.
    # For a fixed orientation theta, the minimum S is half the range of projections onto each normal, 
    # but the center can shift.
    # Actually, for a fixed theta, the problem is to find cx, cy, S minimizing S
    # subject to:
    # -S*APOTHEM <= dot(p_i - c, n_k) <= S*APOTHEM for all i, k.
    # This is a Linear Program.
    
    # We can scan theta from 0 to pi/6 (60 degrees) due to symmetry.
    
    def calc_min_S(theta):
        """
        Calculate minimum S for a given theta using LP.
        Variables: cx, cy, S
        Constraints:
        For each point i and normal k:
        dot(p_i, n_k) - dot(c, n_k) <= S * APOTHEM
        dot(c, n_k) - dot(p_i, n_k) <= S * APOTHEM
        
        Let u = cx, v = cy, t = S * APOTHEM.
        Minimize t.
        Subject to:
        p_i_x n_k_x + p_i_y n_k_y - u n_k_x - v n_k_y <= t
        u n_k_x + v n_k_y - p_i_x n_k_x - p_i_y n_k_y <= t
        
        This is a standard LP.
        """
        from scipy.optimize import linprog
        
        n_points = len(centers)
        n_normals = 6
        
        # Normals
        normals = np.array([
            (math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(n_normals)
        ])
        
        # Objective: minimize t (index 2)
        c_obj = np.array([0.0, 0.0, 1.0])
        
        # Constraints: A_ub * x <= b_ub
        # x = [cx, cy, t]
        # Number of constraints: 2 * n_points * n_normals
        n_constraints = 2 * n_points * n_normals
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        idx = 0
        for i in range(n_points):
            px, py = centers[i]
            for k in range(n_normals):
                nx, ny = normals[k]
                
                # Constraint 1: p . n - c . n <= t
                # -nx * cx - ny * cy - 1 * t <= - (px * nx + py * ny)
                A_ub[idx, 0] = -nx
                A_ub[idx, 1] = -ny
                A_ub[idx, 2] = -1.0
                b_ub[idx] = -(px * nx + py * ny)
                idx += 1
                
                # Constraint 2: c . n - p . n <= t
                # nx * cx + ny * cy - 1 * t <= px * nx + py * ny
                A_ub[idx, 0] = nx
                A_ub[idx, 1] = ny
                A_ub[idx, 2] = -1.0
                b_ub[idx] = px * nx + py * ny
                idx += 1
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                t = res.x[2]
                S = t / APOTHEM
                cx = res.x[0]
                cy = res.x[1]
                return S, cx, cy
            else:
                return float('inf'), 0, 0
        except Exception:
            return float('inf'), 0, 0

    # Scan theta
    best_S = float('inf')
    best_theta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Coarse scan
    n_steps = 100
    for i in range(n_steps):
        theta = i * (math.pi / 6.0) / n_steps
        S, cx, cy = calc_min_S(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
            best_cx = cx
            best_cy = cy
            
    # Fine scan around best
    step = (math.pi / 6.0) / n_steps
    for i in range(-10, 11):
        theta = best_theta + i * step / 10.0
        if theta < 0 or theta >= math.pi/6: continue
        S, cx, cy = calc_min_S(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
            best_cx = cx
            best_cy = cy

    # Polish with Nelder-Mead on the exact objective if time permits
    # The objective function is S(theta, cx, cy) but cx, cy are determined by theta in the LP.
    # So we just minimize S(theta).
    
    def obj_theta(theta_val):
        S, _, _ = calc_min_S(theta_val)
        return S
        
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize(obj_theta, best_theta, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_theta = res.x[0]
                # Recalculate center
                _, best_cx, best_cy = calc_min_S(best_theta)
    except Exception:
        pass

    # Final Result
    # The inner hexagons are at centers, all with angle 0 (aligned with lattice)
    # The outer hexagon is centered at (best_cx, best_cy), side best_S, rotated by best_theta
    
    inner = []
    for x, y in centers:
        inner.append([float(x), float(y), 0.0])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(math.degrees(best_theta))
    
    return inner, center, side, rotation
