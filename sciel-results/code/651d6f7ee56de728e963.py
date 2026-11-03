import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute normals for unit hexagon separation (angle 0)
    # Separation condition: |dot(d, n)| >= sqrt(3) for all 6 normals
    # Normals are perpendicular to sides. For a hexagon with flat top (angle 0), sides are at 0, 60, ...
    # Normals are at 30, 90, ...
    hex_normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    
    # Pair indices for constraints
    pair_indices = [(i, j) for i in range(N) for j in range(i+1, N)]
    n_pairs = len(pair_indices)

    def get_axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_exact_S_and_center(pts, beta):
        """
        Given fixed inner hexagon centers (pts) and outer hexagon rotation (beta),
        find the minimal outer hexagon side length S and center (cx, cy) that encloses them.
        This is a Linear Programming problem.
        """
        # Outer hexagon normals
        outer_normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        # Variables: cx, cy, D (where D is the apothem of the outer hexagon)
        # Objective: minimize D
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        # Constraints:
        # For each normal n_k:
        # n_k . (p_i - c) <= D  =>  n_k . p_i - n_k . c <= D  =>  -n_k.x*cx - n_k.y*cy - D <= -n_k . p_i
        # -n_k . (p_i - c) <= D =>  -n_k . p_i + n_k . c <= D  =>  n_k.x*cx + n_k.y*cy - D <= n_k . p_i
        
        for nx, ny in outer_normals:
            for px, py in pts:
                dot = px * nx + py * ny
                # Constraint 1: dot - (nx*cx + ny*cy) <= D  =>  -nx*cx - ny*cy - D <= -dot
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-dot)
                # Constraint 2: -dot + (nx*cx + ny*cy) <= D  =>  nx*cx + ny*cy - D <= dot
                A_ub.append([nx, ny, -1.0])
                b_ub.append(dot)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def objective_and_grad(x):
        """
        Objective function for SLSQP.
        x: [cx_1, cy_1, ..., cx_N, cy_N, beta]
        Returns: (value, gradient)
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), np.zeros_like(x)
            
        pts = x[:2*N].reshape(N, 2)
        beta = x[2*N]
        
        # 1. Calculate geometric cost (S) via LP
        S, center, D = calc_exact_S_and_center(pts, beta)
        if S == float('inf'):
             # If LP fails, return large value
             return 1000.0, np.zeros_like(x)
        
        # 2. Calculate penalty for overlaps
        # Constraint: |dot(d, n)| >= sqrt(3)
        # Penalty: sum( max(0, sqrt(3) - |dot(d, n)|)^2 )
        penalty = 0.0
        grad_pts = np.zeros((N, 2))
        
        sqrt3 = math.sqrt(3)
        penalty_weight = 100.0
        
        for i, j in pair_indices:
            d = pts[i] - pts[j]
            # dot products with normals
            projs = np.abs(d @ hex_normals.T)
            violations = np.maximum(0, sqrt3 - projs)
            penalty += np.sum(violations**2)
            
            # Gradient calculation (approximate or exact?)
            # Let f = max(0, sqrt3 - |d.n|)^2
            # If violated: f = (sqrt3 - |d.n|)^2
            # df/d(d.n) = -2(sqrt3 - |d.n|) * sign(d.n)
            # d(d.n)/d(pts[i]) = n, d(d.n)/d(pts[j]) = -n
            for k in range(6):
                v = violations[k]
                if v > 0:
                    # f = (sqrt3 - |dot|)^2
                    # df/d(dot) = -2 * (sqrt3 - |dot|) * sign(dot)
                    dot_val = d @ hex_normals[k]
                    sign_dot = 1.0 if dot_val > 0 else -1.0
                    grad_dot = -2.0 * v * sign_dot
                    
                    n_vec = hex_normals[k]
                    grad_pts[i] += grad_dot * n_vec
                    grad_pts[j] -= grad_dot * n_vec
        
        total_penalty = penalty_weight * penalty
        total_cost = S + total_penalty
        
        # Gradient
        # Gradient of S w.r.t pts and beta is tricky. 
        # Since we are using SLSQP, we can use finite differences for the gradient if analytic is too complex,
        # or we can just return the function value and let SLSQP approximate? 
        # No, SLSQP can use finite difference approx if grad is None. 
        # But providing grad is faster. 
        # However, S from LP is piecewise linear/non-smooth. 
        # Let's just use finite difference approximation for the whole thing to be safe and simple,
        # or provide the penalty gradient and assume SLSQP handles the rest?
        # Actually, let's return None for gradient to let SLSQP use numerical differentiation.
        # This is slower per step but robust. Given N=11, 23 variables, it should be okay.
        
        return total_cost, None

    # Initial candidates
    candidates = []
    
    # 1. Hexagonal spiral (standard packing)
    axial_spiral = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    pts0 = get_axial_to_cartesian(axial_spiral)
    candidates.append(np.append(pts0.flatten(), 0.0))
    
    # 2. Slightly perturbed spiral
    np.random.seed(42)
    pts1 = pts0 + np.random.randn(N, 2) * 0.1
    candidates.append(np.append(pts1.flatten(), 0.0))
    
    # 3. Another axial configuration
    axial_alt = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    pts2 = get_axial_to_cartesian(axial_alt)
    candidates.append(np.append(pts2.flatten(), 0.0))
    
    # 4. Random cluster
    pts3 = np.random.randn(N, 2) * 1.5
    candidates.append(np.append(pts3.flatten(), 0.0))

    best_S = float('inf')
    best_x = None
    best_center = None

    for x0 in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            res = minimize(objective_and_grad, x0, method='SLSQP', 
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success:
                # Evaluate the actual geometric S for the result
                pts_res = res.x[:2*N].reshape(N, 2)
                beta_res = res.x[2*N]
                S_res, center_res, _ = calc_exact_S_and_center(pts_res, beta_res)
                
                # Check validity (no overlaps)
                valid = True
                sqrt3 = math.sqrt(3)
                for i, j in pair_indices:
                    d = pts_res[i] - pts_res[j]
                    projs = np.abs(d @ hex_normals.T)
                    if np.min(projs) < sqrt3 - 1e-6:
                        valid = False
                        break
                
                if valid and S_res < best_S:
                    best_S = S_res
                    best_x = res.x
                    best_center = center_res
                    
        except Exception as e:
            pass

    # Fallback to best candidate if optimization failed
    if best_x is None:
        # Use the first candidate (spiral)
        pts_res = get_axial_to_cartesian(axial_spiral)
        beta_res = 0.0
        best_S, best_center, _ = calc_exact_S_and_center(pts_res, beta_res)
        best_x = np.append(pts_res.flatten(), beta_res)
        
    # Final Polish:
    # We have best_x which gives a valid configuration with side best_S.
    # We can try to further reduce S by fixing the positions and optimizing beta only?
    # Or fixing beta and optimizing positions?
    # Actually, the LP in calc_exact_S_and_center already finds the optimal center and S for given positions and beta.
    # So if we have a valid set of positions, the S is exact for that beta.
    # We could do a final 1D optimization over beta for the best positions.
    
    pts_best = best_x[:2*N].reshape(N, 2)
    
    def beta_obj(b):
        S, _, _ = calc_exact_S_and_center(pts_best, b)
        return S
    
    # Optimize beta
    try:
        from scipy.optimize import minimize_scalar
        res_beta = minimize_scalar(beta_obj, bounds=(0, math.pi/6), method='bounded', options={'maxiter': 100, 'xatol': 1e-10})
        if res_beta.fun < best_S:
            best_S = res_beta.fun
            best_beta = res_beta.x
            _, best_center, _ = calc_exact_S_and_center(pts_best, best_beta)
        else:
            best_beta = best_x[2*N]
    except Exception:
        best_beta = best_x[2*N]

    # Construct output
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for i in range(N):
        x = float(pts_best[i, 0])
        y = float(pts_best[i, 1])
        inner.append([x, y, angle_deg])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

# Dummy call to ensure imports work
# construct()
