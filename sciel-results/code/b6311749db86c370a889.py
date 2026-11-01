import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate 11 points on a triangular lattice
    # Axial coordinates (q, r)
    # This is a compact "flower" shape: center + 6 neighbors + 4 from the next ring
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Use a very small expansion to ensure strict disjointness (no touching)
    # while keeping the size minimal.
    SCALE = 1.0 + 1e-6
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    n_pts = len(pts)
    
    def solve_lp(cx, cy, beta):
        """
        Solves the LP for given outer center and rotation.
        Returns D (the max projection offset) or inf on failure.
        """
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: D only
        # Constraint: | (p_i - c) . n_k | <= D for all i, k
        # i.e., (p_i - c) . n_k <= D AND -(p_i - c) . n_k <= D
        
        c_obj = np.array([1.0])  # Minimize D
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx)nx + (py - cy)ny <= D
                # px*nx - cx*nx + py*ny - cy*ny - D <= 0
                # -D <= -(px*nx + py*ny) + cx*nx + cy*ny
                A_ub.append([-1.0])
                b_ub.append(-(px * nx + py * ny) + cx * nx + cy * ny)
                
                # -((px - cx)nx + (py - cy)ny) <= D
                # (cx - px)nx + (cy - py)ny - D <= 0
                # -D <= -(cx - px)nx - (cy - py)ny
                A_ub.append([-1.0])
                b_ub.append(-(cx - px) * nx - (cy - py) * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x[0]
            else:
                return float('inf')
        except Exception:
            return float('inf')

    def solve_for_beta(beta, cx_init=0.0, cy_init=0.0):
        """
        Solves the full LP for a given outer rotation beta.
        Returns (cx, cy, S) or (None, None, inf)
        """
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None, None, float('inf')
        
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, D
        c = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return cx, cy, S
            else:
                return None, None, float('inf')
        except Exception:
            return None, None, float('inf')

    def smooth_objective(x):
        """
        Smooth surrogate for the joint optimization.
        x = [cx, cy, beta, S]
        Uses soft-max to smooth the max over all constraints.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        cx, cy, beta, S = x
        S = max(S, 0.1)  # Ensure positive
        
        # Normals of the outer hexagon
        normals = np.array([[math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                            math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
        
        # For each point i and normal k, compute the required constraint:
        # (p_i - c) . n_k + APOTHEM <= S * APOTHEM
        # (p_i - c) . n_k <= S * APOTHEM - APOTHEM
        # Let margin_k_i = S * APOTHEM - APOTHEM - (p_i - c) . n_k
        # We need all margins >= 0
        
        # Compute all margins
        margins = np.zeros((n_pts, 6))
        for i in range(n_pts):
            for k in range(6):
                margins[i, k] = S * APOTHEM - APOTHEM - ((pts[i, 0] - cx) * normals[k, 0] + (pts[i, 1] - cy) * normals[k, 1])
        
        # Also need the negative side:
        # -(p_i - c) . n_k + APOTHEM <= S * APOTHEM
        # -(p_i - c) . n_k <= S * APOTHEM - APOTHEM
        margins_neg = np.zeros((n_pts, 6))
        for i in range(n_pts):
            for k in range(6):
                margins_neg[i, k] = S * APOTHEM - APOTHEM - (-(pts[i, 0] - cx) * normals[k, 0] - (pts[i, 1] - cy) * normals[k, 1])
        
        # Combine all margins
        all_margins = np.concatenate([margins.flatten(), margins_neg.flatten()])
        
        # Soft-min: we want the minimum margin to be as large as possible
        # Minimize -min(margins) = minimize soft_min(-margins)
        # soft_min(a) = -log-sum-exp(-a) / tau
        tau = 0.05
        neg_margins = -all_margins
        max_neg = np.max(neg_margins)
        # Numerical stability
        if max_neg > 100:
            lse = max_neg + np.log(np.sum(np.exp(neg_margins - max_neg)))
        else:
            lse = np.log(np.sum(np.exp(neg_margins)))
        soft_min = -lse / tau
        
        # Objective: minimize S + penalty for violated constraints
        # We want S to be small, and all margins to be >= 0
        # Use: S + lambda * max(0, -soft_min)
        penalty = max(0.0, soft_min)  # soft_min is negative when constraints violated
        obj = S + 10.0 * penalty
        
        return obj

    def smooth_objective_grad(x):
        """
        Gradient of the smooth objective for better convergence.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10, np.zeros(4)
        
        cx, cy, beta, S = x
        S = max(S, 0.1)
        
        normals = np.array([[math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                            math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
        
        margins = np.zeros((n_pts, 12))
        idx = 0
        for i in range(n_pts):
            for k in range(6):
                margins[i, idx] = S * APOTHEM - APOTHEM - ((pts[i, 0] - cx) * normals[k, 0] + (pts[i, 1] - cy) * normals[k, 1])
                idx += 1
            for k in range(6):
                margins[i, idx] = S * APOTHEM - APOTHEM - (-(pts[i, 0] - cx) * normals[k, 0] - (pts[i, 1] - cy) * normals[k, 1])
                idx += 1
        
        all_margins = margins.flatten()
        tau = 0.05
        neg_margins = -all_margins
        max_neg = np.max(neg_margins)
        if max_neg > 100:
            exps = np.exp(neg_margins - max_neg)
            lse = max_neg + np.log(np.sum(exps))
        else:
            exps = np.exp(neg_margins)
            lse = np.log(np.sum(exps))
        soft_min = -lse / tau
        
        penalty = max(0.0, soft_min)
        obj = S + 10.0 * penalty
        
        # Gradient computation
        # d(obj)/dS = 1 + 10 * d(penalty)/dS if penalty > 0
        # d(soft_min)/dS = (1/tau) * sum(exp(-m_j) * dm_j/dS) / sum(exp(-m_j))
        # dm_j/dS = APOTHEM for each margin
        
        if penalty > 0:
            weights = exps / np.sum(exps)
            d_softmin_dS = (1.0 / tau) * APOTHEM  # sum(weights * APOTHEM) = APOTHEM
            grad_S = 1.0 + 10.0 * d_softmin_dS
            
            # d(soft_min)/dcx = (1/tau) * sum(exp(-m_j) * dm_j/dcx) / sum(exp(-m_j))
            # dm_j/dcx for positive side: -n_k_x
            # dm_j/dcx for negative side: n_k_x
            d_softmin_dcx = 0.0
            d_softmin_dcy = 0.0
            d_softmin_dbeta = 0.0
            
            idx = 0
            for i in range(n_pts):
                for k in range(6):
                    w = weights[idx]
                    # dm/dcx = -normals[k,0]
                    d_softmin_dcx += w * (-normals[k, 0])
                    d_softmin_dcy += w * (-normals[k, 1])
                    # dm/dbeta: d/d beta of (p_i - c).n_k = -(p_i-c).(-sin, cos) * d/d beta of angle
                    # n_k = (cos(a), sin(a)) where a = beta + pi/6 + k*pi/3
                    # dn/dbeta = (-sin(a), cos(a))
                    # d((p_i-c).n_k)/dbeta = (p_i-c).(-sin(a), cos(a))
                    # dm/dbeta = -d((p_i-c).n_k)/dbeta = -(p_i-c).(-sin(a), cos(a))
                    a = beta + math.pi / 6.0 + k * math.pi / 3.0
                    d_softmin_dbeta += w * -((pts[i, 0] - cx) * (-math.sin(a)) + (pts[i, 1] - cy) * math.cos(a))
                    idx += 1
                for k in range(6):
                    w = weights[idx]
                    # dm/dcx = normals[k,0]
                    d_softmin_dcx += w * (normals[k, 0])
                    d_softmin_dcy += w * (normals[k, 1])
                    a = beta + math.pi / 6.0 + k * math.pi / 3.0
                    # dm/dbeta for negative side: d/d beta of -(p_i-c).n_k = (p_i-c).(-sin(a), cos(a))
                    # dm/dbeta = -d(-(p_i-c).n_k)/dbeta = -(p_i-c).(-sin(a), cos(a))... wait
                    # Let me redo: margin_neg = S*APOTHEM - APOTHEM - (-(p_i-c).n_k)
                    # d(margin_neg)/dbeta = -d(-(p_i-c).n_k)/dbeta = d((p_i-c).n_k)/dbeta
                    # d((p_i-c).n_k)/dbeta = (p_i-c).dn_k/dbeta = (p_i-c).(-sin(a), cos(a))
                    d_softmin_dbeta += w * ((pts[i, 0] - cx) * (-math.sin(a)) + (pts[i, 1] - cy) * math.cos(a))
                    idx += 1
            
            d_softmin_dcx /= tau
            d_softmin_dcy /= tau
            d_softmin_dbeta /= tau
            
            grad_cx = 10.0 * d_softmin_dcx
            grad_cy = 10.0 * d_softmin_dcy
            grad_beta = 10.0 * d_softmin_dbeta
        else:
            grad_S = 1.0
            grad_cx = 0.0
            grad_cy = 0.0
            grad_beta = 0.0
        
        return obj, np.array([grad_cx, grad_cy, grad_beta, grad_S])

    # Objective function for minimize_scalar
    def obj(beta):
        _, _, S = solve_for_beta(beta)
        return S

    # Initial guess: beta = 0
    best_beta = 0.0
    best_cx, best_cy, best_S = solve_for_beta(best_beta)
    
    # If initial solve failed, try a small offset or return fallback
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Phase 1: Minimize over beta in [0, pi/6] using the LP
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 300, 'xatol': 1e-8})
            if res.fun < best_S:
                best_beta = res.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
                
                # Second pass with tighter tolerance
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    local_bounds = (max(0, best_beta - 0.02), min(math.pi / 6, best_beta + 0.02))
                    res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                          options={'maxiter': 500, 'xatol': 1e-10})
                    if res2.fun < best_S:
                        best_beta = res2.x
                        best_cx, best_cy, best_S = solve_for_beta(best_beta)
    except Exception:
        pass

    # Phase 2: Joint optimization using smooth surrogate
    # Variables: [cx, cy, beta, S]
    x0 = np.array([best_cx, best_cy, best_beta, best_S])
    
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Try L-BFGS-B with the smooth objective
            res_smooth = minimize(smooth_objective, x0, method='L-BFGS-B',
                                  bounds=[(-2, 2), (-2, 2), (0, math.pi/6), (0.5, 5.0)],
                                  options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res_smooth.fun < best_S and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx_new, cy_new, beta_new, S_new = res_smooth.x
                # Verify with exact LP
                cx_lp, cy_lp, S_lp = solve_for_beta(beta_new, cx_new, cy_new)
                if S_lp < best_S:
                    best_cx, best_cy, best_S = cx_lp, cy_lp, S_lp
                    best_beta = beta_new
    except Exception:
        pass

    # Phase 3: Final high-precision LP polish around the best beta found
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Very fine local search
            local_bounds = (max(0, best_beta - 0.005), min(math.pi / 6, best_beta + 0.005))
            res3 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                  options={'maxiter': 1000, 'xatol': 1e-12})
            if res3.fun < best_S:
                best_beta = res3.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
                
                # Even tighter
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    local_bounds2 = (max(0, best_beta - 0.001), min(math.pi / 6, best_beta + 0.001))
                    res4 = minimize_scalar(obj, bounds=local_bounds2, method='bounded',
                                          options={'maxiter': 2000, 'xatol': 1e-14})
                    if res4.fun < best_S:
                        best_beta = res4.x
                        best_cx, best_cy, best_S = solve_for_beta(best_beta)
    except Exception:
        pass

    # Final check
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
