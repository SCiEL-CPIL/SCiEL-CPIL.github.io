import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the equilateral triangle
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])
CENTROID = (V1 + V2 + V3) / 3.0

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    Input: (N, 2) array of (u, v). w = 1 - u - v.
    Output: (N, 2) array of (x, y).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    x = u * V1[0] + v * V2[0] + w * V3[0]
    y = u * V1[1] + v * V2[1] + w * V3[1]
    
    return np.column_stack((x, y))

def _project_bary(bary):
    """
    Project barycentric coordinates to the simplex u >= 0, v >= 0, u+v <= 1.
    Input: (N, 2) array of (u, v).
    Output: (N, 2) array of projected (u, v).
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Simple projection for speed and robustness
    # u >= 0, v >= 0, u+v <= 1
    # This is a projection onto the intersection of three half-spaces.
    # We can do a few iterations of clamping or use a more sophisticated method.
    # For L-BFGS-B, we can just use bounds in the optimizer, but for initialization 
    # or if we use a different method, this is useful.
    # Actually, L-BFGS-B handles bounds directly, so we might not need this 
    # during the optimization loop if we define bounds correctly.
    # But it's good for fixing initial points or post-processing.
    
    # Let's implement a simple iterative projection
    for _ in range(10):
        # u >= 0
        u = np.maximum(u, 0)
        # v >= 0
        v = np.maximum(v, 0)
        # u + v <= 1
        s = u + v
        if np.any(s > 1):
            # Scale down u and v to sum to 1
            # u = u * (1/s), v = v * (1/s)
            # But this might violate u>=0, v>=0 if s is small? No, s>=1.
            # However, simple scaling might not be the closest point.
            # For simplicity and speed, we'll use a heuristic or just clamp.
            # Clamping:
            # If u+v > 1, we need to reduce u and v.
            # The closest point on u+v=1 to (u,v) in L2 norm is (u,v)*(1/(u+v)).
            # Let's use that.
            mask = s > 1
            u[mask] = u[mask] / s[mask]
            v[mask] = v[mask] / s[mask]
            
    return np.column_stack((u, v))

def _min_area_bary(bary, beta=10.0):
    """
    Calculate the softmin of triangle areas for points defined by barycentric coords.
    Also returns the gradient of this softmin with respect to the barycentric coords.
    
    Args:
        bary: (N, 2) array of (u, v) barycentric coordinates.
        beta: Sharpness parameter for softmin. Higher is closer to true min.
        
    Returns:
        softmin_val: float, the approximated minimum area.
        grad: (N, 2) array, gradient of softmin_val w.r.t. bary.
    """
    points = _bary_to_cart(bary)
    
    # Calculate areas for all triples (i, j, k)
    # We want to compute A_ijk = 0.5 * |det(P_j - P_i, P_k - P_i)|
    # Using barycentric coordinates directly might be faster?
    # Area in Cartesian = 0.5 * |det|
    # Let's stick to Cartesian for area calculation, it's straightforward.
    
    n = len(points)
    areas = []
    # Store indices to compute gradients later? 
    # Or compute gradient via finite differences? No, analytical is better.
    # Let's compute the gradient analytically.
    
    # The softmin is: m_beta(A) = -1/beta * log(sum(exp(-beta * A_i)))
    # Gradient w.r.t. x: 
    # d(m_beta)/dx = -1/beta * d/dx log(sum(exp(-beta A_i)))
    # = -1/beta * [ sum(-beta * exp(-beta A_i) * dA_i/dx) ] / sum(exp(-beta A_i))
    # = sum( p_i * dA_i/dx ) where p_i = exp(-beta A_i) / sum(exp(-beta A_j))
    # So we need dA_i/dx for each triple.
    
    # This is complex to vectorize fully without storing all triples.
    # Given N=11, number of triples is 165.
    # We can loop over triples, compute area and gradient contribution.
    
    exp_vals = np.zeros(165)
    areas_arr = np.zeros(165)
    
    idx = 0
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                # Let's use the formula: 0.5 * | (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) |
                # Let D = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
                # A = 0.5 * |D|
                
                dx_j = pj[0] - pi[0]
                dy_j = pj[1] - pi[1]
                dx_k = pk[0] - pi[0]
                dy_k = pk[1] - pi[1]
                
                D = dx_j * dy_k - dy_j * dx_k
                A = 0.5 * abs(D)
                
                areas_arr[idx] = A
                exp_vals[idx] = np.exp(-beta * A)
                
                # Gradient of A w.r.t. points
                # A = 0.5 * |D|
                # dA/dP = 0.5 * sign(D) * dD/dP
                # If D == 0, gradient is undefined, but we can use 0 or subgradient.
                # Let's assume D != 0 for now.
                
                sD = np.sign(D)
                if sD == 0:
                    sD = 1.0 # Arbitrary, contribution is 0 anyway if we handle weights? 
                             # Actually if A=0, exp(-beta*0)=1, so it matters.
                             # But area 0 implies collinearity.
                             # Let's just use 1.
                
                # dD/dPi:
                # D = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
                # dD/dx_i = -(y_k-y_i) + (y_j-y_i) = y_j - y_k
                # dD/dy_i = -(x_k-x_i) + (x_j-x_i) = x_j - x_k
                
                # dD/dPj:
                # dD/dx_j = (y_k-y_i)
                # dD/dy_j = -(x_k-x_i)
                
                # dD/dPk:
                # dD/dx_k = -(y_j-y_i)
                # dD/dy_k = (x_j-x_i)
                
                grad_Pi = np.array([y_j - y_k, x_j - x_k]) * 0.5 * sD
                grad_Pj = np.array([y_k - y_i, -(x_k - x_i)]) * 0.5 * sD
                grad_Pk = np.array([-(y_j - y_i), x_j - x_i]) * 0.5 * sD
                
                # We need dA/dBary.
                # P = u*V1 + v*V2 + w*V3
                # dP/du = V1 - V3 (since w=1-u-v, dw/du=-1, dP/du = V1 - V3)
                # dP/dv = V2 - V3
                # dA/dBary_i = grad_Pi dot dP/du, grad_Pi dot dP/dv
                # Wait, P_i depends on bary_i.
                # grad_A/dx_i = grad_Pi[0], grad_A/dy_i = grad_Pi[1]
                # grad_A/du_i = grad_Pi[0] * (V1[0]-V3[0]) + grad_Pi[1] * (V1[1]-V3[1])
                # grad_A/dv_i = grad_Pi[0] * (V2[0]-V3[0]) + grad_Pi[1] * (V2[1]-V3[1])
                
                # Precompute basis vectors
                # dP/du = V1 - V3
                # dP/dv = V2 - V3
                # These are constant.
                
                # We will accumulate gradients in a separate array later.
                # For now, store the Cartesian gradients.
                # This is getting memory heavy if we store all 165 gradients.
                # Let's compute the softmin first, then the gradient.
                
                idx += 1
                
    # Compute softmin
    max_exp = np.max(exp_vals)
    if max_exp == 0:
        # All areas are huge? Unlikely.
        softmin_val = 0.0
    else:
        # Shift to avoid underflow
        shifted_exp = exp_vals - max_exp
        sum_exp = np.sum(np.exp(shifted_exp))
        log_sum = np.log(sum_exp) + max_exp
        softmin_val = -1.0 / beta * log_sum
        
    # Compute gradient
    # p_i = exp(-beta A_i) / sum(exp(-beta A_j))
    # To avoid underflow, use the shifted version.
    # p_i = exp(shifted_exp_i) / sum(exp(shifted_exp_j))
    
    if sum_exp == 0:
        grad = np.zeros_like(bary)
    else:
        probs = np.exp(shifted_exp) / sum_exp
        
        # Now we need to recompute the gradients or store them.
        # Recomputing is safer for memory.
        grad_bary = np.zeros_like(bary)
        
        idx = 0
        for i in range(n - 2):
            pi = points[i]
            for j in range(i + 1, n - 1):
                pj = points[j]
                for k in range(j + 1, n):
                    pk = points[k]
                    
                    dx_j = pj[0] - pi[0]
                    dy_j = pj[1] - pi[1]
                    dx_k = pk[0] - pi[0]
                    dy_k = pk[1] - pi[1]
                    
                    D = dx_j * dy_k - dy_j * dx_k
                    sD = np.sign(D)
                    if sD == 0:
                        sD = 1.0
                    
                    # Gradients of Area w.r.t Cartesian coords
                    # dA/dPi = 0.5 * sD * [y_j - y_k, x_j - x_k]
                    # dA/dPj = 0.5 * sD * [y_k - y_i, -(x_k - x_i)]
                    # dA/dPk = 0.5 * sD * [-(y_j - y_i), x_j - x_i]
                    
                    # Convert to Barycentric Gradients
                    # dA/du_i = dA/dx_i * (V1[0]-V3[0]) + dA/dy_i * (V1[1]-V3[1])
                    # dA/dv_i = dA/dx_i * (V2[0]-V3[0]) + dA/dy_i * (V2[1]-V3[1])
                    
                    # Constants
                    # V1=(0,0), V2=(1,0), V3=(0.5, sqrt(3)/2)
                    # V1-V3 = (-0.5, -sqrt(3)/2)
                    # V2-V3 = (0.5, -sqrt(3)/2)
                    
                    # dA/dx_i = 0.5 * sD * (y_j - y_k)
                    # dA/dy_i = 0.5 * sD * (x_j - x_k)
                    # dA/du_i = 0.5 * sD * [ (y_j - y_k)*(-0.5) + (x_j - x_k)*(-sqrt(3)/2) ]
                    # dA/dv_i = 0.5 * sD * [ (y_j - y_k)*(0.5) + (x_j - x_k)*(-sqrt(3)/2) ]
                    
                    # Similarly for j and k.
                    
                    # Let's define the transformation matrix M from Cartesian to Barycentric gradients
                    # dA/dBary = M^T * dA/dCart ?
                    # P = B * Bary
                    # dA/dBary = (dP/dBary)^T * dA/dP
                    # dP/du = V1 - V3
                    # dP/dv = V2 - V3
                    # So dA/du = dA/dP dot (V1-V3)
                    # dA/dv = dA/dP dot (V2-V3)
                    
                    # Let's compute these dots.
                    # dA/dPi = [gxi, gyi]
                    # dA/du_i = gxi * (V1[0]-V3[0]) + gyi * (V1[1]-V3[1])
                    # dA/dv_i = gxi * (V2[0]-V3[0]) + gyi * (V2[1]-V3[1])
                    
                    gxi = 0.5 * sD * (pj[1] - pk[1])
                    gyi = 0.5 * sD * (pj[0] - pk[0])
                    
                    g_x_j = 0.5 * sD * (pk[1] - pi[1])
                    g_y_j = 0.5 * sD * (-(pk[0] - pi[0]))
                    
                    g_x_k = 0.5 * sD * (-(pj[1] - pi[1]))
                    g_y_k = 0.5 * sD * (pj[0] - pi[0])
                    
                    # Weights
                    w_i = probs[idx]
                    w_j = probs[idx]
                    w_k = probs[idx]
                    
                    # Add to gradient
                    # grad_bary[i, 0] += w_i * dA/du_i
                    # grad_bary[i, 1] += w_i * dA/dv_i
                    
                    du_i = gxi * (V1[0] - V3[0]) + gyi * (V1[1] - V3[1])
                    dv_i = gxi * (V2[0] - V3[0]) + gyi * (V2[1] - V3[1])
                    
                    du_j = g_x_j * (V1[0] - V3[0]) + g_y_j * (V1[1] - V3[1])
                    dv_j = g_x_j * (V2[0] - V3[0]) + g_y_j * (V2[1] - V3[1])
                    
                    du_k = g_x_k * (V1[0] - V3[0]) + g_y_k * (V1[1] - V3[1])
                    dv_k = g_x_k * (V2[0] - V3[0]) + g_y_k * (V2[1] - V3[1])
                    
                    grad_bary[i, 0] += w_i * du_i
                    grad_bary[i, 1] += w_i * dv_i
                    
                    grad_bary[j, 0] += w_j * du_j
                    grad_bary[j, 1] += w_j * dv_j
                    
                    grad_bary[k, 0] += w_k * du_k
                    grad_bary[k, 1] += w_k * dv_k
                    
                    idx += 1
                    
    return softmin_val, grad_bary

def _objective_softmin(x_flat, beta=10.0):
    """
    Objective function for L-BFGS-B.
    x_flat: 1D array of length 2*N (u1, v1, u2, v2, ...)
    Returns: -softmin_val (to maximize)
    """
    bary = x_flat.reshape(N, 2)
    softmin_val, grad = _min_area_bary(bary, beta)
    # We want to maximize softmin_val, so minimize -softmin_val
    # Gradient of -softmin_val is -grad
    return -softmin_val, -grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_points = None
    
    # Bounds for barycentric coordinates: u >= 0, v >= 0, u+v <= 1
    # L-BFGS-B supports bounds.
    # We can set bounds for u and v individually as [0, 1].
    # The constraint u+v <= 1 is not handled by simple bounds.
    # However, if we start inside and the gradient points inwards, it might stay inside.
    # Or we can use a penalty method or project after each step.
    # L-BFGS-B doesn't support inequality constraints of the form u+v<=1 directly.
    # We can use a change of variables or penalty.
    # Let's try a penalty approach or just rely on the optimizer staying in the box [0,1]x[0,1]
    # and hope it doesn't go into u+v>1.
    # Actually, the simplex is a subset of the square [0,1]x[0,1].
    # If we use bounds [0,1] for u and v, we allow points outside the triangle.
    # We need to handle u+v <= 1.
    
    # Alternative: Use a different parametrization or penalty.
    # Let's add a penalty to the objective for u+v > 1.
    
    def _objective_with_penalty(x_flat, beta=10.0):
        bary = x_flat.reshape(N, 2)
        u = bary[:, 0]
        v = bary[:, 1]
        
        # Penalty for u+v > 1
        # s = u + v
        # Penalty = sum( max(0, s - 1)^2 )
        s = u + v
        viol = np.maximum(s - 1, 0)
        penalty = np.sum(viol ** 2)
        
        # Also penalty for u<0 or v<0, though bounds handle this.
        # But just in case.
        u_viol = np.maximum(-u, 0)
        v_viol = np.maximum(-v, 0)
        penalty += np.sum(u_viol ** 2) + np.sum(v_viol ** 2)
        
        softmin_val, grad = _min_area_bary(bary, beta)
        
        # Gradient of penalty
        # d(penalty)/du = 2 * viol * 1 (if s>1) + 2 * u_viol * (-1)
        # d(penalty)/dv = 2 * viol * 1 (if s>1) + 2 * v_viol * (-1)
        
        grad_pen_u = np.zeros_like(u)
        grad_pen_v = np.zeros_like(v)
        
        mask_s = s > 1
        grad_pen_u[mask_s] = 2 * viol[mask_s]
        grad_pen_v[mask_s] = 2 * viol[mask_s]
        
        mask_u = u < 0
        grad_pen_u[mask_u] = -2 * u[mask_u] # 2 * (-u) * (-1) = -2u? No.
        # Penalty = u^2 if u<0. d/du = 2u.
        # Wait, penalty = max(0, -u)^2. If u<0, penalty = u^2. d/du = 2u.
        # If u>=0, penalty = 0. d/du = 0.
        # So grad_pen_u = 2u for u<0, 0 otherwise.
        
        mask_v = v < 0
        grad_pen_v[mask_v] = 2 * v[mask_v]
        
        # Total gradient
        # Objective = -softmin + penalty
        # Grad = -grad_softmin + grad_penalty
        
        total_grad = np.column_stack((-grad[:, 0] + grad_pen_u, -grad[:, 1] + grad_pen_v))
        
        return -softmin_val + penalty, total_grad.flatten()
    
    # Strategy: Multiple restarts with L-BFGS-B
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize points in barycentric coordinates
        if restart == 0:
            # Structured initialization
            # Center
            bary_pts = [np.array([1/3, 1/3])]
            # 6 points around
            for k in range(6):
                angle = np.pi / 3 * k
                # In barycentric, "around" is tricky.
                # Let's just sample random points inside for simplicity in init.
                pass
            # Just use random for now, it's faster to code.
            bary_pts = []
            for i in range(N):
                while True:
                    u = rng.uniform(0, 1)
                    v = rng.uniform(0, 1)
                    if u + v <= 1:
                        bary_pts.append([u, v])
                        break
            bary = np.array(bary_pts)
        else:
            bary = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u = rng.uniform(0, 1)
                    v = rng.uniform(0, 1)
                    if u + v <= 1:
                        bary[i] = [u, v]
                        break
        
        x0 = bary.flatten()
        
        # Bounds: u in [0,1], v in [0,1]
        bounds = [(0, 1)] * (2 * N)
        
        options = {
            'maxiter': 200,
            'gtol': 1e-5,
            'disp': False
        }
        
        try:
            res = minimize(_objective_with_penalty, x0, method='L-BFGS-B', 
                           jac=True, bounds=bounds, options=options)
            
            new_bary = res.x.reshape(N, 2)
            
            # Verify validity
            u = new_bary[:, 0]
            v = new_bary[:, 1]
            valid = np.all(u >= -1e-5) and np.all(v >= -1e-5) and np.all(u + v <= 1 + 1e-5)
            
            if valid:
                # Project to ensure strict validity
                new_bary = _project_bary(new_bary)
                points = _bary_to_cart(new_bary)
                
                # Calculate true min area for scoring
                # Reuse logic from _min_area_bary but with beta=infinity (true min)
                # Or just compute it directly.
                min_area = np.inf
                n = len(points)
                for i in range(n - 2):
                    pi = points[i]
                    for j in range(i + 1, n - 1):
                        pj = points[j]
                        for k in range(j + 1, n):
                            pk = points[k]
                            dx_j = pj[0] - pi[0]
                            dy_j = pj[1] - pi[1]
                            dx_k = pk[0] - pi[0]
                            dy_k = pk[1] - pi[1]
                            D = dx_j * dy_k - dy_j * dx_k
                            A = 0.5 * abs(D)
                            if A < min_area:
                                min_area = A
                
                if min_area > best_score:
                    best_score = min_area
                    best_points = points.copy()
                    
        except Exception:
            pass

    if best_points is None:
        # Fallback
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                p = rng.uniform(0, 1, 2)
                if p[0] + p[1] <= 1:
                    # Convert bary to cart
                    u, v = p
                    w = 1 - u - v
                    x = u * V1[0] + v * V2[0] + w * V3[0]
                    y = u * V1[1] + v * V2[1] + w * V3[1]
                    points[i] = [x, y]
                    break
        best_points = points
        
    return best_points
