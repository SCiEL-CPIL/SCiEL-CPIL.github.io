import numpy as np
import time
import itertools
from scipy.optimize import linprog

N = 11
SQRT3 = np.sqrt(3)

# Precompute all triples (i, j, k) with i < j < k
_TRIPLES = list(itertools.combinations(range(N), 3))
_T_IDX = np.array(_TRIPLES)
_NUM_TRIPLES = len(_TRIPLES)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum triangle area (normalized).
    """
    pi = p[_T_IDX[:, 0]]
    pj = p[_T_IDX[:, 1]]
    pk = p[_T_IDX[:, 2]]
    
    dets = (pi[:, 0] * (pj[:, 1] * pk[:, 2] - pj[:, 2] * pk[:, 1])
           - pi[:, 1] * (pj[:, 0] * pk[:, 2] - pj[:, 2] * pk[:, 0])
           + pi[:, 2] * (pj[:, 0] * pk[:, 1] - pj[:, 1] * pk[:, 0]))
           
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y) for equilateral triangle."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """Project barycentric coordinates to be valid (non-negative, sum to 1)."""
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _optimize_point_lp(p, point_idx, t_lower=0.0):
    """
    Optimize a single point p[point_idx] using Linear Programming.
    We fix all other points. The area of any triangle involving point_idx is linear in p[point_idx].
    We want to maximize the minimum area of all triangles involving point_idx.
    Triangles not involving point_idx are constant.
    
    We handle the absolute value by assuming a consistent orientation.
    We will solve for the case where det > 0 and det < 0 separately? 
    Actually, we can just maximize the minimum of the signed areas if we enforce a sign, 
    or we can maximize the minimum of the absolute values by splitting the problem.
    
    Simpler approach for LP:
    Let A_ijk be the area of triangle (i,j,k).
    For triples NOT containing point_idx, A_ijk is constant. Let C = min(A_ijk for k not in triple).
    For triples CONTAINING point_idx, A_ijk = 0.5 * | linear_func(p[point_idx]) |.
    We want to maximize t such that:
    t <= C
    t <= 0.5 * | L_k(p[point_idx]) | for all k in triples containing point_idx.
    
    The constraint t <= 0.5 * | L_k(x) | is equivalent to:
    (t <= 0.5 * L_k(x)) OR (t <= -0.5 * L_k(x))
    This is not linear.
    
    However, if we assume the orientation of all triples containing point_idx is the same as the current orientation,
    we can drop the absolute value and just use the signed area.
    If the optimal solution flips the sign, we can retry with the flipped sign.
    
    Let's assume we want to maximize the minimum of the signed areas, assuming all relevant determinants have the same sign s (either +1 or -1).
    Constraint: s * 0.5 * L_k(x) >= t.
    This is linear.
    
    We will try both signs and pick the best valid result.
    """
    # Identify triples containing point_idx
    # A triple (i, j, k) contains point_idx if point_idx is one of i, j, k.
    # For each such triple, the determinant is linear in p[point_idx].
    # det = c0 * x0 + c1 * x1 + c2 * x2 + c_const? 
    # Actually, det(p_i, p_j, p_k) is linear in p_k if p_i, p_j are fixed.
    # det = p_i0*(p_j1*p_k2 - p_j2*p_k1) - p_i1*(p_j0*p_k2 - p_j2*p_k0) + p_i2*(p_j0*p_k1 - p_j1*p_k0)
    # = p_k2*(p_i0*p_j1 - p_i1*p_j0) - p_k1*(p_i0*p_j2 - p_i2*p_j0) + p_k0*(p_i1*p_j2 - p_i2*p_j1)
    # So it is of the form A*x0 + B*x1 + C*x2. No constant term because if p_k=0, det=0.
    
    # We want to maximize t.
    # Variables: x0, x1, x2, t.
    # Objective: minimize -t.
    # Constraints:
    # 1. x0 + x1 + x2 = 1
    # 2. x0, x1, x2 >= 0
    # 3. For each triple (i,j,k) containing point_idx:
    #    If point_idx is k: s * 0.5 * (A*x0 + B*x1 + C*x2) >= t
    #    If point_idx is j: s * 0.5 * (A*x0 + B*x1 + C*x2) >= t (with appropriate coeffs)
    #    If point_idx is i: s * 0.5 * (A*x0 + B*x1 + C*x2) >= t (with appropriate coeffs)
    # 4. t >= t_lower (optional, to ensure we don't go backwards)
    # 5. t <= min_area_others (the minimum area of triples NOT containing point_idx)
    
    # Calculate min area of triples NOT containing point_idx
    min_area_others = 1.0
    for i, j, k in _TRIPLES:
        if point_idx not in [i, j, k]:
            # Calculate area
            pi = p[i]
            pj = p[j]
            pk = p[k]
            det = (pi[0] * (pj[1] * pk[2] - pj[2] * pk[1])
                 - pi[1] * (pj[0] * pk[2] - pj[2] * pk[0])
                 + pi[2] * (pj[0] * pk[1] - pj[1] * pk[0]))
            area = 0.5 * abs(det)
            if area < min_area_others:
                min_area_others = area
                
    # If we can't beat the current best, we might as well return current? 
    # But we are optimizing to maximize the min, so t will be limited by min_area_others.
    
    # Build LP constraints for triples containing point_idx
    # We need to determine the coefficients for each triple.
    # Let's collect the linear coefficients (A, B, C) for each such triple.
    # And the sign s. We will try s=1 and s=-1.
    
    A_coeffs = []
    B_coeffs = []
    C_coeffs = []
    
    for i, j, k in _TRIPLES:
        if point_idx == i:
            # det = p_i0*(p_j1*p_k2 - p_j2*p_k1) - p_i1*(p_j0*p_k2 - p_j2*p_k0) + p_i2*(p_j0*p_k1 - p_j1*p_k0)
            # x is p_i
            A = p[j][1]*p[k][2] - p[j][2]*p[k][1]
            B = -(p[j][0]*p[k][2] - p[j][2]*p[k][0])
            C = p[j][0]*p[k][1] - p[j][1]*p[k][0]
        elif point_idx == j:
            # det = p_i0*(p_j1*p_k2 - p_j2*p_k1) - p_i1*(p_j0*p_k2 - p_j2*p_k0) + p_i2*(p_j0*p_k1 - p_j1*p_k0)
            # x is p_j
            A = -(p[i][0]*p[k][2] - p[i][2]*p[k][0]) # coeff of p_j1? No.
            # Let's re-derive carefully.
            # det = | p_i0 p_i1 p_i2 |
            #       | x0   x1   x2 |
            #       | p_k0 p_k1 p_k2 |
            # = x0*(p_i1*p_k2 - p_i2*p_k1) - x1*(p_i0*p_k2 - p_i2*p_k0) + x2*(p_i0*p_k1 - p_i1*p_k0)
            A = p[i][1]*p[k][2] - p[i][2]*p[k][1]
            B = -(p[i][0]*p[k][2] - p[i][2]*p[k][0])
            C = p[i][0]*p[k][1] - p[i][1]*p[k][0]
        elif point_idx == k:
            # det = | p_i0 p_i1 p_i2 |
            #       | p_j0 p_j1 p_j2 |
            #       | x0   x1   x2 |
            # = x0*(p_i1*p_j2 - p_i2*p_j1) - x1*(p_i0*p_j2 - p_i2*p_j0) + x2*(p_i0*p_j1 - p_i1*p_j0)
            A = p[i][1]*p[j][2] - p[i][2]*p[j][1]
            B = -(p[i][0]*p[j][2] - p[i][2]*p[j][0])
            C = p[i][0]*p[j][1] - p[i][1]*p[j][0]
        else:
            continue
            
        A_coeffs.append(A)
        B_coeffs.append(B)
        C_coeffs.append(C)
        
    if not A_coeffs:
        return p[point_idx].copy(), 0.0
        
    A_coeffs = np.array(A_coeffs)
    B_coeffs = np.array(B_coeffs)
    C_coeffs = np.array(C_coeffs)
    
    best_x = p[point_idx].copy()
    best_t = 0.0
    
    for sign in [1.0, -1.0]:
        # Variables: x0, x1, x2, t
        # Minimize -t
        c = np.array([0, 0, 0, -1])
        
        # Equality constraints: x0 + x1 + x2 = 1
        A_eq = np.array([[1, 1, 1, 0]])
        b_eq = np.array([1])
        
        # Inequality constraints:
        # 1. t <= min_area_others  =>  -t <= -min_area_others
        # 2. sign * 0.5 * (A*x0 + B*x1 + C*x2) >= t  =>  -sign*0.5*A*x0 - sign*0.5*B*x1 - sign*0.5*C*x2 - t <= 0
        # 3. x0, x1, x2 >= 0 (handled by bounds)
        # 4. t >= 0 (handled by bounds)
        
        A_ub = []
        b_ub = []
        
        # t <= min_area_others
        A_ub.append([0, 0, 0, -1])
        b_ub.append(-min_area_others)
        
        # Area constraints
        for idx in range(len(A_coeffs)):
            row = np.array([-sign * 0.5 * A_coeffs[idx], 
                            -sign * 0.5 * B_coeffs[idx], 
                            -sign * 0.5 * C_coeffs[idx], 
                            -1])
            A_ub.append(row)
            b_ub.append(0)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, 1), (0, 1), (0, 1), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                t_val = res.x[3]
                x_val = res.x[:3]
                # Verify the solution is valid (sign assumption holds)
                # Check if the actual signed areas match the sign
                # And if the min area is indeed t_val
                new_p = p.copy()
                new_p[point_idx] = x_val
                # Calculate actual min area
                actual_min = _min_area_bary(new_p)
                # If the sign assumption was wrong, the LP might have found a solution 
                # that violates the sign constraint in reality (i.e. det has opposite sign).
                # But the LP enforced sign * 0.5 * L >= t. 
                # If sign=1, it enforced L >= 2t. So det >= 2t. Since t>0, det>0.
                # If sign=-1, it enforced -L >= 2t => L <= -2t. So det < 0.
                # So the sign is enforced by the constraint.
                # However, we must ensure that the OTHER triples (not containing point_idx) 
                # are not smaller than t. We added t <= min_area_others, so this is handled.
                
                if t_val > best_t:
                    best_t = t_val
                    best_x = x_val
        except Exception:
            pass
            
    return best_x, best_t

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(12345)
    
    best_area = 0.0
    best_bary = None
    
    def _is_time_up():
        return time.time() > deadline - 10.0
        
    # Initializations
    inits = []
    
    # Random
    inits.append(rng.dirichlet([1, 1, 1], size=N))
    
    # Structured: Vertices, Midpoints, Center
    bary_struct = np.zeros((N, 3))
    bary_struct[0] = [0.9, 0.05, 0.05]
    bary_struct[1] = [0.05, 0.9, 0.05]
    bary_struct[2] = [0.05, 0.05, 0.9]
    bary_struct[3] = [0.45, 0.45, 0.1]
    bary_struct[4] = [0.1, 0.45, 0.45]
    bary_struct[5] = [0.45, 0.1, 0.45]
    bary_struct[6] = [1/3, 1/3, 1/3]
    for i in range(7, N):
        bary_struct[i] = rng.dirichlet([1, 1, 1])
    inits.append(bary_struct)
    
    # Another structured: Grid on barycentric axes
    bary_grid = np.zeros((N, 3))
    # 1 center
    bary_grid[0] = [1/3, 1/3, 1/3]
    # 3 near vertices
    bary_grid[1] = [0.8, 0.1, 0.1]
    bary_grid[2] = [0.1, 0.8, 0.1]
    bary_grid[3] = [0.1, 0.1, 0.8]
    # 3 near midpoints
    bary_grid[4] = [0.4, 0.4, 0.2]
    bary_grid[5] = [0.4, 0.2, 0.4]
    bary_grid[6] = [0.2, 0.4, 0.4]
    # 3 more
    bary_grid[7] = [0.6, 0.2, 0.2]
    bary_grid[8] = [0.2, 0.6, 0.2]
    bary_grid[9] = [0.2, 0.2, 0.6]
    bary_grid[10] = [0.3, 0.3, 0.4]
    inits.append(bary_grid)
    
    for restart in range(20):
        if _is_time_up():
            break
            
        if restart < len(inits):
            current_bary = _project_bary(inits[restart].copy())
        else:
            current_bary = rng.dirichlet([1, 1, 1], size=N)
            current_bary = _project_bary(current_bary)
            
        # Coordinate Descent with LP
        # Iterate through points multiple times
        for sweep in range(10):
            if _is_time_up():
                break
            for i in range(N):
                if _is_time_up():
                    break
                new_pt, _ = _optimize_point_lp(current_bary, i)
                current_bary[i] = new_pt
                current_bary = _project_bary(current_bary)
                
        current_area = _min_area_bary(current_bary)
        
        if current_area > best_area:
            best_area = current_area
            best_bary = current_bary.copy()
            
        # Occasionally perturb best
        if restart % 3 == 2 and best_bary is not None:
            perturbed = best_bary + rng.normal(0, 0.05, size=(N, 3))
            perturbed = _project_bary(perturbed)
            
            for sweep in range(5):
                if _is_time_up():
                    break
                for i in range(N):
                    if _is_time_up():
                        break
                    new_pt, _ = _optimize_point_lp(perturbed, i)
                    perturbed[i] = new_pt
                    perturbed = _project_bary(perturbed)
                    
            area_pert = _min_area_bary(perturbed)
            if area_pert > best_area:
                best_area = area_pert
                best_bary = perturbed.copy()

    if best_bary is None:
        bary = np.zeros((N, 3))
        for i in range(N):
            bary[i] = [1/3, 1/3, 1/3] + rng.normal(0, 0.1, size=3)
        best_bary = _project_bary(bary)
    
    cart = _bary_to_cart(best_bary)
    return cart
