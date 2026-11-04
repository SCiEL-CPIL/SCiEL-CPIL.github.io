# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta, penalty_lambda=50.0):
    """
    Smooth objective for L-BFGS-B using log-sum-exp soft-min.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # Soft min: m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    # = m - (1/beta) * log(sum(exp(-beta * shifted)))
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _local_optimize_nm(x0, bounds, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, bounds, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - selected, axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _topological_sort_points(bary):
    """
    Sort points so that for every triple (i,j,k) with i<j<k,
    the determinant det(P_i, P_j, P_k) has a consistent sign.
    Returns the permutation and the sign pattern (+1 or -1 for each triple).
    
    Strategy: sort by x-coordinate (v + w*0.5 = v + (1-u-v)*0.5 = 0.5 + v/2 - u/2),
    then try to find a consistent orientation.
    """
    # Compute a "linear" coordinate for sorting: use the x-coordinate in Cartesian
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x_cart = v * 1.0 + w * 0.5  # = v + 0.5 - 0.5*u - 0.5*v = 0.5 + 0.5*v - 0.5*u
    
    # Sort by x_cart, break ties by y
    y_cart = w * np.sqrt(3) / 2
    order = np.lexsort((y_cart, x_cart))
    
    # Try this ordering: check if all dets have the same sign
    perm_bary = bary[order]
    dets = _all_dets_from_bary(perm_bary)
    signs = np.sign(dets)
    signs[signs == 0] = 1  # avoid zero
    
    # Count consistent signs
    pos_count = np.sum(signs > 0)
    neg_count = np.sum(signs < 0)
    
    # If mostly one sign, use that; otherwise try reverse
    if neg_count > pos_count:
        # Reverse the order
        order = order[::-1]
        perm_bary = bary[order]
        dets = _all_dets_from_bary(perm_bary)
        signs = np.sign(dets)
        signs[signs == 0] = 1
        pos_count = np.sum(signs > 0)
        neg_count = np.sum(signs < 0)
    
    # Determine the target sign (majority)
    if pos_count >= neg_count:
        target_sign = 1
    else:
        target_sign = -1
        order = order[::-1]
        perm_bary = bary[order]
        dets = _all_dets_from_bary(perm_bary)
        signs = np.sign(dets)
        signs[signs == 0] = target_sign
    
    return order, target_sign

def _solve_lp_for_fixed_orientation(order, target_sign, t_init=None):
    """
    Solve the LP: maximize t subject to:
      - For each triple (i,j,k) in the sorted order: target_sign * det_i_j_k >= t
      - 0 <= u_i <= 1, 0 <= v_i <= 1, u_i + v_i <= 1 for all i
    
    Variables: [u_0, v_0, u_1, v_1, ..., u_{N-1}, v_{N-1}, t]
    Total: 2N + 1 = 23 variables
    
    The determinant for points P_i, P_j, P_k in barycentric (u,v,w) is:
    det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    
    Since w = 1-u-v, this is linear in u,v.
    
    We use scipy.optimize.linprog with method='highs'.
    """
    # Variables: x[0:2N] are (u_0, v_0, u_1, v_1, ...), x[2N] is t
    n_vars = 2 * N + 1
    
    # Objective: minimize -t => c has -1 at position 2N
    c = np.zeros(n_vars)
    c[2 * N] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each triple (i,j,k) with i<j<k in the permuted order:
    #   target_sign * det >= t
    #   => -target_sign * det + t <= 0
    #   => -target_sign * (linear_expr_in_u,v) + t <= 0
    
    # Compute the linear coefficients for each determinant
    # det(P_i, P_j, P_k) where P_m = (u_m, v_m, 1-u_m-v_m)
    # = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k) 
    #   - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
    #   + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
    
    # This is linear in u_i, v_i, u_j, v_j, u_k, v_k
    # Let's expand:
    # det = u_i*v_j - u_i*v_j*u_k - u_i*v_j*v_k - u_i*v_k + u_i*v_k*u_j + u_i*v_k*v_j
    #       - v_i*u_j + v_i*u_j*u_k + v_i*u_j*v_k + v_i*u_k - v_i*u_k*u_j - v_i*u_k*v_j
    #       + u_j*v_k - v_j*u_k - u_i*u_j*v_k + u_i*v_j*u_k - v_i*u_j*v_k + v_i*v_j*u_k
    
    # Wait, let me be more careful. Let me compute it symbolically.
    # P_i = (u_i, v_i, 1-u_i-v_i)
    # P_j = (u_j, v_j, 1-u_j-v_j)  
    # P_k = (u_k, v_k, 1-u_k-v_k)
    
    # det = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k)
    #       - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
    #       + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
    
    # Let me expand term by term:
    # Term1 = u_i * v_j * (1-u_k-v_k) - u_i * (1-u_j-v_j) * v_k
    #        = u_i*v_j - u_i*v_j*u_k - u_i*v_j*v_k - u_i*v_k + u_i*v_k*u_j + u_i*v_k*v_j
    #        = u_i*v_j - u_i*v_j*u_k - u_i*v_j*v_k - u_i*v_k + u_i*u_j*v_k + u_i*v_j*v_k
    #        = u_i*v_j - u_i*v_j*u_k - u_i*v_k + u_i*u_j*v_k
    
    # Hmm, this is getting complex. Let me just compute the gradient numerically
    # by evaluating the determinant at basis points.
    
    # Actually, let me just build the constraint matrix directly.
    # For a fixed triple (i,j,k), the determinant as a function of (u_i,v_i,u_j,v_j,u_k,v_k)
    # is linear. I can compute the coefficients by evaluating at 6 basis vectors.
    
    n_constraints = N_TRIPLES
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    for idx, (i, j, k) in enumerate(_IDX_TRIPLES):
        # For this triple, the constraint is: -target_sign * det + t <= 0
        # We need the coefficients of det in terms of u_i, v_i, u_j, v_j, u_k, v_k
        
        # Compute det at 6 unit vectors to get linear coefficients
        # det is linear in each coordinate of each point
        
        # Coefficient of u_i: det when only u_i varies (set others to 0)
        # Actually, let me just compute the partial derivatives.
        
        # det = |u_i  v_i  1-u_i-v_i|
        #       |u_j  v_j  1-u_j-v_j|
        #       |u_k  v_k  1-u_k-v_k|
        
        # = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k)
        #   - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
        #   + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        
        # Let me expand this fully:
        # Let a = u_i, b = v_i, c = u_j, d = v_j, e = u_k, f = v_k
        
        # det = a*(d*(1-e-f) - (1-c-d)*f) - b*(c*(1-e-f) - (1-c-d)*e) + (1-a-b)*(c*f - d*e)
        
        # = a*d - a*d*e - a*d*f - a*f + a*f*c + a*f*d - b*c + b*c*e + b*c*f + b*e - b*e*c - b*e*f + c*f - d*e - a*c*f + a*d*e - b*c*f + b*d*e
        
        # Collect terms:
        # Constant: 0
        # a: 0
        # b: 0
        # c: 0
        # d: 0
        # e: 0
        # f: 0
        # a*d: +1
        # a*d*e: -1
        # a*d*f: -1
        # a*f: -1
        # a*f*c: +1  => a*c*f: +1
        # a*f*d: +1  => a*d*f: +1 (cancels with -a*d*f above)
        # b*c: -1
        # b*c*e: +1
        # b*c*f: +1
        # b*e: +1
        # b*e*c: -1  => b*c*e: -1 (cancels with +b*c*e)
        # b*e*f: -1
        # c*f: +1
        # d*e: -1
        # a*c*f: -1 (from -a*c*f)
        # a*d*e: +1 (from +a*d*e, cancels with -a*d*e)
        # b*c*f: -1 (from -b*c*f, cancels with +b*c*f)
        # b*d*e: +1
        
        # So after cancellation:
        # a*d: +1
        # a*f: -1
        # a*c*f: +1 - 1 = 0
        # b*c: -1
        # b*e: +1
        # b*e*f: -1
        # b*d*e: +1
        # c*f: +1
        # d*e: -1
        
        # Wait, I think I made errors. Let me redo this more carefully.
        
        # det = a*(d - d*e - d*f - f + c*f + d*f) - b*(c - c*e - c*f - e + c*e + d*e) + (1-a-b)*(c*f - d*e)
        
        # Inner term 1: d*(1-e-f) - (1-c-d)*f = d - d*e - d*f - f + c*f + d*f = d - d*e - f + c*f
        # So first part: a*(d - d*e - f + c*f) = a*d - a*d*e - a*f + a*c*f
        
        # Inner term 2: c*(1-e-f) - (1-c-d)*e = c - c*e - c*f - e + c*e + d*e = c - c*f - e + d*e
        # So second part: -b*(c - c*f - e + d*e) = -b*c + b*c*f + b*e - b*d*e
        
        # Inner term 3: (1-a-b)*(c*f - d*e) = c*f - d*e - a*c*f + a*d*e - b*c*f + b*d*e
        
        # Sum:
        # a*d - a*d*e - a*f + a*c*f - b*c + b*c*f + b*e - b*d*e + c*f - d*e - a*c*f + a*d*e - b*c*f + b*d*e
        
        # Cancel:
        # a*d*e: -1 + 1 = 0
        # a*c*f: +1 - 1 = 0
        # b*c*f: +1 - 1 = 0
        # b*d*e: -1 + 1 = 0
        
        # Remaining:
        # a*d: +1
        # a*f: -1
        # b*c: -1
        # b*e: +1
        # c*f: +1
        # d*e: -1
        
        # So det = a*d - a*f - b*c + b*e + c*f - d*e
        #        = u_i*v_j - u_i*v_k - v_i*u_j + v_i*u_k + u_j*v_k - v_j*u_k
        
        # Let me verify: this should equal the 2D cross product of (P_j - P_i) x (P_k - P_i)
        # in barycentric space... actually it's the determinant of the 3x3 matrix with rows (u,v,w).
        
        # Coefficients:
        # u_i (a): coefficient of a is (d - f) = v_j - v_k
        # v_i (b): coefficient of b is (-c + e) = -u_j + u_k
        # u_j (c): coefficient of c is (-b + f) = -v_i + v_k
        # v_j (d): coefficient of d is (a - e) = u_i - u_k
        # u_k (e): coefficient of e is (b - d) = v_i - v_j
        # v_k (f): coefficient of f is (-a + c) = -u_i + u_j
        
        # So the linear constraint for triple (i,j,k) is:
        # -target_sign * [(v_j - v_k)*u_i + (u_k - u_j)*v_i + (v_k - v_i)*u_j + (u_i - u_k)*v_j + (v_i - v_j)*u_k + (u_j - u_i)*v_k] + t <= 0
        
        # This is linear in u_i, v_i, u_j, v_j, u_k, v_k, and t.
        
        # Variable indices: u_i is at 2*i, v_i is at 2*i+1
        # t is at 2*N
        
        A_ub[idx, 2*i] = -target_sign * (bary[j, 1] - bary[k, 1])  # coeff of u_i
        A_ub[idx, 2*i+1] = -target_sign * (bary[k, 0] - bary[j, 0])  # coeff of v_i
        A_ub[idx, 2*j] = -target_sign * (bary[k, 1] - bary[i, 1])  # coeff of u_j
        A_ub[idx, 2*j+1] = -target_sign * (bary[i, 0] - bary[k, 0])  # coeff of v_j
        A_ub[idx, 2*k] = -target_sign * (bary[i, 1] - bary[j, 1])  # coeff of u_k
        A_ub[idx, 2*k+1] = -target_sign * (bary[j, 0] - bary[i, 0])  # coeff of v_k
        A_ub[idx, 2*N] = 1.0  # coeff of t
        
        b_ub[idx] = 0.0
    
    # Simplex constraints:
    # u_i >= 0 => -u_i <= 0
    # v_i >= 0 => -v_i <= 0
    # u_i + v_i <= 1
    
    # Add these to A_ub, b_ub
    n_simplex = 3 * N
    A_simplex = np.zeros((n_simplex, n_vars))
    b_simplex = np.zeros(n_simplex)
    
    for i in range(N):
        A_simplex[3*i, 2*i] = -1.0       # -u_i <= 0
        A_simplex[3*i+1, 2*i+1] = -1.0   # -v_i <= 0
        A_simplex[3*i+2, 2*i] = 1.0      # u_i + v_i <= 1
        A_simplex[3*i+2, 2*i+1] = 1.0
        b_simplex[3*i+2] = 1.0
    
    A_full = np.vstack([A_ub, A_simplex])
    b_full = np.concatenate([b_ub, b_simplex])
    
    # Bounds: 0 <= u_i, v_i <= 1, t unbounded below (but we want to maximize)
    bounds = [(0, 1)] * (2 * N) + [(None, None)]
    
    try:
        res = linprog(c, A_ub=A_full, b_ub=b_full, bounds=bounds, method='highs')
        if res.success:
            t_opt = res.x[2*N]
            u_vals = res.x[0:2*N:2]
            v_vals = res.x[1:2*N:2]
            
            # Map back to original order
            bary_orig = np.zeros((N, 2))
            for new_idx, orig_idx in enumerate(order):
                bary_orig[orig_idx, 0] = u_vals[new_idx]
                bary_orig[orig_idx, 1] = v_vals[new_idx]
            
            bary_orig = _project_to_bary_simplex(bary_orig)
            score = _min_area_ratio_from_bary(bary_orig)
            return bary_orig, score, t_opt
        else:
            return None, -1.0, -1.0
    except Exception:
        return None, -1.0, -1.0

def _final_polish(x0, bounds, time_budget=30.0):
    """
    Final high-precision polish: SLSQP with very tight tolerances,
    then a second SLSQP pass, then Nelder-Mead with adaptive.
    """
    start = time.time()
    current = x0.copy()
    best_score = _min_area_ratio_from_bary(current.reshape(N, 2))
    
    # Pass 1: SLSQP with tight tolerance
    if time.time() - start < time_budget * 0.4:
        try:
            res = minimize(
                _objective_slsqp, current, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    # Pass 2: SLSQP again from improved point
    if time.time() - start < time_budget * 0.7:
        try:
            res = minimize(
                _objective_slsqp, current, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    # Pass 3: Nelder-Mead with very tight tolerance
    if time.time() - start < time_budget * 0.9:
        try:
            res = minimize(
                _objective_slsqp, current, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    # Pass 4: Final SLSQP
    if time.time() - start < time_budget:
        try:
            res = minimize(
                _objective_slsqp, current, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(15):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts from Inspiration
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    inits.append(_project_to_bary_simplex(np.array(bary4[:N])).flatten())
    
    bary5 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.3, 0.0],
        [0.7, 0.0],
        [0.0, 0.3],
        [0.0, 0.7],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary5).flatten())
    
    bounds = [(0, 1)] * (2 * N)
    
    # Phase 0: LP-based optimization with fixed orientation
    # For each initial configuration, try to solve the LP with the topological sort
    for init in inits:
        if time_left() < 10:
            break
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            order, target_sign = _topological_sort_points(init_bary)
            lp_result, lp_score, t_val = _solve_lp_for_fixed_orientation(order, target_sign)
            
            if lp_result is not None and lp_score > best_score:
                best_score = lp_score
                best_x = lp_result
        except Exception:
            pass
    
    # Phase 1: Local optimization from structured inits
    for init in inits:
        if time_left() < 5:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            best_local_x = init_bary
            best_local_score = _min_area_ratio_from_bary(init_bary)
            
            # Strategy: Multi-method refinement
            # 1. SLSQP
            if time_left() > 10:
                x_slsqp = _local_optimize_slsqp(init_bary.flatten(), bounds, maxiter=200)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                if score_slsqp > best_local_score:
                    best_local_score = score_slsqp
                    best_local_x = bary_slsqp
                    init_bary = bary_slsqp

            # 2. Nelder-Mead
            if time_left() > 10:
                x_nm = _local_optimize_nm(init_bary.flatten(), bounds, maxiter=300)
                bary_nm = _project_to_bary_simplex(x_nm.reshape(N, 2))
                score_nm = _min_area_ratio_from_bary(bary_nm)
                if score_nm > best_local_score:
                    best_local_score = score_nm
                    best_local_x = bary_nm
                    init_bary = bary_nm

            # 3. L-BFGS-B with Smooth Objective (annealing)
            if time_left() > 10:
                current_x = init_bary.flatten()
                betas = [50.0, 100.0, 300.0, 800.0]
                for beta in betas:
                    if time_left() < 5:
                        break
                    res = minimize(
                        _objective_smooth,
                        current_x,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    x_opt = res.x.reshape(N, 2)
                    x_opt = _project_to_bary_simplex(x_opt)
                    score = _min_area_ratio_from_bary(x_opt)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_x = x_opt
                        current_x = x_opt.flatten()

            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
                
        except Exception:
            pass

    # Phase 2: Differential Evolution for global search
    if time_left() > 30:
        try:
            de_bounds = [(0, 1)] * (2 * N)
            
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = _project_to_bary_simplex(bary)
                score = _min_area_ratio_from_bary(bary)
                if score < 0:
                    return 1.0  # Penalty for invalid
                return -score
            
            # Run differential evolution with limited population and iterations
            res_de = differential_evolution(
                de_objective,
                de_bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,  # We'll do our own polishing
                update_interval=1,
                callback=None,
                init='sobol'
            )
            
            if res_de.success or res_de.fun < -best_score:
                de_bary = _project_to_bary_simplex(res_de.x.reshape(N, 2))
                de_score = _min_area_ratio_from_bary(de_bary)
                if de_score > best_score:
                    best_score = de_score
                    best_x = de_bary
                    
        except Exception:
            pass

    # Phase 3: Random restarts with SLSQP if time remains
    if time_left() > 20:
        for _ in range(10):
            if time_left() < 10:
                break
            try:
                pts = np.zeros((N, 2))
                for i in range(N):
                    u = rng.random()
                    v = rng.random()
                    while u + v > 1.0:
                        u = rng.random()
                        v = rng.random()
                    pts[i, 0] = u
                    pts[i, 1] = v
                init_bary = _project_to_bary_simplex(pts)
                
                x_slsqp = _local_optimize_slsqp(init_bary.flatten(), bounds, maxiter=100)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                
                if score_slsqp > best_score:
                    best_score = score_slsqp
                    best_x = bary_slsqp
            except Exception:
                pass

    if best_x is None:
        best_x = _structured_init_fps()
    
    # Final high-precision polish on the best solution
    if time_left() > 15:
        try:
            bounds_final = [(0, 1)] * (2 * N)
            polished = _final_polish(best_x.flatten(), bounds_final, time_budget=min(40.0, time_left() - 10))
            polished_bary = _project_to_bary_simplex(polished.reshape(N, 2))
            polished_score = _min_area_ratio_from_bary(polished_bary)
            if polished_score > best_score:
                best_x = polished_bary
                best_score = polished_score
        except Exception:
            pass
    
    # Final LP polish: try to improve the best solution via LP with its own topological sort
    if time_left() > 5:
        try:
            order, target_sign = _topological_sort_points(best_x)
            lp_result, lp_score, t_val = _solve_lp_for_fixed_orientation(order, target_sign)
            if lp_result is not None and lp_score > best_score:
                best_x = lp_result
                best_score = lp_score
        except Exception:
            pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
