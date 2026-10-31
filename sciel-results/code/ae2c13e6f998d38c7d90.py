"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute triangle indices
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    c = 2 * y / SQRT3
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _initial_placement_symmetric():
    """Symmetric initial placement."""
    pts = []
    # Center
    pts.append([1/3, H/3])
    # 3 points near vertices
    m = 0.15
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.2
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # 3 more points in a ring
    cx, cy = 1/3, H/3
    r = 0.25
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _nlp_objective(x):
    """
    NLP objective: minimize -alpha.
    x = [alpha, x1, y1, ..., x11, y11]
    """
    alpha = x[0]
    return -alpha

def _nlp_constraints(x):
    """
    Constraints for NLP.
    Returns a list of dictionaries for scipy.optimize.minimize.
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    
    # 1. Area constraints: area(i,j,k) - alpha >= 0
    areas = _all_areas(pts)
    area_cons = areas - alpha
    
    # 2. Triangle boundary constraints
    # For each point p = (x, y):
    # y >= 0
    # sqrt(3)*x + y <= sqrt(3)  =>  sqrt(3) - sqrt(3)*x - y >= 0
    # y - sqrt(3)*x >= 0
    x_pts = pts[:, 0]
    y_pts = pts[:, 1]
    
    bound_cons = np.concatenate([
        y_pts,
        SQRT3 - SQRT3 * x_pts - y_pts,
        y_pts - SQRT3 * x_pts
    ])
    
    # Combine into a single inequality vector (all must be >= 0)
    all_cons = np.concatenate([area_cons, bound_cons])
    
    return all_cons

def _nlp_jacobian(x):
    """
    Jacobian of the constraints.
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    
    # Gradient of area constraints w.r.t x
    # d(area)/d(alpha) = -1
    # d(area)/d(pts) ...
    
    # We compute the gradient of the cross product term.
    # Let C = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # Area = 0.5 * |C|
    # d(Area)/d(p_m) = 0.5 * sign(C) * dC/d(p_m)
    
    grad_areas = np.zeros((_N_TRI, 2 * N))
    areas = _all_areas(pts)
    # We need the raw cross products to determine sign
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    signs = np.sign(cross)
    # If cross is 0, gradient is undefined, but we can use 0 or 1. 
    # In practice, this rarely happens in optimization.
    signs[signs == 0] = 1.0
    
    # dC/dx_i: -dy2 + dy1  => -dy2 + dy1
    # dC/dy_i: -dx2 + dx1  => -dx2 + dx1
    # dC/dx_j: dy2
    # dC/dy_j: -dx2
    # dC/dx_k: -dy1
    # dC/dy_k: dx1
    
    for idx, (i, j, k) in enumerate(_TRI_IDX):
        s = 0.5 * signs[idx]
        
        grad_areas[idx, 2*i]   = s * (-dy2[idx] + dy1[idx])
        grad_areas[idx, 2*i+1] = s * (-dx2[idx] + dx1[idx])
        
        grad_areas[idx, 2*j]   = s * (dy2[idx])
        grad_areas[idx, 2*j+1] = s * (-dx2[idx])
        
        grad_areas[idx, 2*k]   = s * (-dy1[idx])
        grad_areas[idx, 2*k+1] = s * (dx1[idx])
        
        grad_areas[idx, 0] = -1.0 # d(alpha - area)/d(alpha) = -1 ? No, constraint is area - alpha >= 0.
        # Wait, constraint is area - alpha.
        # d(area - alpha)/d(alpha) = -1.
        # So grad_areas[:, 0] should be -1.
        
    # Gradient of boundary constraints
    # Constraints:
    # y_i >= 0 -> grad: e_{2i+1}
    # SQRT3 - SQRT3 x_i - y_i >= 0 -> grad: -SQRT3 e_{2i} - e_{2i+1}
    # y_i - SQRT3 x_i >= 0 -> grad: -SQRT3 e_{2i} + e_{2i+1}
    
    grad_bounds = np.zeros((3 * N, 2 * N))
    for i in range(N):
        # y_i >= 0
        grad_bounds[i, 2*i+1] = 1.0
        
        # SQRT3 - SQRT3 x_i - y_i >= 0
        grad_bounds[N + i, 2*i] = -SQRT3
        grad_bounds[N + i, 2*i+1] = -1.0
        
        # y_i - SQRT3 x_i >= 0
        grad_bounds[2*N + i, 2*i] = -SQRT3
        grad_bounds[2*N + i, 2*i+1] = 1.0
        
    # Combine gradients
    # Total constraints: _N_TRI + 3*N
    # Total vars: 1 + 2*N
    jac = np.zeros((_N_TRI + 3*N, 1 + 2*N))
    jac[:_N_TRI, :] = grad_areas
    jac[_N_TRI:, :] = grad_bounds
    
    return jac

def _nlp_hessian_lagrange(x, v):
    """
    Hessian of the Lagrangian.
    L = -alpha + sum(v_i * (area_i - alpha)) + sum(w_j * bound_j)
    Hessian w.r.t x.
    The area terms are bilinear, so their second derivatives are 0.
    The boundary terms are linear, so their second derivatives are 0.
    Thus, the Hessian is 0.
    """
    return np.zeros((1 + 2*N, 1 + 2*N))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit (actually 150s limit, but let's be safe)
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bounds for x = [alpha, x1, y1, ..., x11, y11]
    # alpha >= 0
    # x in [0, 1]
    # y in [0, H]
    bounds = [(0.0, 1.0)] # alpha
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, H))   # y
    
    # Initial guess
    init_pts = _initial_placement_symmetric()
    init_alpha = _min_area(init_pts) * 0.9
    x0 = np.concatenate([[init_alpha], init_pts.flatten()])
    
    # Ensure x0 is feasible or close
    x0 = _clamp_nlp_var(x0)
    
    best_pts = init_pts
    best_val = _min_area(best_pts)
    
    # Phase 1: Differential Evolution for a global-ish start
    # DE is slow, so we use it only if we have plenty of time
    if time_left() > 60.0:
        try:
            def _de_objective(flat_pts):
                pts = _clamp_all(flat_pts.reshape(-1, 2))
                return -_min_area(pts)
            
            de_bounds = [(0.0, 1.0), (0.0, H)] * N
            res_de = differential_evolution(
                _de_objective,
                de_bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                polish=False
            )
            de_pts = _clamp_all(res_de.x.reshape(-1, 2))
            de_val = _min_area(de_pts)
            if de_val > best_val:
                best_pts = de_pts
                best_val = de_val
                x0 = np.concatenate([[best_val * 0.9], best_pts.flatten()])
        except Exception:
            pass
    
    # Phase 2: SLSQP NLP Optimization
    if time_left() > 20.0:
        try:
            # We want to maximize alpha, so minimize -alpha
            # Constraints: c(x) >= 0
            cons = {
                'type': 'ineq',
                'fun': _nlp_constraints,
                'jac': _nlp_jacobian
            }
            
            res_slsqp = minimize(
                _nlp_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res_slsqp.success:
                slsqp_pts = _clamp_all(res_slsqp.x[1:].reshape(-1, 2))
                slsqp_val = _min_area(slsqp_pts)
                if slsqp_val > best_val:
                    best_pts = slsqp_pts
                    best_val = slsqp_val
        except Exception:
            pass
            
    # Phase 3: Trust-Constr (if available and time permits)
    if time_left() > 10.0:
        try:
            from scipy.optimize import LinearConstraint, NonlinearConstraint
            # Trust-constr can be more robust for some problems
            # But SLSQP is usually sufficient for this size.
            # Let's try a local polish with Powell on the hard min area instead, 
            # which is often more effective for the final step.
            
            res_powell = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Powell',
                bounds=[(0.0, 1.0), (0.0, H)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            powell_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            powell_val = _min_area(powell_pts)
            if powell_val > best_val:
                best_pts = powell_pts
                best_val = powell_val
        except Exception:
            pass
            
    # Phase 4: Coordinate Descent / Nelder-Mead final polish
    if time_left() > 5.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            nm_val = _min_area(nm_pts)
            if nm_val > best_val:
                best_pts = nm_pts
                best_val = nm_val
        except Exception:
            pass

    return _clamp_all(best_pts)

def _clamp_nlp_var(x):
    """Clamp x to be within bounds and triangle."""
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    pts = _clamp_all(pts)
    # Ensure alpha is not larger than min area (for feasibility start)
    min_a = _min_area(pts)
    alpha = min(alpha, min_a)
    alpha = max(0.0, alpha)
    return np.concatenate([[alpha], pts.flatten()])

# EVOLVE-BLOCK-END
