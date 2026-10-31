"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, LinearConstraint

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
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
    # Barycentric coordinates
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5c
    # y = c * H
    c = 2 * y / np.sqrt(3)
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

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    shifted = -beta * (areas - m)
    # Use log-sum-exp trick for stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _objective_de(flat_pts):
    """
    Objective for Differential Evolution.
    Minimizes negative soft-min area + penalty for points outside triangle.
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    # Compute barycentric coordinates
    # c = 2*y/sqrt(3), b = x - 0.5*c, a = 1-b-c
    c = 2 * pts[:, 1] / np.sqrt(3)
    b = pts[:, 0] - 0.5 * c
    a = 1 - b - c
    
    # Distance to boundary is min(a, b, c) if negative
    # We want to penalize if any of a,b,c < 0
    # Penalty = sum(max(0, -a)^2 + max(0, -b)^2 + max(0, -c)^2)
    pen_a = np.maximum(0, -a)
    pen_b = np.maximum(0, -b)
    pen_c = np.maximum(0, -c)
    penalty = np.sum(pen_a**2 + pen_b**2 + pen_c**2)
    
    # Clamp points for area calculation to avoid numerical issues with area
    pts_clamped = _clamp_all(pts)
    
    # Use a soft min area for smoothness
    area_val = _soft_min_area(pts_clamped, beta=20.0)
    
    # Large penalty weight to keep points inside
    return -area_val + 100.0 * penalty

def _nlp_objective(x):
    """
    NLP objective: minimize -alpha where x = [alpha, x1, y1, ..., x11, y11]
    Constraints: each triangle area >= alpha
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    return -alpha

def _nlp_constraints(x):
    """
    Constraints for NLP: each triangle area >= alpha
    Returns array of (area_i - alpha) for all i
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    areas = _all_areas(pts)
    return areas - alpha

def _nlp_bounds():
    """
    Bounds for NLP: alpha in [0, 0.5], points in triangle
    """
    bounds = [(0.0, 0.5)]  # alpha
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, H))    # y_i
    return bounds

def _nlp_jacobian_constraints(x):
    """
    Jacobian of constraints w.r.t. x
    Constraint i: area_i - alpha >= 0
    d(area_i)/d(alpha) = -1
    d(area_i)/d(point_k) depends on which points are in triangle i
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    n_vars = 1 + 2 * N
    
    # Jacobian shape: (_N_TRI, n_vars)
    jac = np.zeros((_N_TRI, n_vars))
    
    # d(area_i)/d(alpha) = -1 for all i
    jac[:, 0] = -1.0
    
    # For each triangle (i, j, k), compute gradients w.r.t. points i, j, k
    # area = 0.5 * |cross(pj - pi, pk - pi)|
    # Let A = pi, B = pj, C = pk
    # area = 0.5 * |(B-A) x (C-A)| = 0.5 * |(Bx - Ax)*(Cy - Ay) - (By - Ay)*(Cx - Ax)|
    # d(area)/d(Ax) = 0.5 * [-(Cy - Ay) + (By - Ay)] = 0.5 * (By - Cy)
    # d(area)/d(Ay) = 0.5 * [-(Cx - Ax) + (Bx - Ax)] = 0.5 * (Bx - Cx)
    # d(area)/d(Bx) = 0.5 * (Cy - Ay)
    # d(area)/d(By) = 0.5 * (Ax - Cx)
    # d(area)/d(Cx) = 0.5 * (Ay - By)
    # d(area)/d(Cy) = 0.5 * (Bx - Ax)
    
    # But we need to handle the absolute value. For simplicity, assume the sign
    # is consistent near the optimum. We'll use the signed area and take care
    # of the sign in the constraint.
    
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    
    # Signed cross product
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    sign = np.sign(cross)
    sign[sign == 0] = 1.0
    
    # Gradients for point i (first point in triangle)
    # d(area)/d(xi) = 0.5 * sign * (yj - yk)
    # d(area)/d(yi) = 0.5 * sign * (xk - xj)
    jac[:, 1 + 2*_TRI_IDX[:, 0]] = 0.5 * sign * (pj[:, 1] - pk[:, 1])
    jac[:, 2 + 2*_TRI_IDX[:, 0]] = 0.5 * sign * (pk[:, 0] - pj[:, 0])
    
    # Gradients for point j (second point in triangle)
    # d(area)/d(xj) = 0.5 * sign * (yi - yk)
    # d(area)/d(yj) = 0.5 * sign * (xk - xi)
    jac[:, 1 + 2*_TRI_IDX[:, 1]] = 0.5 * sign * (pi[:, 1] - pk[:, 1])
    jac[:, 2 + 2*_TRI_IDX[:, 1]] = 0.5 * sign * (pk[:, 0] - pi[:, 0])
    
    # Gradients for point k (third point in triangle)
    # d(area)/d(xk) = 0.5 * sign * (yj - yi)
    # d(area)/d(yk) = 0.5 * sign * (xi - xj)
    jac[:, 1 + 2*_TRI_IDX[:, 2]] = 0.5 * sign * (pj[:, 1] - pi[:, 1])
    jac[:, 2 + 2*_TRI_IDX[:, 2]] = 0.5 * sign * (pi[:, 0] - pj[:, 0])
    
    return jac

def _nlp_jacobian_objective(x):
    """Jacobian of objective: d(-alpha)/dx = [-1, 0, 0, ..., 0]"""
    n_vars = 1 + 2 * N
    jac = np.zeros(n_vars)
    jac[0] = -1.0
    return jac

def _nlp_jacobian_hessian_lagrange(x, v):
    """Hessian of Lagrangian (zero for linear constraints)"""
    n_vars = 1 + 2 * N
    return np.zeros((n_vars, n_vars))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # First, get a good initial point using differential evolution
    bounds_de = [(0.0, 1.0)] * N + [(0.0, H)] * N
    best_pts = None
    
    try:
        if time_left() > 30.0:
            res_de = differential_evolution(
                _objective_de,
                bounds_de,
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=12,
                polish=False,
                updating='immediate'
            )
            if res_de.success or res_de.fun < 1e6:
                best_pts = _clamp_all(res_de.x.reshape(-1, 2))
            else:
                best_pts = _initial_placement_symmetric()
        else:
            best_pts = _initial_placement_symmetric()
    except Exception:
        best_pts = _initial_placement_symmetric()
    
    # Now try NLP approach with trust-constr
    if time_left() > 20.0:
        try:
            # Initial guess: [alpha, x1, y1, ..., x11, y11]
            alpha_init = _min_area(best_pts) * 0.9
            x0 = np.zeros(1 + 2 * N)
            x0[0] = alpha_init
            x0[1:] = best_pts.flatten()
            
            # Bounds
            nlp_bounds = _nlp_bounds()
            
            # Constraint: all areas >= alpha, i.e., areas - alpha >= 0
            # We use NonlinearConstraint
            from scipy.optimize import NonlinearConstraint
            
            def _constraint_func(x):
                alpha = x[0]
                pts = x[1:].reshape(-1, 2)
                areas = _all_areas(pts)
                return areas - alpha
            
            def _constraint_jac(x):
                return _nlp_jacobian_constraints(x)
            
            nlp_con = NonlinearConstraint(
                _constraint_func,
                0.0,
                np.inf,
                jac=_constraint_jac
            )
            
            res_nlp = minimize(
                _nlp_objective,
                x0,
                method='trust-constr',
                bounds=nlp_bounds,
                constraints=[nlp_con],
                jac=_nlp_jacobian_objective,
                hess=_nlp_jacobian_hessian_lagrange,
                options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            
            if res_nlp.success or (hasattr(res_nlp, 'fun') and res_nlp.fun < 0):
                nlp_pts = _clamp_all(res_nlp.x[1:].reshape(-1, 2))
                nlp_alpha = res_nlp.x[0]
                if _min_area(nlp_pts) > _min_area(best_pts):
                    best_pts = nlp_pts
        except Exception:
            pass
    
    # Polish with L-BFGS-B on smooth objective
    if time_left() > 10.0:
        try:
            flat_pts = best_pts.flatten()
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=50.0),
                flat_pts,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N + [(0.0, H)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            polished_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            if _min_area(polished_pts) > _min_area(best_pts):
                best_pts = polished_pts
        except Exception:
            pass

    # Final polish with Nelder-Mead on hard objective
    if time_left() > 5.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            if _min_area(final_pts) > _min_area(best_pts):
                best_pts = final_pts
        except Exception:
            pass
            
    return _clamp_all(best_pts)

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

# EVOLVE-BLOCK-END
