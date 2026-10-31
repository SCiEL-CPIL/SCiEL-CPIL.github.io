"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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
    c = 2 * pts[:, 1] / np.sqrt(3)
    b = pts[:, 0] - 0.5 * c
    a = 1 - b - c
    
    pen_a = np.maximum(0, -a)
    pen_b = np.maximum(0, -b)
    pen_c = np.maximum(0, -c)
    penalty = np.sum(pen_a**2 + pen_b**2 + pen_c**2)
    
    pts_clamped = _clamp_all(pts)
    
    area_val = _soft_min_area(pts_clamped, beta=20.0)
    
    return -area_val + 100.0 * penalty

def _nlp_objective(x):
    """
    NLP objective: minimize -alpha where x = [alpha, x1, y1, ..., x11, y11]
    """
    return -x[0]

def _nlp_constraints(x):
    """
    Constraints: area_i - alpha >= 0 for all i
    Returns array of shape (_N_TRI,)
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    areas = _all_areas(pts)
    return areas - alpha

def _nlp_jacobian(x):
    """
    Jacobian of constraints w.r.t. x
    Shape: (_N_TRI, 1 + 2*N)
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    n_vars = 1 + 2 * N
    
    jac = np.zeros((_N_TRI, n_vars))
    
    # d(area_i - alpha)/d(alpha) = -1
    jac[:, 0] = -1.0
    
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    
    # Signed cross product
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    sign = np.sign(cross)
    sign[sign == 0] = 1.0
    
    # area = 0.5 * sign * cross
    # d(area)/d(p_i) = -0.5 * sign * (p_k - p_j) x z? No, 2D cross product gradient.
    # Let's derive:
    # A = 0.5 * | (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) |
    # Let S = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
    # If S > 0, A = 0.5 S.
    # dA/dx_i = 0.5 * ( -(y_k-y_i) + (y_j-y_i) ) = 0.5 * (y_j - y_k)
    # dA/dy_i = 0.5 * ( -(x_k-x_i) + (x_j-x_i) ) = 0.5 * (x_j - x_k)
    # dA/dx_j = 0.5 * (y_k - y_i)
    # dA/dy_j = 0.5 * (x_i - x_k)
    # dA/dx_k = 0.5 * (y_i - y_j)
    # dA/dy_k = 0.5 * (x_j - x_i)
    
    # These formulas are for S > 0. If S < 0, A = -0.5 S, so gradients are negated.
    # So, multiply by sign.
    
    # Indices in x: 1+2*i, 2+2*i
    idx_i = 1 + 2 * _TRI_IDX[:, 0]
    idx_j = 1 + 2 * _TRI_IDX[:, 1]
    idx_k = 1 + 2 * _TRI_IDX[:, 2]
    
    # Gradient w.r.t. point i
    jac[:, idx_i] = 0.5 * sign * (pj[:, 1] - pk[:, 1])
    jac[:, idx_i + 1] = 0.5 * sign * (pk[:, 0] - pj[:, 0])
    
    # Gradient w.r.t. point j
    jac[:, idx_j] = 0.5 * sign * (pi[:, 1] - pk[:, 1])
    jac[:, idx_j + 1] = 0.5 * sign * (pk[:, 0] - pi[:, 0])
    
    # Gradient w.r.t. point k
    jac[:, idx_k] = 0.5 * sign * (pj[:, 1] - pi[:, 1])
    jac[:, idx_k + 1] = 0.5 * sign * (pi[:, 0] - pj[:, 0])
    
    return jac

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

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bounds for differential evolution
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N
    
    best_pts = _initial_placement_symmetric()
    
    # Phase 1: Differential Evolution
    try:
        if time_left() > 40.0:
            res_de = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=150,
                tol=1e-8,
                popsize=15,
                polish=False,
                updating='immediate'
            )
            if res_de.success or res_de.fun < 1e6:
                candidate_pts = _clamp_all(res_de.x.reshape(-1, 2))
                if _min_area(candidate_pts) > _min_area(best_pts):
                    best_pts = candidate_pts
        else:
            best_pts = _initial_placement_symmetric()
    except Exception:
        best_pts = _initial_placement_symmetric()
    
    # Phase 2: NLP Polishing with Trust-Constr
    if time_left() > 20.0:
        try:
            # Construct initial x for NLP
            alpha_init = _min_area(best_pts) * 0.9
            x0 = np.concatenate(([alpha_init], best_pts.flatten()))
            
            # Bounds for NLP
            nlp_bounds = [(0.0, 0.5)]  # alpha
            for i in range(N):
                nlp_bounds.append((0.0, 1.0))
                nlp_bounds.append((0.0, H))
            
            # Use LinearConstraint for the triangle containment? 
            # No, trust-constr handles bounds well.
            
            res_nlp = minimize(
                _nlp_objective,
                x0,
                method='trust-constr',
                bounds=nlp_bounds,
                constraints=[
                    {
                        'type': 'ineq',
                        'fun': _nlp_constraints,
                        'jac': _nlp_jacobian
                    }
                ],
                options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            
            if res_nlp.success or (hasattr(res_nlp, 'x') and not np.any(np.isnan(res_nlp.x))):
                nlp_pts = _clamp_all(res_nlp.x[1:].reshape(-1, 2))
                if _min_area(nlp_pts) > _min_area(best_pts):
                    best_pts = nlp_pts
        except Exception:
            pass
    
    # Phase 3: Fallback Polish with L-BFGS-B if NLP didn't improve much or failed
    if time_left() > 5.0:
        try:
            flat_pts = best_pts.flatten()
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=100.0),
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

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
