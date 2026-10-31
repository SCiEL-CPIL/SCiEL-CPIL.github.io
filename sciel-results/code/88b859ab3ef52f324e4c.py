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

def _to_barycentric(pts):
    """Convert Cartesian points to barycentric coordinates (a, b, c) with a+b+c=1."""
    x, y = pts[:, 0], pts[:, 1]
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    return np.column_stack([a, b, c])

def _from_barycentric(abc):
    """Convert barycentric coordinates to Cartesian points."""
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    x = b + 0.5 * c
    y = c * H
    return np.column_stack([x, y])

def _clamp_barycentric(abc):
    """Clamp barycentric coordinates to be valid (all >= 0, sum = 1)."""
    eps = 1e-12
    abc = np.maximum(eps, abc)
    abc /= abc.sum(axis=1, keepdims=True)
    return abc

def _clamp_all(pts):
    """Clamp all points to triangle via barycentric coordinates."""
    abc = _to_barycentric(pts)
    abc = _clamp_barycentric(abc)
    return _from_barycentric(abc)

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    shifted = -beta * (areas - m)
    shifted -= np.max(shifted)  # numerical stability
    log_sum = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
    soft_min_val = m - log_sum / beta
    return soft_min_area_val if False else m - log_sum / beta

def _nlp_objective(x):
    """
    NLP objective: minimize -t where x = [t, a1, b1, c1, a2, b2, c2, ...]
    Barycentric coordinates are used to naturally enforce triangle constraints.
    """
    t = x[0]
    abc = x[1:].reshape(-1, 3)
    abc = _clamp_barycentric(abc)
    pts = _from_barycentric(abc)
    return -t

def _nlp_constraints(x):
    """
    Constraints: each triangle area >= t
    Returns array of (area_i - t) for all i, should be >= 0
    """
    t = x[0]
    abc = x[1:].reshape(-1, 3)
    abc = _clamp_barycentric(abc)
    pts = _from_barycentric(abc)
    areas = _all_areas(pts)
    return areas - t

def _nlp_jacobian_objective(x):
    """Jacobian of NLP objective: d(-t)/dx = [-1, 0, 0, ..., 0]"""
    jac = np.zeros_like(x)
    jac[0] = -1.0
    return jac

def _nlp_jacobian_constraints(x):
    """
    Jacobian of constraints: d(area_i - t)/dx
    For each constraint i: d(area_i)/d(pts) * d(pts)/d(abc) - dt
    """
    t = x[0]
    abc = x[1:].reshape(-1, 3)
    abc_clamped = _clamp_barycentric(abc)
    pts = _from_barycentric(abc_clamped)
    
    # Compute d(area_i)/d(pts) for each triangle
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    # d(area)/d(pi) = -0.5 * cross(pj-pi, pk-pi) / |...| * ... 
    # Simpler: use the signed area formula
    # A = 0.5 * ((pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x))
    # dA/dpi_x = 0.5 * (-(pk_y - pi_y) + (pj_y - pi_y)) = 0.5 * (pj_y - pk_y)
    # dA/dpi_y = 0.5 * ((pk_x - pi_x) - (pj_x - pi_x)) = 0.5 * (pk_x - pj_x)
    # dA/dpj_x = 0.5 * (pk_y - pi_y)
    # dA/dpj_y = 0.5 * (-(pk_x - pi_x)) = 0.5 * (pi_x - pk_x)
    # dA/dpk_x = 0.5 * (-(pj_y - pi_y)) = 0.5 * (pi_y - pj_y)
    # dA/dpk_y = 0.5 * (pj_x - pi_x)
    
    dA_dpi_x = 0.5 * (pj[:, 1] - pk[:, 1])
    dA_dpi_y = 0.5 * (pk[:, 0] - pj[:, 0])
    dA_dpj_x = 0.5 * (pk[:, 1] - pi[:, 1])
    dA_dpj_y = 0.5 * (pi[:, 0] - pk[:, 0])
    dA_dpk_x = 0.5 * (pi[:, 1] - pj[:, 1])
    dA_dpk_y = 0.5 * (pj[:, 0] - pi[:, 0])
    
    # d(pts)/d(abc): pts = a*V0 + b*V1 + c*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, H)
    # x = b + 0.5c, y = c*H
    # dx/da=0, dx/db=1, dx/dc=0.5
    # dy/da=0, dy/db=0, dy/dc=H
    
    # Build Jacobian for constraints: shape (_N_TRI, 1 + 3*N)
    jac = np.zeros((_N_TRI, 1 + 3 * N))
    jac[:, 0] = -1.0  # dt term
    
    # For each triangle i and each point p, add the gradient
    # Point p contributes to triangles where p is pi, pj, or pk
    for p_idx in range(N):
        # Find triangles where this point is pi, pj, or pk
        mask_pi = _TRI_IDX[:, 0] == p_idx
        mask_pj = _TRI_IDX[:, 1] == p_idx
        mask_pk = _TRI_IDX[:, 2] == p_idx
        
        col_base = 1 + p_idx * 3
        
        # Contribution from being pi
        if np.any(mask_pi):
            jac[mask_pi, col_base + 0] += dA_dpi_x[mask_pi] * 0.0 + dA_dpi_y[mask_pi] * 0.0  # da contribution: dx/da=0, dy/da=0
            jac[mask_pi, col_base + 1] += dA_dpi_x[mask_pi] * 1.0 + dA_dpi_y[mask_pi] * 0.0  # db: dx/db=1
            jac[mask_pi, col_base + 2] += dA_dpi_x[mask_pi] * 0.5 + dA_dpi_y[mask_pi] * H   # dc: dx/dc=0.5, dy/dc=H
        
        # Contribution from being pj
        if np.any(mask_pj):
            jac[mask_pj, col_base + 0] += 0.0
            jac[mask_pj, col_base + 1] += dA_dpj_x[mask_pj] * 1.0
            jac[mask_pj, col_base + 2] += dA_dpj_x[mask_pj] * 0.5 + dA_dpj_y[mask_pj] * H
        
        # Contribution from being pk
        if np.any(mask_pk):
            jac[mask_pk, col_base + 0] += 0.0
            jac[mask_pk, col_base + 1] += dA_dpk_x[mask_pk] * 1.0
            jac[mask_pk, col_base + 2] += dA_dpk_x[mask_pk] * 0.5 + dA_dpk_y[mask_pk] * H
    
    return jac

def _nlp_bounds():
    """Bounds for NLP: t in [0, 0.5], each barycentric coord in [0, 1]"""
    bounds = [(0.0, 0.5)]  # t
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    return bounds

def _objective_de(flat_pts):
    """
    Objective for Differential Evolution using Cartesian coordinates.
    Minimizes negative soft-min area + penalty for points outside triangle.
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    abc = _to_barycentric(pts)
    pen = np.sum(np.maximum(0, -abc) ** 2)
    
    pts_clamped = _clamp_all(pts)
    area_val = _soft_min_area(pts_clamped, beta=20.0)
    
    return -area_val + 100.0 * pen

def _initial_placement_symmetric():
    """Symmetric initial placement in Cartesian coordinates."""
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
    
    # Bounds for differential evolution (Cartesian)
    bounds_cart = [(0.0, 1.0)] * N + [(0.0, H)] * N
    
    # Phase 1: Differential Evolution for global search
    best_pts = _initial_placement_symmetric()
    try:
        if time_left() > 30.0:
            res_de = differential_evolution(
                _objective_de,
                bounds_cart,
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=12,
                polish=False,
                updating='immediate'
            )
            if res_de.fun < 1e6:
                candidate_pts = _clamp_all(res_de.x.reshape(-1, 2))
                if _min_area(candidate_pts) > _min_area(best_pts):
                    best_pts = candidate_pts
    except Exception:
        pass
    
    current_best = _min_area(best_pts)
    
    # Phase 2: SLSQP NLP optimization with auxiliary variable t
    if time_left() > 15.0:
        try:
            # Initialize with barycentric coordinates
            abc_init = _to_barycentric(best_pts)
            abc_init = _clamp_barycentric(abc_init)
            t_init = current_best * 0.9  # Start slightly below current best
            x_init = np.concatenate([[t_init], abc_init.flatten()])
            
            bounds_nlp = _nlp_bounds()
            
            res_slsqp = minimize(
                _nlp_objective,
                x_init,
                method='SLSQP',
                bounds=bounds_nlp,
                constraints=[{'type': 'ineq', 'fun': _nlp_constraints}],
                jac=_nlp_jacobian_objective,
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            
            if res_slsqp.success or res_slsqp.fun < -1e-6:
                t_opt = res_slsqp.x[0]
                abc_opt = _clamp_barycentric(res_slsqp.x[1:].reshape(-1, 3))
                pts_opt = _from_barycentric(abc_opt)
                area_opt = _min_area(pts_opt)
                if area_opt > current_best:
                    best_pts = pts_opt
                    current_best = area_opt
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B on smooth soft-min objective for refinement
    if time_left() > 10.0:
        try:
            flat_pts = best_pts.flatten()
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=80.0),
                flat_pts,
                method='L-BFGS-B',
                bounds=bounds_cart,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            if _min_area(polished_pts) > current_best:
                best_pts = polished_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass
    
    # Phase 4: Powell on hard objective
    if time_left() > 8.0:
        try:
            res_powell = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Powell',
                bounds=bounds_cart,
                options={'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-12}
            )
            final_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            if _min_area(final_pts) > current_best:
                best_pts = final_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass
    
    # Phase 5: Another SLSQP NLP pass for high-precision polish
    if time_left() > 5.0:
        try:
            abc_init = _to_barycentric(best_pts)
            abc_init = _clamp_barycentric(abc_init)
            t_init = current_best * 0.95
            x_init = np.concatenate([[t_init], abc_init.flatten()])
            
            bounds_nlp = _nlp_bounds()
            
            res_slsqp2 = minimize(
                _nlp_objective,
                x_init,
                method='SLSQP',
                bounds=bounds_nlp,
                constraints=[{'type': 'ineq', 'fun': _nlp_constraints}],
                jac=_nlp_jacobian_objective,
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            
            if res_slsqp2.success or res_slsqp2.fun < -1e-6:
                t_opt = res_slsqp2.x[0]
                abc_opt = _clamp_barycentric(res_slsqp2.x[1:].reshape(-1, 3))
                pts_opt = _from_barycentric(abc_opt)
                area_opt = _min_area(pts_opt)
                if area_opt > current_best:
                    best_pts = pts_opt
                    current_best = area_opt
        except Exception:
            pass
    
    # Phase 6: Final Nelder-Mead polish
    if time_left() > 3.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            if _min_area(final_pts) > current_best:
                best_pts = final_pts
        except Exception:
            pass
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
