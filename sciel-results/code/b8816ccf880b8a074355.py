"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
# Area of this triangle
TRI_AREA = S3 / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])


def _bary_to_cart(bc):
    """Convert barycentric (b, c) coordinates (n,2) to Cartesian (n,2).
    a = 1 - b - c
    x = b + 0.5*c
    y = c*sqrt(3)/2
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (S3 / 2)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian (n,2) to barycentric (b, c) coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / (S3 / 2)
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _clamp_bary(bc):
    """Clamp barycentric coordinates to valid range: b>=0, c>=0, b+c<=1."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    
    # c >= 0
    c = np.clip(c, 0, None)
    # b >= 0
    b = np.clip(b, 0, None)
    # b + c <= 1
    excess = b + c - 1.0
    if np.any(excess > 0):
        # Scale down proportionally
        scale = 1.0 / (1.0 + excess)
        b = b * scale
        c = c * scale
    
    return np.column_stack([b, c])


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


def _min_triangle_area_fast(points):
    """Compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    bary = _cart_to_bary(points)
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))


def _objective_bary(bc_flat):
    """
    Objective function for SLSQP.
    Minimizes the negative of the minimum triangle area.
    Input: flat array of length 2*N, representing (b_i, c_i) for each point.
    """
    bc = bc_flat.reshape(N, 2)
    # Clamp to ensure validity
    bc = _clamp_bary(bc)
    points = _bary_to_cart(bc)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords (barycentric).
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords (u,v,w < 0)
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    excess2 = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(excess2**2)
    excess3 = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(excess3**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # log-sum-exp for numerical stability
    max_shift = np.max(-beta * shifted)
    log_sum_exp = max_shift + np.log(np.sum(np.exp(-beta * shifted - max_shift)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty


def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    # SLSQP handles bounds, but we clamp for safety in area calc
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    # Ensure u+v <= 1
    excess = u + v - 1.0
    if np.any(excess > 0):
        scale = 1.0 / (1.0 + excess)
        u = u * scale
        v = v * scale
    
    bary_clamped = np.column_stack([u, v])
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.default_rng(42)
    
    # Constraints for SLSQP in barycentric coordinates (b, c)
    # a = 1 - b - c >= 0  =>  b + c <= 1
    # b >= 0
    # c >= 0
    
    constraints = []
    for i in range(N):
        # b_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        # c_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # 1 - b_i - c_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    # Bounds for b and c: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def random_bary_init():
        """Generate N random points in barycentric coordinates inside the triangle."""
        bc_init = np.zeros((N, 2))
        for k in range(N):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            bc_init[k, 0] = u
            bc_init[k, 1] = v
        return bc_init
    
    # Phase 1: Multi-start SLSQP in barycentric coordinates
    n_starts_slsqp = 15
    
    for i in range(n_starts_slsqp):
        if time.time() > deadline - 10.0:
            break
            
        bc_init = random_bary_init()
        x0 = bc_init.flatten()
        
        try:
            result = minimize(
                _objective_slsqp,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            bc_opt = result.x.reshape(N, 2)
            points_opt = _bary_to_cart(bc_opt)
            
            # Safety clip
            points_opt = np.array([_clamp_bary(bc_opt).reshape(-1, 2)]).reshape(N, 2)
            points_opt = _bary_to_cart(_clamp_bary(bc_opt))
            
            score = _min_triangle_area_fast(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
                
        except Exception:
            continue
    
    # Phase 2: If we have time, do L-BFGS-B refinement with smooth objective
    if best_points is not None and time.time() < deadline - 15.0:
        best_bary = _cart_to_bary(best_points)
        x_start = best_bary.flatten()
        
        # Try a few different beta values
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                result = minimize(
                    lambda x: _objective_smooth(x, beta),
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                bc_opt = result.x.reshape(N, 2)
                bc_opt = _clamp_bary(bc_opt)
                points_opt = _bary_to_cart(bc_opt)
                
                score = _min_triangle_area_fast(points_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    x_start = bc_opt.flatten()
                    
            except Exception:
                continue
    
    # Phase 3: Final Nelder-Mead refinement in Cartesian if time permits
    if best_points is not None and time.time() < deadline - 5.0:
        x_cart = best_points.flatten()
        
        def _objective_cart(x):
            points = x.reshape(-1, 2)
            # Project to triangle
            points = _clip_to_triangle(points)
            return -_min_triangle_area_fast(points)
        
        try:
            result = minimize(
                _objective_cart,
                x_cart,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            points_opt = result.x.reshape(N, 2)
            points_opt = _clip_to_triangle(points_opt)
            
            score = _min_triangle_area_fast(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: random points
        bc_init = random_bary_init()
        best_points = _bary_to_cart(bc_init)
    
    # Final safety check: ensure returned points are inside
    best_points = _clip_to_triangle(best_points)
    
    return best_points


def _clip_to_triangle(points):
    """Project points into the equilateral triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # y bounds
    y_max = S3 / 2
    y = np.clip(y, 0, y_max)
    
    # x bounds based on y
    # Left edge: x >= y / sqrt(3)
    # Right edge: x <= 1 - y / sqrt(3)
    x_min = y / S3
    x_max = 1 - y / S3
    
    # If x is out of [x_min, x_max], project to the nearest edge
    x = np.clip(x, x_min, x_max)
    
    return np.column_stack([x, y])
