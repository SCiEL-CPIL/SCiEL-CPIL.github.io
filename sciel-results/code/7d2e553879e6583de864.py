"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

N = 11
S3 = np.sqrt(3.0)
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
TRI_AREA = S3 / 4.0

_IDX_TRIPLES = np.array(
    [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
)
_I = _IDX_TRIPLES[:, 0]
_J = _IDX_TRIPLES[:, 1]
_K = _IDX_TRIPLES[:, 2]


def _bary_to_cart(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (S3 / 2.0)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    x = points[:, 0]
    y = points[:, 1]
    c = y / (S3 / 2.0)
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_to_bary_simplex(bc):
    """Project each row (b, c) into the barycentric simplex {b>=0, c>=0, b+c<=1}."""
    b = bc[:, 0].astype(float).copy()
    c = bc[:, 1].astype(float).copy()
    b = np.clip(b, 0.0, None)
    c = np.clip(c, 0.0, None)
    excess = b + c - 1.0
    mask = excess > 0
    if np.any(mask):
        s = 1.0 / (1.0 + excess[mask])
        b[mask] *= s
        c[mask] *= s
    return np.column_stack([b, c])


def _clip_to_triangle(points):
    x = points[:, 0].astype(float).copy()
    y = points[:, 1].astype(float).copy()
    y = np.clip(y, 0.0, S3 / 2.0)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _dets_from_bary(bary):
    """Twice-area determinants for all 165 triples, in barycentric coordinates."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui, vi, wi = u[_I], v[_I], w[_I]
    uj, vj, wj = u[_J], v[_J], w[_J]
    uk, vk, wk = u[_K], v[_K], w[_K]
    det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
    return det


def _min_area_ratio_from_bary(bary):
    """Minimum triangle area as a fraction of the triangle area."""
    dets = _dets_from_bary(bary)
    return float(np.min(np.abs(dets)))


def _combinatorial_refine(points, deadline):
    """
    Attempt to improve the minimum area by swapping points that are part of
    the minimum area triangles with other points, if it increases the min area.
    """
    bary = _cart_to_bary(points)
    best_bary = bary.copy()
    best_score = _min_area_ratio_from_bary(best_bary)
    
    # Pre-compute all triangle areas for current configuration
    dets = _dets_from_bary(best_bary)
    A = np.abs(dets)
    min_area = np.min(A)
    
    # Find indices of triangles that achieve the minimum area
    min_indices = np.where(A <= min_area * 1.000001)[0]
    
    # Points involved in these minimum triangles
    involved_points = set()
    for idx in min_indices:
        involved_points.add(_I[idx])
        involved_points.add(_J[idx])
        involved_points.add(_K[idx])
        
    # Try swapping each involved point with every other point
    # We only swap if the new min area is strictly larger
    for p_idx in list(involved_points):
        if time.time() > deadline:
            break
            
        for q_idx in range(N):
            if q_idx == p_idx:
                continue
                
            # Swap p_idx and q_idx in barycentric coordinates
            new_bary = best_bary.copy()
            new_bary[p_idx] = best_bary[q_idx]
            new_bary[q_idx] = best_bary[p_idx]
            
            # Project to simplex to ensure validity
            new_bary = _project_to_bary_simplex(new_bary)
            
            new_dets = _dets_from_bary(new_bary)
            new_A = np.abs(new_dets)
            new_min_area = np.min(new_A)
            
            if new_min_area > best_score * (1 + 1e-10):
                best_bary = new_bary
                best_score = new_min_area
                # Update min indices and involved points for next iteration
                min_indices = np.where(new_A <= new_min_area * 1.000001)[0]
                involved_points = set()
                for idx in min_indices:
                    involved_points.add(_I[idx])
                    involved_points.add(_J[idx])
                    involved_points.add(_K[idx])
                # Break out of q loop to restart p loop with new state
                break
                
    return _bary_to_cart(best_bary)

def _structured_init_fps(rng, n_init):
    """Farthest-point-sampling in barycentric coords: start from 3 vertices, then
    repeatedly add the point maximizing the minimum pairwise distance."""
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    for _ in range(n_init - 3):
        best_idx = -1
        best_val = -1.0
        for t in range(4000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            cand = np.array([u, v])
            d2 = np.min(np.sum((np.array(pts) - cand) ** 2, axis=1))
            if d2 > best_val:
                best_val = d2
                best_idx = cand
        pts.append(best_idx)
    return np.array(pts, dtype=float)


def _dets_and_grads(bary):
    """Compute all 165 determinants and their gradients w.r.t. barycentric coords.
    Returns (dets, grad_u, grad_v) where grad_u[i] is d(det)/du_i for each triple."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui, vi, wi = u[_I], v[_I], w[_I]
    uj, vj, wj = u[_J], v[_J], w[_J]
    uk, vk, wk = u[_K], v[_K], w[_K]
    det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
    
    # Gradients for each point in each triple
    gu_i = vj * wk - wj * vk
    gv_i = -(uj * wk - wj * uk)
    gu_j = -vi * wk + wi * vk
    gv_j = vi * wk - wi * uk
    gu_k = vi * wj - wi * vj
    gv_k = ui * wj - wj * ui
    
    return det, (gu_i, gv_i, gu_j, gv_j, gu_k, gv_k)


def _objective_qp(x):
    """
    QP-style objective: minimize -(min det^2) using a smooth approximation.
    We maximize t such that det^2 >= t^2 for all triples.
    Equivalent to: minimize -t where t = min(|det|).
    Use log-sum-exp for smoothness of the minimum.
    """
    bary = x.reshape(N, 2)
    dets, grads = _dets_and_grads(bary)
    A2 = dets ** 2  # squared areas
    m2 = np.min(A2)
    
    # Smooth min of A2 using log-sum-exp
    beta = 200.0
    shifted = A2 - m2
    z = -beta * shifted
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    logsum = zmax + np.log(s)
    smooth_min_sq = m2 - (1.0 / beta) * logsum
    
    # Gradient of smooth_min_sq w.r.t. dets
    expz = np.exp(z - zmax)
    weights = expz / s
    # d(smooth_min_sq)/d(dets_t) = -2 * weights_t * dets_t
    gdets = -2.0 * weights * dets
    
    # Accumulate gradient w.r.t. barycentric coords
    gu = np.zeros(N)
    gv = np.zeros(N)
    gu_i, gv_i, gu_j, gv_j, gu_k, gv_k = grads
    
    for t in range(len(dets)):
        gd = gdets[t]
        if abs(gd) < 1e-15:
            continue
        gu[_I[t]] += gd * gu_i[t]
        gv[_I[t]] += gd * gv_i[t]
        gu[_J[t]] += gd * gu_j[t]
        gv[_J[t]] += gd * gv_j[t]
        gu[_K[t]] += gd * gu_k[t]
        gv[_K[t]] += gd * gv_k[t]
    
    gx = np.zeros_like(x)
    gx[0::2] += gu
    gx[1::2] += gv
    
    value = -smooth_min_sq
    return value, gx


def _objective_slsqp(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def _objective_exact(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def _simulated_annealing(bary_init, start_time, deadline, rng):
    """Simulated annealing with jitter and swaps to escape local optima."""
    best_bary = bary_init.copy()
    best_score = _min_area_ratio_from_bary(best_bary)
    current_bary = best_bary.copy()
    current_score = best_score
    
    T0 = 0.01
    T_end = 0.0001
    cooling = 0.995
    
    dets = _dets_from_bary(current_bary)
    A = np.abs(dets)
    min_area = np.min(A)
    
    while time.time() < deadline and T0 > T_end:
        T0 *= cooling
        
        # Choose a random point to perturb
        p_idx = rng.integers(0, N)
        
        # Random jitter in barycentric coords
        delta_u = rng.normal(0, 0.05)
        delta_v = rng.normal(0, 0.05)
        
        new_bary = current_bary.copy()
        new_bary[p_idx, 0] += delta_u
        new_bary[p_idx, 1] += delta_v
        new_bary = _project_to_bary_simplex(new_bary)
        
        new_dets = _dets_from_bary(new_bary)
        new_A = np.abs(new_dets)
        new_min = np.min(new_A)
        
        delta = new_min - current_score
        if delta > 0 or rng.random() < np.exp(delta / T0):
            current_bary = new_bary
            current_score = new_min
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
        else:
            # Occasionally try a swap
            if rng.random() < 0.1:
                q_idx = rng.integers(0, N)
                if q_idx != p_idx:
                    swap_bary = current_bary.copy()
                    swap_bary[p_idx] = current_bary[q_idx]
                    swap_bary[q_idx] = current_bary[p_idx]
                    swap_bary = _project_to_bary_simplex(swap_bary)
                    swap_dets = _dets_from_bary(swap_bary)
                    swap_min = np.min(np.abs(swap_dets))
                    swap_delta = swap_min - current_score
                    if swap_delta > 0 or rng.random() < np.exp(swap_delta / T0):
                        current_bary = swap_bary
                        current_score = swap_min
                        if current_score > best_score:
                            best_score = current_score
                            best_bary = current_bary.copy()
    
    return best_bary, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s

    rng = np.random.default_rng(1234)

    # Linear constraint matrix for barycentric simplex: u >= 0, v >= 0, u+v <= 1
    # Variables are [u0, v0, u1, v1, ..., u10, v10]
    A_lin = np.zeros((3 * N, 2 * N))
    b_lin = np.zeros(3 * N)
    for i in range(N):
        # u_i >= 0 -> -u_i <= 0
        A_lin[3 * i, 2 * i] = -1.0
        # v_i >= 0 -> -v_i <= 0
        A_lin[3 * i + 1, 2 * i + 1] = -1.0
        # u_i + v_i <= 1
        A_lin[3 * i + 2, 2 * i] = 1.0
        A_lin[3 * i + 2, 2 * i + 1] = 1.0
        b_lin[3 * i + 2] = 1.0
    
    lc = LinearConstraint(A_lin, -np.inf, b_lin)
    bounds = [(0.0, 1.0)] * (2 * N)

    best_points = None
    best_score = -float('inf')

    def consider(bary):
        nonlocal best_points, best_score
        bary = _project_to_bary_simplex(bary)
        points = _bary_to_cart(bary)
        points = _clip_to_triangle(points)
        score = _min_area_ratio_from_bary(_cart_to_bary(points))
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score

    # Phase 1: multi-start from structured FPS inits
    n_starts = 6
    for s in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        bary_init = _structured_init_fps(rng, N)
        x0 = bary_init.flatten()
        consider(bary_init)

        # 1a: QP-style TRF on squared-area smooth objective
        if time.time() < deadline - 15.0:
            try:
                res = minimize(
                    _objective_qp, x0, method='trust-constr',
                    jac=True, bounds=bounds, constraints=[lc],
                    options={'maxiter': 200, 'gtol': 1e-7, 'barrier_tol': 1e-8}
                )
                bary_opt = _project_to_bary_simplex(res.x.reshape(N, 2))
                consider(bary_opt)
                x0 = bary_opt.flatten()
            except Exception:
                pass

        # 1b: SLSQP on exact objective
        if time.time() < deadline - 12.0:
            try:
                res = minimize(_objective_slsqp, x0, method='SLSQP',
                               bounds=bounds, constraints=[lc],
                               options={'maxiter': 150, 'ftol': 1e-9})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

        # 1c: Simulated annealing
        if time.time() < deadline - 8.0:
            sa_bary, sa_score = _simulated_annealing(
                _project_to_bary_simplex(x0.reshape(N, 2)),
                start_time, deadline - 5.0, rng
            )
            consider(sa_bary)
            if sa_score > best_score:
                x0 = sa_bary.flatten()

        # 1d: Combinatorial Refinement
        if time.time() < deadline - 5.0:
            improved_points = _combinatorial_refine(best_points, deadline - 2.0)
            consider(_cart_to_bary(improved_points))

    # Phase 2: refine the global best
    if best_points is not None:
        bary_best = _cart_to_bary(best_points)
        x_best = bary_best.flatten()
        consider(bary_best)

        # 2a: QP-style TRF with more iterations
        if time.time() < deadline - 10.0:
            try:
                res = minimize(
                    _objective_qp, x_best, method='trust-constr',
                    jac=True, bounds=bounds, constraints=[lc],
                    options={'maxiter': 300, 'gtol': 1e-8, 'barrier_tol': 1e-9}
                )
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

        # 2b: SLSQP exact
        if time.time() < deadline - 8.0:
            try:
                res = minimize(_objective_slsqp, x_best, method='SLSQP',
                               bounds=bounds, constraints=[lc],
                               options={'maxiter': 250, 'ftol': 1e-10})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

        # 2c: Nelder-Mead on exact (Cartesian)
        if time.time() < deadline - 5.0:
            def obj_cart(x):
                pts = _clip_to_triangle(x.reshape(N, 2))
                return -_min_area_ratio_from_bary(_cart_to_bary(pts))
            try:
                res = minimize(obj_cart, best_points.flatten(), method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-10,
                                        'fatol': 1e-12, 'adaptive': True})
                consider(_cart_to_bary(_clip_to_triangle(res.x.reshape(N, 2))))
            except Exception:
                pass
        
        # 2d: Final simulated annealing polish
        if time.time() < deadline - 3.0:
            sa_bary, sa_score = _simulated_annealing(
                _cart_to_bary(best_points),
                start_time, deadline - 1.0, rng
            )
            consider(sa_bary)

        # 2e: Final Combinatorial Refinement
        if time.time() < deadline - 1.0:
            improved_points = _combinatorial_refine(best_points, deadline - 0.5)
            consider(_cart_to_bary(improved_points))

    if best_points is None:
        bary_init = _structured_init_fps(rng, N)
        best_points = _bary_to_cart(bary_init)

    return _clip_to_triangle(best_points)
