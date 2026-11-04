# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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

def _smooth_min_area(x, beta=50.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    Returns (value, gradient) where value is the negative smooth min area.
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
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return 0.0, np.zeros_like(x)
    
    m = np.min(areas)
    shifted = areas - m
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    # Gradient computation
    weights = np.exp(-beta * shifted - max_exp) / np.sum(np.exp(-beta * shifted - max_exp))
    grad_val = -weights
    
    # Compute gradients of dets with respect to bary coordinates
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    P = bary_clamped
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    # d(det)/d(Pi), d(det)/d(Pj), d(det)/d(Pk)
    # det = Pi0*(Pj1*Pk2 - Pj2*Pk1) - Pi1*(Pj0*Pk2 - Pj2*Pk0) + Pi2*(Pj0*Pk1 - Pj1*Pk0)
    # In barycentric: P = [u, v, w] where w = 1-u-v
    # So dP/du = [1, 0, -1], dP/dv = [0, 1, -1]
    
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    # Contribution from Pi (point i)
    dPi_du = grad_val[:, None] * np.array([1, 0, -1])  # (ntriples, 3)
    dPi_dv = grad_val[:, None] * np.array([0, 1, -1])
    
    # For each triple, add contribution to point i
    for idx in range(N_TRIPLES):
        i, j, k = i_idx[idx], j_idx[idx], k_idx[idx]
        w_val = grad_val[idx]
        
        # d(det)/d(Pi) = (Pj1*Pk2 - Pj2*Pk1, -(Pj0*Pk2 - Pj2*Pk0), (Pj0*Pk1 - Pj1*Pk0))
        ddet_dPi = np.array([
            Pj[idx, 1] * Pk[idx, 2] - Pj[idx, 2] * Pk[idx, 1],
            -(Pj[idx, 0] * Pk[idx, 2] - Pj[idx, 2] * Pk[idx, 0]),
            Pj[idx, 0] * Pk[idx, 1] - Pj[idx, 1] * Pk[idx, 0]
        ])
        grad_u[i] += w_val * (ddet_dPi[0] - ddet_dPi[2])
        grad_v[i] += w_val * (ddet_dPi[1] - ddet_dPi[2])
        
        # d(det)/d(Pj) = -(Pi1*Pk2 - Pi2*Pk1, -(Pi0*Pk2 - Pi2*Pk0), (Pi0*Pk1 - Pi1*Pk0))
        ddet_dPj = np.array([
            -(Pi[idx, 1] * Pk[idx, 2] - Pi[idx, 2] * Pk[idx, 1]),
            Pi[idx, 0] * Pk[idx, 2] - Pi[idx, 2] * Pk[idx, 0],
            -(Pi[idx, 0] * Pk[idx, 1] - Pi[idx, 1] * Pk[idx, 0])
        ])
        grad_u[j] += w_val * (ddet_dPj[0] - ddet_dPj[2])
        grad_v[j] += w_val * (ddet_dPj[1] - ddet_dPj[2])
        
        # d(det)/d(Pk) = (Pi1*Pj2 - Pi2*Pj1, -(Pi0*Pj2 - Pi2*Pj0), (Pi0*Pj1 - Pi1*Pj0))
        ddet_dPk = np.array([
            Pi[idx, 1] * Pj[idx, 2] - Pi[idx, 2] * Pj[idx, 1],
            -(Pi[idx, 0] * Pj[idx, 2] - Pi[idx, 2] * Pj[idx, 0]),
            Pi[idx, 0] * Pj[idx, 1] - Pi[idx, 1] * Pj[idx, 0]
        ])
        grad_u[k] += w_val * (ddet_dPk[0] - ddet_dPk[2])
        grad_v[k] += w_val * (ddet_dPk[1] - ddet_dPk[2])
    
    grad = np.zeros(2 * N)
    grad[0::2] = grad_u
    grad[1::2] = grad_v
    
    return -smooth_min, grad

def _epigraph_objective(x, t=0.01, beta=100.0):
    """
    Epigraph formulation: minimize -t + penalty for areas < t.
    x: (2N+1,) array, last element is t.
    """
    bary = x[:2*N].reshape(N, 2)
    t_val = x[2*N]
    
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    areas = np.abs(dets)
    
    # Penalty: sum of max(0, t - area)^2 for all triples
    violations = np.maximum(0, t_val - areas)
    penalty = np.sum(violations ** 2)
    
    # Also penalize if t is too large (to keep it bounded)
    t_penalty = max(0, t_val - 0.5) ** 2
    
    return -t_val + penalty + t_penalty

def _epigraph_grad(x, t=0.01, beta=100.0):
    """
    Gradient of epigraph objective.
    """
    bary = x[:2*N].reshape(N, 2)
    t_val = x[2*N]
    
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    areas = np.abs(dets)
    
    # Gradient of penalty: 2 * sum(max(0, t - area)) * (-d(area)/d(x))
    viol_mask = t_val > areas
    viol_vals = np.maximum(0, t_val - areas)
    
    grad = np.zeros(2 * N + 1)
    
    # dt term: -1
    grad[2*N] = -1.0
    
    # t_penalty gradient
    if t_val > 0.5:
        grad[2*N] += 2 * (t_val - 0.5)
    
    # Penalty gradient w.r.t. areas: 2 * viol_vals * (-1) for violated
    area_grad = -2 * viol_vals
    
    # Now compute d(area)/d(bary) for each triple
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    P = bary_clamped
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    for idx in range(N_TRIPLES):
        if not viol_mask[idx]:
            continue
        
        i, j, k = i_idx[idx], j_idx[idx], k_idx[idx]
        w_val = area_grad[idx]
        
        # d|det|/d(Pi) - need to handle sign of det
        sign = np.sign(dets[idx]) if dets[idx] != 0 else 1.0
        
        ddet_dPi = np.array([
            Pj[idx, 1] * Pk[idx, 2] - Pj[idx, 2] * Pk[idx, 1],
            -(Pj[idx, 0] * Pk[idx, 2] - Pj[idx, 2] * Pk[idx, 0]),
            Pj[idx, 0] * Pk[idx, 1] - Pj[idx, 1] * Pk[idx, 0]
        ])
        grad_u[i] += w_val * sign * (ddet_dPi[0] - ddet_dPi[2])
        grad_v[i] += w_val * sign * (ddet_dPi[1] - ddet_dPi[2])
        
        ddet_dPj = np.array([
            -(Pi[idx, 1] * Pk[idx, 2] - Pi[idx, 2] * Pk[idx, 1]),
            Pi[idx, 0] * Pk[idx, 2] - Pi[idx, 2] * Pk[idx, 0],
            -(Pi[idx, 0] * Pk[idx, 1] - Pi[idx, 1] * Pk[idx, 0])
        ])
        grad_u[j] += w_val * sign * (ddet_dPj[0] - ddet_dPj[2])
        grad_v[j] += w_val * sign * (ddet_dPj[1] - ddet_dPj[2])
        
        ddet_dPk = np.array([
            Pi[idx, 1] * Pj[idx, 2] - Pi[idx, 2] * Pj[idx, 1],
            -(Pi[idx, 0] * Pj[idx, 2] - Pi[idx, 2] * Pj[idx, 0]),
            Pi[idx, 0] * Pj[idx, 1] - Pi[idx, 1] * Pj[idx, 0]
        ])
        grad_u[k] += w_val * sign * (ddet_dPk[0] - ddet_dPk[2])
        grad_v[k] += w_val * sign * (ddet_dPk[1] - ddet_dPk[2])
    
    grad[0::2] = grad_u
    grad[1::2] = grad_v
    
    return grad

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

def _local_optimize_smooth(x0, bounds, maxiter=500, beta=50.0):
    """Run L-BFGS-B with smooth soft-min objective."""
    def obj_and_grad(x):
        val, grad = _smooth_min_area(x, beta=beta)
        return val, grad
    
    try:
        res = minimize(
            obj_and_grad, x0, method='L-BFGS-B',
            bounds=bounds,
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_epigraph(x0, bounds, maxiter=300):
    """Run L-BFGS-B with epigraph formulation."""
    # Extend x0 with initial t estimate
    bary0 = x0.reshape(N, 2)
    t0 = _min_area_ratio_from_bary(bary0) * 0.95
    
    x_full = np.concatenate([x0, [t0]])
    bounds_full = bounds + [(0.001, 0.5)]
    
    def obj_and_grad(x):
        val = _epigraph_objective(x)
        grad = _epigraph_grad(x)
        return val, grad
    
    try:
        res = minimize(
            obj_and_grad, x_full, method='L-BFGS-B',
            bounds=bounds_full,
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        return res.x[:2*N]
    except Exception:
        return x0

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

def _simulated_annealing(bary0, time_budget=60.0, T_init=0.01, T_min=0.0001, cooling=0.97):
    """
    Simulated annealing with point swaps and local jitter.
    """
    start = time.time()
    current = bary0.copy()
    current_score = _min_area_ratio_from_bary(current)
    best = current.copy()
    best_score = current_score
    no_improve_count = 0
    max_no_improve = 200
    T = T_init
    
    # Precompute pair distances for swap proposals
    while time.time() - start < time_budget:
        # Choose a move type: swap or jitter
        move_type = np.random.random()
        
        if move_type < 0.3:
            # Swap two points
            i, j = np.random.choice(N, size=2, replace=False)
            candidate = current.copy()
            candidate[i], candidate[j] = candidate[j].copy(), candidate[i].copy()
            candidate = _project_to_bary_simplex(candidate)
        elif move_type < 0.6:
            # Move one point
            i = np.random.randint(N)
            candidate = current.copy()
            sigma = T * 5
            candidate[i] += np.random.normal(0, sigma, size=2)
            candidate = _project_to_bary_simplex(candidate)
        else:
            # Small global jitter
            candidate = current.copy()
            sigma = T * 2
            candidate += np.random.normal(0, sigma, size=(N, 2))
            candidate = _project_to_bary_simplex(candidate)
        
        cand_score = _min_area_ratio_from_bary(candidate)
        
        # Accept if better, or with probability exp(-delta/T)
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta = current_score - cand_score
            if delta < T:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current = candidate
                    current_score = cand_score
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # If stuck for too long, reheat slightly
        if no_improve_count > max_no_improve:
            T = max(T, current_score * 0.02)
            no_improve_count = 0
    
    return best, best_score

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

            # 2.5. Smooth L-BFGS-B with soft-min
            if time_left() > 10:
                x_smooth = _local_optimize_smooth(init_bary.flatten(), bounds, maxiter=300, beta=50.0)
                bary_smooth = _project_to_bary_simplex(x_smooth.reshape(N, 2))
                score_smooth = _min_area_ratio_from_bary(bary_smooth)
                if score_smooth > best_local_score:
                    best_local_score = score_smooth
                    best_local_x = bary_smooth
                    init_bary = bary_smooth

            # 2.6. Epigraph L-BFGS-B
            if time_left() > 10:
                x_epi = _local_optimize_epigraph(init_bary.flatten(), bounds, maxiter=200)
                bary_epi = _project_to_bary_simplex(x_epi.reshape(N, 2))
                score_epi = _min_area_ratio_from_bary(bary_epi)
                if score_epi > best_local_score:
                    best_local_score = score_epi
                    best_local_x = bary_epi
                    init_bary = bary_epi

            # 3. Simulated Annealing
            if time_left() > 15:
                sa_bary, sa_score = _simulated_annealing(init_bary, time_budget=min(30.0, time_left() - 10))
                if sa_score > best_local_score:
                    best_local_score = sa_score
                    best_local_x = sa_bary
                    init_bary = sa_bary

            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
                
        except Exception:
            pass

    # If we have time, do a few random restarts with multiple methods
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
                
                # Try multiple methods
                local_best_bary = init_bary
                local_best_score = _min_area_ratio_from_bary(init_bary)
                
                if time_left() > 15:
                    x_slsqp = _local_optimize_slsqp(init_bary.flatten(), bounds, maxiter=100)
                    bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                    score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                    if score_slsqp > local_best_score:
                        local_best_score = score_slsqp
                        local_best_bary = bary_slsqp
                
                if time_left() > 15:
                    x_smooth = _local_optimize_smooth(init_bary.flatten(), bounds, maxiter=200, beta=50.0)
                    bary_smooth = _project_to_bary_simplex(x_smooth.reshape(N, 2))
                    score_smooth = _min_area_ratio_from_bary(bary_smooth)
                    if score_smooth > local_best_score:
                        local_best_score = score_smooth
                        local_best_bary = bary_smooth
                
                if local_best_score > best_score:
                    best_score = local_best_score
                    best_x = local_best_bary
            except Exception:
                pass

    if best_x is None:
        best_x = _structured_init_fps()
    
    # Final high-precision polish on the best solution
    if time_left() > 15:
        try:
            bounds_final = [(0, 1)] * (2 * N)
            
            # First try smooth L-BFGS-B with high beta
            if time_left() > 20:
                x_smooth_final = _local_optimize_smooth(best_x.flatten(), bounds_final, maxiter=500, beta=200.0)
                bary_smooth_final = _project_to_bary_simplex(x_smooth_final.reshape(N, 2))
                score_smooth_final = _min_area_ratio_from_bary(bary_smooth_final)
                if score_smooth_final > best_score:
                    best_score = score_smooth_final
                    best_x = bary_smooth_final
            
            # Then epigraph
            if time_left() > 20:
                x_epi_final = _local_optimize_epigraph(best_x.flatten(), bounds_final, maxiter=300)
                bary_epi_final = _project_to_bary_simplex(x_epi_final.reshape(N, 2))
                score_epi_final = _min_area_ratio_from_bary(bary_epi_final)
                if score_epi_final > best_score:
                    best_score = score_epi_final
                    best_x = bary_epi_final
            
            # Then standard polish
            polished = _final_polish(best_x.flatten(), bounds_final, time_budget=min(30.0, time_left() - 10))
            polished_bary = _project_to_bary_simplex(polished.reshape(N, 2))
            polished_score = _min_area_ratio_from_bary(polished_bary)
            if polished_score > best_score:
                best_x = polished_bary
                best_score = polished_score
        except Exception:
            pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
