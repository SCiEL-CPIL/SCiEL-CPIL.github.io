"""Improved program for packing 50 equal circles in a unit circle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # 5s buffer before 150s kill


def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    return min(r_pair, r_bound)


def _energy(P, temperature=0.01):
    """
    Smooth surrogate energy: sum of soft min of pairwise distances/2 and boundary distances.
    Lower energy = better packing.
    """
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_r = dist / 2.0
    # Soft min using log-sum-exp for pairwise
    # We want min over all pair distances and boundary distances
    all_vals = np.concatenate([d_bound, pair_r[np.triu_indices_from(pair_r, k=1)]])
    # Use log-sum-exp for soft min: -t * log(sum(exp(-x/t)))
    # For numerical stability
    min_val = np.min(all_vals)
    if min_val < -100:
        return -min_val  # very bad
    log_sum = np.log(np.sum(np.exp(-(all_vals - min_val) / temperature)))
    soft_min = min_val - temperature * log_sum
    return -soft_min  # we want to minimize this, so negate


def _energy_grad(P, temperature=0.01):
    """Gradient of the smooth energy."""
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    dist_origin = np.linalg.norm(P, axis=1)
    dist_origin_safe = np.where(dist_origin > 1e-12, dist_origin, 1e-12)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_r = dist / 2.0
    
    # Soft min weights
    all_vals = np.concatenate([d_bound, pair_r[np.triu_indices_from(pair_r, k=1)]])
    min_val = np.min(all_vals)
    if min_val < -100:
        # Fallback: just push everything toward center
        grad = np.zeros_like(P)
        for i in range(N):
            if dist_origin[i] > 1e-12:
                grad[i] = P[i] / dist_origin[i]
        return grad
    
    exp_vals = np.exp(-(all_vals - min_val) / temperature)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of soft min w.r.t. each variable
    # d(soft_min)/d(x_k) = -weights_k
    # d(energy)/d(x_k) = -d(soft_min)/d(x_k) = weights_k
    
    # Boundary terms: d_bound_i = 1 - |P_i|
    # d(d_bound_i)/d(P_i) = -P_i / |P_i|
    # Boundary weights
    n_bound = N
    n_pair = N * (N - 1) // 2
    w_bound = weights[:n_bound]
    w_pair = weights[n_bound:]
    
    grad = np.zeros_like(P)
    
    # Boundary contribution
    # energy = -soft_min, so d(energy)/d(P_i) = -d(soft_min)/d(P_i)
    # d(soft_min)/d(P_i) = -w_bound_i * d(d_bound_i)/d(P_i) = -w_bound_i * (-P_i/|P_i|) = w_bound_i * P_i/|P_i|
    # So d(energy)/d(P_i) from boundary = -w_bound_i * P_i/|P_i|
    for i in range(N):
        if dist_origin[i] > 1e-12:
            grad[i] -= w_bound[i] * P[i] / dist_origin[i]
    
    # Pairwise contribution
    # pair_r_ij = |P_i - P_j| / 2
    # d(pair_r_ij)/d(P_i) = (P_i - P_j) / (2 * |P_i - P_j|)
    # d(soft_min)/d(P_i) from pair (i,j) = -w_pair * d(pair_r_ij)/d(P_i)
    # d(energy)/d(P_i) = -d(soft_min)/d(P_i) = w_pair * (P_i - P_j) / (2*dist_ij)
    
    # Map upper triangle indices to (i, j) pairs
    i_idx, j_idx = np.triu_indices_from(pair_r, k=1)
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        w = w_pair[k]
        d = dist[i, j]
        if d > 1e-12:
            direction = (P[i] - P[j]) / d
            grad[i] += w * direction / 2.0
            grad[j] -= w * direction / 2.0
    
    return grad


def _get_ring_initialization():
    """
    Generate a ring-structured initialization: 1 center + rings of 6, 12, 18, 13.
    """
    points = []
    # Center
    points.append([0.0, 0.0])
    
    # Ring parameters: (count, radius_fraction)
    # We'll tune these
    rings = [
        (6, 0.25),
        (12, 0.50),
        (18, 0.75),
        (13, 0.95),
    ]
    
    for count, r_frac in rings:
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            points.append([r_frac * np.cos(a), r_frac * np.sin(a)])
    
    P = np.array(points[:N])
    return P


def _get_hex_initial_points():
    """Generate a hexagonal lattice and select the best N points."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def _local_refine(P, deadline):
    """Local refinement using L-BFGS-B with smooth energy, then Nelder-Mead."""
    P_best = P.copy()
    score_best = _compute_score(P_best)
    
    # L-BFGS-B with smooth energy
    try:
        res = minimize(_energy, P_best.flatten(), method='L-BFGS-B',
                       jac=lambda x: _energy_grad(x.reshape(N, 2)),
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        P_opt = res.x.reshape(N, 2)
        score_opt = _compute_score(P_opt)
        if score_opt > score_best:
            P_best = P_opt
            score_best = score_opt
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return P_best, score_best
    
    # Nelder-Mead for final polish
    try:
        res2 = minimize(lambda x: -_compute_score(x.reshape(N, 2)), P_best.flatten(),
                        method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
        P_opt2 = res2.x.reshape(N, 2)
        score_opt2 = _compute_score(P_opt2)
        if score_opt2 > score_best:
            P_best = P_opt2
            score_best = score_opt2
    except Exception:
        pass
    
    return P_best, score_best


def _simulated_annealing(P, deadline, max_iters=5000):
    """Simulated annealing with random perturbations."""
    P_curr = P.copy()
    r_curr = _compute_score(P_curr)
    P_best = P_curr.copy()
    r_best = r_curr
    
    T = 0.05
    T_min = 0.001
    cooling = 0.995
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation
        P_new = P_curr.copy()
        n_perturb = np.random.randint(1, 5)
        idx = np.random.choice(N, n_perturb, replace=False)
        perturbations = np.random.normal(0, T, (n_perturb, 2))
        P_new[idx] += perturbations
        
        r_new = _compute_score(P_new)
        
        if r_new > r_curr or np.random.random() < np.exp((r_new - r_curr) / max(T, 1e-8)):
            P_curr = P_new
            r_curr = r_new
            if r_new > r_best:
                P_best = P_new.copy()
                r_best = r_new
        
        T = max(T * cooling, T_min)
    
    return P_best, r_best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Multiple initializations
    P_ring = _get_ring_initialization()
    P_hex = _get_hex_initial_points()
    
    P_best = P_ring.copy()
    score_best = max(_compute_score(P_ring), _compute_score(P_hex))
    if _compute_score(P_hex) > score_best:
        P_best = P_hex.copy()
    
    # Local refinement from both starts
    P1, s1 = _local_refine(P_ring, deadline)
    if s1 > score_best:
        P_best = P1
        score_best = s1
    
    if time.time() > deadline - 30:
        return P_best
    
    P2, s2 = _local_refine(P_hex, deadline)
    if s2 > score_best:
        P_best = P2
        score_best = s2
    
    # Differential evolution on a subset of parameters for global search
    if time.time() > deadline - 40:
        # Try simulated annealing
        P3, s3 = _simulated_annealing(P_best, deadline, max_iters=2000)
        if s3 > score_best:
            P_best = P3
            score_best = s3
        return P_best
    
    # Differential evolution: optimize ring parameters
    def _ring_params_to_points(params):
        """Convert ring parameters to point positions."""
        # params: [cx, cy, r1, r2, r3, r4, a1, a2, a3, a4]
        # where r_i are ring radii, a_i are angular offsets
        cx, cy = params[0], params[1]
        r1, r2, r3, r4 = params[2:6]
        a1, a2, a3, a4 = params[6:10]
        
        points = [[cx, cy]]
        
        counts = [6, 12, 18, 13]
        radii = [r1, r2, r3, r4]
        offsets = [a1, a2, a3, a4]
        
        for count, r, off in zip(counts, radii, offsets):
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + off
            for a in angles:
                points.append([cx + r * np.cos(a), cy + r * np.sin(a)])
        
        P = np.array(points[:N])
        return P
    
    def _neg_score_ring(params):
        P = _ring_params_to_points(params)
        return -_compute_score(P)
    
    try:
        bounds = [(-0.1, 0.1), (-0.1, 0.1),
                  (0.1, 0.4), (0.3, 0.6), (0.5, 0.8), (0.7, 1.0),
                  (0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)]
        
        de_res = differential_evolution(_neg_score_ring, bounds, maxiter=50, 
                                         popsize=15, seed=42, tol=1e-8,
                                         polish=True, workers=1)
        P_de = _ring_params_to_points(de_res.x)
        score_de = _compute_score(P_de)
        if score_de > score_best:
            P_best = P_de
            score_best = score_de
    except Exception:
        pass
    
    # Final local refinement
    if time.time() < deadline - 10:
        P_final, score_final = _local_refine(P_best, deadline)
        if score_final > score_best:
            P_best = P_final
            score_best = score_final
    
    return P_best
