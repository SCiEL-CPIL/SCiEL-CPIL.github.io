"""
Circle packing: maximize radius of 50 equal circles in unit disk.
Combined approach:
1. Position-Based Dynamics (PBD) for robust initial packing and constraint satisfaction.
2. Soft-min surrogate with L-BFGS-B for smooth gradient-based refinement.
3. Nelder-Mead polish on the exact objective for final precision.
4. Multiple diverse initializations (Hex, Ring, Random).
5. Explicit deadline guard.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None
_DEADLINE = 140.0  # Safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0

    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin >= 1.0):
        return 0.0

    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)

    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Differentiable soft-min objective and gradient.
    Minimizing this maximizes the soft-min of the margins.
    """
    pts = flat.reshape(N, 2)

    # Boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))

    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]

    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])

    # Soft min with numerical stability
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to minimize -soft_min(x) = 1/beta * log(sum(exp(-beta * x)))
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta

    # Gradient
    # d/dx (1/beta * log(sum(exp(-beta*x)))) = 1/beta * (1/S) * sum(exp(-beta*x_i) * (-beta * dx_i))
    # = - (1/S) * sum(exp(-beta*x_i) * dx_i)
    # But we are minimizing the negative soft min? 
    # Let's check signs.
    # We want to MAXIMIZE min(dists).
    # soft_min approximates min.
    # If we minimize -soft_min, we maximize soft_min.
    # Let f(x) = -soft_min(x) = 1/beta * log(sum(exp(-beta*x)))
    # grad f = 1/beta * (1/S) * sum(exp(-beta*x_i) * (-beta * grad x_i))
    # grad f = - (1/S) * sum(exp(-beta*x_i) * grad x_i)
    
    weights = np.exp(y_shifted) / sum_exp # exp(-beta*x_i) / S

    grad_pts = np.zeros_like(pts)

    # Boundary terms: x_i = 1 - |c_i|
    # grad x_i = - c_i / |c_i|
    # Contribution to grad f: - w_i * (- c_i / |c_i|) = w_i * c_i / |c_i|
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    # grad_pts += w_boundary[:, np.newaxis] * pts / safe_dists[:, np.newaxis]
    # Wait, the formula derived above: grad f = - sum(w_i * grad x_i)
    # grad x_i (boundary) = - c_i / |c_i|
    # So term is - w_i * (- c_i / |c_i|) = w_i * c_i / |c_i|
    grad_pts += (w_boundary / safe_dists)[:, np.newaxis] * pts

    # Pairwise terms: x_ij = |c_i - c_j| / 2
    # grad x_ij w.r.t c_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # grad x_ij w.r.t c_j = (c_j - c_i) / (2 * |c_i - c_j|)
    # Contribution to grad f: - w_ij * grad x_ij
    # So for c_i: - w_ij * (c_i - c_j) / (2 * d_ij)
    # For c_j: - w_ij * (c_j - c_i) / (2 * d_ij)
    
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]

    # grad c_i part: -0.5 * w * unit_vec
    # grad c_j part: -0.5 * w * (-unit_vec) = 0.5 * w * unit_vec
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)

    grad_flat = grad_pts.flatten()

    return obj_val, grad_flat


def _pbd_relax(pts, target_r=0.125, n_iters=1000):
    """
    Position-Based Dynamics relaxation.
    Projects points to satisfy non-overlap and boundary constraints.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    limit = 1.0 - target_r

    for _ in range(n_iters):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break

        # 1. Resolve pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < target_d and d > 1e-10:
                    # Move apart by half the overlap
                    overlap = target_d - d
                    direction = diff / d
                    pts[i] += 0.5 * overlap * direction
                    pts[j] -= 0.5 * overlap * direction
                elif d <= 1e-10:
                    # If coincident, push randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    pts[i] += 0.5 * target_d * np.array([np.cos(angle), np.sin(angle)])
                    pts[j] -= 0.5 * target_d * np.array([np.cos(angle), np.sin(angle)])

        # 2. Resolve boundary violations
        dists = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists[i] > limit:
                # Project onto circle of radius 'limit'
                pts[i] = pts[i] * (limit / dists[i])
            elif dists[i] < 1e-10:
                # If at center, push out slightly to avoid singularity in future steps
                angle = np.random.uniform(0, 2 * np.pi)
                pts[i] = 0.1 * np.array([np.cos(angle), np.sin(angle)])

    return pts


def _hex_lattice_init(r_guess=0.125):
    """Generate a hexagonal lattice packing as initial guess."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(ring_sizes) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _random_init(seed):
    """Random initialization within unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        x = rng.uniform(-1, 1, size=(N * 2, 2))
        r = np.hypot(x[:, 0], x[:, 1])
        valid = r <= 0.95
        pts.extend(x[valid].tolist())
    pts = np.array(pts[:N])
    return pts


def _soft_min_optimize(pts_init, beta_schedule, max_iter_per_stage=300):
    """
    Multi-stage soft-min optimization with beta annealing.
    """
    current_pts = pts_init.copy()

    for stage, beta in enumerate(beta_schedule):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break

        def objective(x):
            return _soft_min_obj_and_grad(x, beta)

        x0 = current_pts.flatten()
        try:
            result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                              options={'maxiter': max_iter_per_stage, 'ftol': 1e-14,
                                       'gtol': 1e-10, 'maxls': 50})
            current_pts = result.x.reshape(N, 2)
        except Exception:
            continue

        # Project to ensure validity
        dists = np.linalg.norm(current_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / dists[mask, np.newaxis]

    return current_pts


def _polish(pts, budget_seconds=15.0):
    """
    Final Nelder-Mead polish on the true nonsmooth objective.
    """
    t0 = time.time()

    def neg_score(x):
        pts_local = x.reshape(N, 2)
        s = _score(pts_local)
        return -s

    x0 = pts.flatten()
    try:
        res = minimize(neg_score, x0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                                'adaptive': True})
        pts_polished = res.x.reshape(N, 2)
        score_polished = _score(pts_polished)

        # Only accept if improved and valid
        if score_polished > _score(pts) and score_polished > 0:
            return pts_polished
        return pts
    except Exception:
        return pts


def construct():
    global _START_TIME
    _START_TIME = time.time()

    best_pts = None
    best_score = -1.0

    # Beta schedule: start smooth, end sharp
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]

    # Multiple initializations
    inits = []

    # Hex lattice
    inits.append(_hex_lattice_init(0.125))
    inits.append(_hex_lattice_init(0.12))

    # Ring configurations
    inits.append(_ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]))
    inits.append(_ring_init([1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.78, 0.95],
                            [0.0, np.pi / 6, np.pi / 3, np.pi / 6, 0.0]))

    # Random initializations
    inits.append(_random_init(42))
    inits.append(_random_init(123))
    inits.append(_random_init(7))

    # PBD parameters
    pbd_target_r = 0.125
    pbd_iters = 500

    for init_idx, init_pts in enumerate(inits):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        if len(init_pts) < N:
            continue

        # PBD pre-conditioning
        pts_pre = _pbd_relax(init_pts, target_r=pbd_target_r, n_iters=pbd_iters)
        
        # If PBD didn't improve much or took too long, skip to next
        score_pre = _score(pts_pre)
        
        # Multi-stage soft-min optimization
        pts_opt = _soft_min_optimize(pts_pre, beta_schedule, max_iter_per_stage=200)
        
        # Compute score
        score = _score(pts_opt)
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()

        # Check if we have time for polishing
        time_elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - time_elapsed

        if remaining > 10 and score > 0.12:
            # Polish the best result so far
            if init_idx == len(inits) - 1 or score > 0.125:
                pts_polished = _polish(pts_opt, budget_seconds=min(remaining - 5, 15.0))
                score_polished = _score(pts_polished)
                if score_polished > best_score:
                    best_score = score_polished
                    best_pts = pts_polished.copy()

        # Early exit if we're very close to the known best
        if best_score > 0.12582:
            break

    # Final polish on the best result if time permits
    time_elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - time_elapsed
    if remaining > 10 and best_pts is not None:
        pts_final = _polish(best_pts, budget_seconds=min(remaining - 5, 15.0))
        score_final = _score(pts_final)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final.copy()

    if best_pts is None:
        best_pts = _hex_lattice_init(0.125)

    return best_pts


if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
