"""Packing of 50 equal circles in a unit square via force-based repulsion + accelerated SA."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
HALF = 0.5


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _violation_obj(pts_flat, r_target):
    """
    Smooth violation surrogate: sum of squared violations for a given target radius.
    Violation for pair (i,j): max(0, 2*r_target - dist_ij)^2
    Violation for wall i,k: max(0, r_target - dist_to_wall)^2
    This is differentiable and suitable for scipy minimize.
    """
    pts = pts_flat.reshape(N, 2)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = np.maximum(0.0, 2.0 * r_target - pair_dist)
    pair_cost = np.sum(pair_viol ** 2)
    # Wall distances
    wall_dist = np.minimum(
        np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1])
    )
    wall_viol = np.maximum(0.0, r_target - wall_dist)
    wall_cost = np.sum(wall_viol ** 2)
    return pair_cost + wall_cost


def _violation_grad(pts_flat, r_target):
    """
    Analytic gradient of the violation surrogate.
    """
    pts = pts_flat.reshape(N, 2)
    grad = np.zeros_like(pts)
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e15)
    # For each pair where dist < 2*r_target:
    # d/dx_i [max(0, 2r - d)^2] = -2*(2r-d) * (x_i - x_j)/d
    overlap = np.maximum(0.0, 2.0 * r_target - dist)
    inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
    # Force on i from j: -2*overlap * diff_ij / dist
    pair_force = -2.0 * overlap * inv_dist
    pair_force = np.where(np.isfinite(pair_force), pair_force, 0.0)
    # Sum over j for each i
    grad += np.sum(pair_force[:, :, None] * diff, axis=1)
    # Wall gradients
    for dim in range(2):
        # Left wall: x < -HALF + r
        left_viol = np.maximum(0.0, r_target - (HALF + pts[:, dim]))
        # Right wall: x > HALF - r
        right_viol = np.maximum(0.0, r_target - (HALF - pts[:, dim]))
        # Bottom wall: y < -HALF + r
        bot_viol = np.maximum(0.0, r_target - (HALF + pts[:, 1]))
        # Top wall: y > HALF - r
        top_viol = np.maximum(0.0, r_target - (HALF - pts[:, 1]))
        # Gradient w.r.t. x_i:
        # d/dx [max(0, r - (HALF+x))^2] = -2*(r-(HALF+x)) if x < -HALF+r
        # d/dx [max(0, r - (HALF-x))^2] = +2*(r-(HALF-x)) if x > HALF-r
        grad[:, 0] += -2.0 * np.maximum(0.0, r_target - (HALF + pts[:, 0]))
        grad[:, 0] += 2.0 * np.maximum(0.0, r_target - (HALF - pts[:, 0]))
        grad[:, 1] += -2.0 * np.maximum(0.0, r_target - (HALF + pts[:, 1]))
        grad[:, 1] += 2.0 * np.maximum(0.0, r_target - (HALF - pts[:, 1]))
    return grad.flatten()


def _scipy_polish(pts0, r_target, deadline, max_iter=5000):
    """
    Use scipy L-BFGS-B to minimize violation for a given target radius.
    Returns improved points.
    """
    x0 = pts0.flatten()
    res = minimize(
        _violation_obj, x0,
        jac=_violation_grad,
        args=(r_target,),
        method='L-BFGS-B',
        bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
    )
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts


def _compute_forces(pts, r_target=None):
    """
    Compute repulsion forces on each circle.
    If r_target is given, use a hard constraint model: circles closer than
    r_target*2 get strong repulsion. Otherwise use inverse-distance repulsion.
    Returns array of shape (N, 2) with force on each circle.
    """
    forces = np.zeros((N, 2))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e15)

    if r_target is not None:
        # Hard constraint: repel if dist < 2*r_target
        min_dist = 2.0 * r_target
        overlap = np.maximum(0.0, min_dist - dist)
        # Force proportional to overlap, direction = normalized diff
        inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
        force_mag = overlap * inv_dist
        force_mag[~np.isfinite(force_mag)] = 0.0
        forces = np.sum(force_mag[:, :, None] * diff, axis=1)
    else:
        # Inverse-distance repulsion (soft)
        inv_dist = np.where(dist > 1e-10, 1.0 / (dist ** 2 + 1e-8), 0.0)
        force_mag = inv_dist
        force_mag[~np.isfinite(force_mag)] = 0.0
        forces = np.sum(force_mag[:, :, None] * diff, axis=1)

    # Wall repulsion
    for i in range(N):
        for dim in range(2):
            if pts[i, dim] < -HALF + r_target if r_target else -HALF + 0.02:
                forces[i, dim] += 50.0 * ((-HALF + (r_target if r_target else 0.02)) - pts[i, dim])
            if pts[i, dim] > HALF - r_target if r_target else HALF - 0.02:
                forces[i, dim] -= 50.0 * (pts[i, dim] - (HALF - (r_target if r_target else 0.02)))

    return forces


def _force_optimize(pts0, r_target=None, n_steps=5000, lr=0.001, deadline=None):
    """
    Force-based optimization: iteratively move circles along repulsion forces.
    Much faster than L-BFGS with finite-difference gradients.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_r = _compute_r(pts)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = _compute_forces(pts, r_target)
        # Normalize forces to avoid exploding steps
        fmag = np.linalg.norm(forces, axis=1, keepdims=True)
        fmag = np.maximum(fmag, 1e-10)
        forces = forces / fmag

        pts = pts + lr * forces
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)

        if step % 50 == 0:
            r = _compute_r(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()

    r = _compute_r(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()

    return best_pts, best_r


def _compute_r_fast(pts, prev_dist=None, moved_idx=None):
    """
    Compute radius. If prev_dist and moved_idx are given, only recompute
    distances involving moved circles for speed.
    """
    if prev_dist is None or moved_idx is None:
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
    else:
        dist = prev_dist.copy()
        for idx in moved_idx:
            diff = pts[idx:idx+1, :] - pts
            dist[idx, :] = np.sqrt(np.sum(diff ** 2, axis=1))
            diff = pts - pts[idx:idx+1, :]
            dist[:, idx] = np.sqrt(np.sum(diff ** 2, axis=1))
            dist[idx, idx] = np.inf

    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    r = min(r, float(np.min(dist)) / 2.0)
    return r, dist


def _accelerated_sa(pts0, rng, deadline, t_start=0.1, t_end=0.0001, n_iter=None):
    """
    Accelerated simulated annealing with incremental distance updates.
    """
    pts = pts0.copy()
    current_r, dist_mat = _compute_r_fast(pts)
    best_pts = pts.copy()
    best_r = current_r

    if n_iter is None:
        n_iter = 100000

    for it in range(n_iter):
        if time.time() > deadline:
            break

        progress = it / n_iter
        t = t_start * (t_end / t_start) ** progress

        # Perturb 1-3 random circles for more exploration
        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)

        step_size = 0.015 * (1.0 - progress * 0.7)
        new_pts = pts.copy()
        for idx in idxs:
            direction = rng.normal(0, 1, size=2)
            direction /= (np.linalg.norm(direction) + 1e-15)
            new_pts[idx] += direction * step_size

        new_pts = np.clip(new_pts, -HALF + 1e-9, HALF - 1e-9)
        new_r, new_dist = _compute_r_fast(new_pts, dist_mat, idxs)

        delta = new_r - current_r
        if delta > 0:
            pts = new_pts
            dist_mat = new_dist
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            if t > 1e-15:
                prob = np.exp(delta / t)
                if rng.random() < prob:
                    pts = new_pts
                    dist_mat = new_dist
                    current_r = new_r

    return best_pts, best_r


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice, jittered."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF + 1e-9, HALF - 1e-9)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _binary_search_radius(pts0, rng, deadline, n_iter=10):
    """
    Binary search on radius: for each candidate radius, run force optimization
    with that as a hard constraint, then check if the achieved radius meets it.
    """
    best_pts = pts0.copy()
    best_r = _compute_r(pts0)

    r_low = best_r
    r_high = min(best_r + 0.015, 0.12)

    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        r_mid = (r_low + r_high) / 2.0

        # Force-optimize with r_mid as target
        pts, r_achieved = _force_optimize(
            best_pts, r_target=r_mid,
            n_steps=3000, lr=0.0005, deadline=deadline
        )

        # Also try SA polish
        if time.time() < deadline - 5:
            pts, r_achieved = _accelerated_sa(
                pts, rng, deadline,
                t_start=0.05, t_end=0.0001, n_iter=20000
            )

        r_achieved = _compute_r(pts)
        if r_achieved > best_r:
            best_r = r_achieved
            best_pts = pts.copy()

        if r_achieved >= r_mid - 1e-9:
            r_low = r_mid
        else:
            r_high = r_mid

    return best_pts, best_r


def _final_polish(pts0, rng, deadline):
    """
    Final high-precision polish:
    1. Estimate current radius
    2. Try scipy polish at slightly higher radius targets
    3. SA at very low temperature
    4. Force polish at very small lr
    """
    best_pts = pts0.copy()
    best_r = _compute_r(pts0)

    # Try scipy polish at current radius (to resolve any small overlaps)
    if time.time() < deadline - 10:
        r_cur = best_r
        pts = _scipy_polish(best_pts, r_cur * 0.999, deadline, max_iter=2000)
        r_new = _compute_r(pts)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts.copy()

    # Try scipy polish at slightly higher radius
    for r_try in [best_r * 1.001, best_r * 1.003, best_r * 1.005]:
        if time.time() > deadline - 8:
            break
        if r_try > 0.12:
            break
        pts = _scipy_polish(best_pts, r_try, deadline, max_iter=1500)
        r_new = _compute_r(pts)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts.copy()

    # Very low temperature SA
    if time.time() < deadline - 5:
        pts, r = _accelerated_sa(
            best_pts, rng, deadline,
            t_start=0.01, t_end=0.00001, n_iter=30000
        )
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Final force polish
    if time.time() < deadline - 2:
        pts, r = _force_optimize(best_pts, n_steps=2000, lr=0.00005, deadline=deadline)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 145.0
    rng = np.random.default_rng(12345)

    best_pts = None
    best_r = 0.0

    def try_update(pts):
        nonlocal best_pts, best_r
        r = _compute_r(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 1: Multi-start with force optimization (fast, no gradient overhead)
    inits = [
        _hex_init(rng, jitter=0.005),
        _hex_init(rng, jitter=0.02),
        _hex_init(rng, jitter=0.05),
        _grid_init(rng, jitter=0.01),
        _grid_init(rng, jitter=0.04),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
    ]

    for i, init in enumerate(inits):
        if time.time() > deadline - 40.0:
            break
        pts, r = _force_optimize(init, n_steps=5000, lr=0.002, deadline=deadline)
        try_update(pts)

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Phase 2: Accelerated SA from best point (faster due to incremental updates)
    if time.time() < deadline - 50.0:
        sa_pts, sa_r = _accelerated_sa(
            best_pts, rng, deadline,
            t_start=0.08, t_end=0.0001, n_iter=120000
        )
        try_update(sa_pts)

    # Phase 3: Force polish with decreasing learning rate
    if time.time() < deadline - 30.0:
        for lr in [0.001, 0.0005, 0.0002]:
            if time.time() > deadline - 15.0:
                break
            pts, r = _force_optimize(best_pts, n_steps=4000, lr=lr, deadline=deadline)
            try_update(pts)

    # Phase 4: Binary search on radius with force + SA
    if time.time() < deadline - 30.0:
        bs_pts, bs_r = _binary_search_radius(best_pts, rng, deadline, n_iter=10)
        try_update(bs_pts)

    # Phase 5: Scipy-based polish (new: smooth violation minimization)
    if time.time() < deadline - 15.0:
        r_cur = best_r
        pts = _scipy_polish(best_pts, r_cur, deadline, max_iter=3000)
        r_new = _compute_r(pts)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts.copy()
        # Try at slightly higher target
        for r_try in [best_r * 1.002, best_r * 1.005]:
            if time.time() > deadline - 10:
                break
            if r_try > 0.12:
                break
            pts = _scipy_polish(best_pts, r_try, deadline, max_iter=2000)
            r_new = _compute_r(pts)
            if r_new > best_r:
                best_r = r_new
                best_pts = pts.copy()

    # Phase 6: Final high-precision polish
    if time.time() < deadline - 5.0:
        final_pts, final_r = _final_polish(best_pts, rng, deadline)
        if final_r > best_r:
            best_r = final_r
            best_pts = final_pts.copy()

    return best_pts
