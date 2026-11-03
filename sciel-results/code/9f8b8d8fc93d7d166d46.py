"""
Pack 200 equal circles in a unit square.

Strategy:
1. Generate a near-optimal hexagonal lattice packing as the primary initial
   configuration (spacing tuned to fit ~200 circles in the unit square).
2. Run a fast force-directed repulsion phase to remove any overlaps and push
   circles apart while respecting the boundary.
3. Use a BFGS quasi-Newton method on a soft-min objective (with increasing
   sharpness beta) to maximize the minimum distance to walls and to other
   circles. BFGS exploits curvature information from the Hessian to take
   more efficient steps than pure gradient methods, which is crucial for
   high-dimensional (400-dim) problems where L-BFGS-B can stall.
4. Multi-start: try several different hex lattice spacings, phase shifts,
   and small random perturbations to escape local minima.
5. Final high-precision polish with very high beta.
6. Track the best radius found and return the best configuration.

The soft-min objective:
    min_i c_i  ≈  -(1/β) log(Σ_i exp(-β c_i))
where c_i are the constraint values (wall distances and half-pairwise distances).
We maximize this, i.e., minimize the negative.

BFGS is chosen over L-BFGS-B because:
- BFGS maintains a full inverse Hessian approximation, giving better curvature
  information in the early-to-mid iterations.
- For 400 variables, the 400x400 matrix is manageable in memory.
- BFGS can take larger, more informed steps when the landscape has consistent
  curvature, which is the case near a good packing.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 148  # safety margin before 150s kill

def _get_radius(pts):
    """
    Compute the maximum common radius for a set of centers:
    min of (distance to walls) and (half of minimum pairwise distance).
    """
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)

    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0

    return min(min_wall, min_pair)

def _objective_and_grad(x, beta):
    """
    Objective for minimization: -soft_min(constraints).
    Constraints: 4 wall distances per point, plus pairwise distances / 2.
    Returns (value, gradient) with respect to x (flat 2N vector).
    """
    pts = x.reshape((N, 2))

    # Wall constraint values
    C_wall = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])

    # Pair constraint values
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    pair_dists = np.sqrt(dx**2 + dy**2)
    C_pair = pair_dists / 2.0

    C = np.concatenate([C_wall, C_pair])
    n_c = len(C)

    # Soft min: -(1/beta) * log(sum(exp(-beta * C)))
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val

    # Weights for each constraint
    weights = np.exp(shifted) / sum_exp

    # Gradient with respect to pts
    grad_pts = np.zeros((N, 2))

    # Wall gradients: d(C_wall)/d(pts)
    # C_wall[0:N]     = pts[:,0] + 0.5   -> d/d(pts[:,0]) = 1
    # C_wall[N:2N]    = 0.5 - pts[:,0]   -> d/d(pts[:,0]) = -1
    # C_wall[2N:3N]   = pts[:,1] + 0.5   -> d/d(pts[:,1]) = 1
    # C_wall[3N:4N]   = 0.5 - pts[:,1]   -> d/d(pts[:,1]) = -1
    w_wall = weights[:4 * N]
    grad_pts[:, 0] += w_wall[0:N] - w_wall[N:2 * N]
    grad_pts[:, 1] += w_wall[2 * N:3 * N] - w_wall[3 * N:4 * N]

    # Pair gradients: d(C_pair)/d(pts)
    # C_pair[k] = 0.5 * dist(pts[i_k], pts[j_k])
    # d(C_pair[k])/d(pts[i_k]) = 0.5 * (pts[i_k] - pts[j_k]) / dist
    # d(C_pair[k])/d(pts[j_k]) = -0.5 * (pts[i_k] - pts[j_k]) / dist
    w_pair = weights[4 * N:]
    valid = pair_dists > 1e-12
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    d_safe = np.where(valid, pair_dists, 1.0)

    gx_i = np.where(valid, 0.5 * dx_safe / d_safe, 0.0)
    gy_i = np.where(valid, 0.5 * dy_safe / d_safe, 0.0)

    np.add.at(grad_pts[:, 0], i_idx, gx_i * w_pair)
    np.add.at(grad_pts[:, 0], j_idx, -gx_i * w_pair)
    np.add.at(grad_pts[:, 1], i_idx, gy_i * w_pair)
    np.add.at(grad_pts[:, 1], j_idx, -gy_i * w_pair)

    # Total gradient: d(obj)/d(x) = -d(soft_min)/d(x) = -(1/beta) * d(sum_log)/d(x)
    # But soft_min = -(1/beta)*log(sum exp(-beta C)), so d(soft_min)/d(x) = -(1/beta)*d(log(...))/d(x)
    # d(log(sum exp(-beta C)))/d(x) = sum_k w_k * d(-beta C_k)/d(x) = -beta * sum_k w_k * d(C_k)/d(x)
    # So d(soft_min)/d(x) = (1/beta) * sum_k w_k * d(C_k)/d(x)
    # obj = -soft_min, so d(obj)/d(x) = -(1/beta) * sum_k w_k * d(C_k)/d(x) = -grad_pts / beta
    grad = (-grad_pts / beta).flatten()

    return obj_val, grad

def _generate_hex_lattice(spacing, phase_x=0.0, phase_y=0.0):
    """
    Generate a hexagonal lattice of points inside [-0.5, 0.5]^2.
    Return exactly N points (pad with duplicates of nearest if too few,
    or trim to N if too many).
    """
    points = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5 + row_h / 2 + phase_y
    row_idx = 0
    while y < 0.5:
        shift = (spacing / 2.0) if (row_idx % 2 == 1) else 0.0
        x = -0.5 + spacing / 2 + shift + phase_x
        while x < 0.5:
            if -0.499 <= x <= 0.499 and -0.499 <= y <= 0.499:
                points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)
    if len(pts) == 0:
        return np.random.uniform(-0.4, 0.4, (N, 2))

    if len(pts) > N:
        # Select N points closest to the center
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts = pts[indices[:N]]
    elif len(pts) < N:
        # Pad by duplicating the last point (will be fixed by force-directed phase)
        extra = np.random.uniform(-0.4, 0.4, (N - len(pts), 2))
        pts = np.vstack([pts, extra])

    return pts

def _force_directed_relax(pts, time_limit, step0=0.005, decay=0.999):
    """
    Fast force-directed phase: repulsive forces between close pairs + boundary
    repulsion. Vectorized, no Python loops over pairs.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    step = step0

    while True:
        elapsed = time.time() - t_start
        if elapsed > time_limit:
            break

        forces = np.zeros_like(pts)

        # Pairwise repulsion (inverse square, only for close pairs)
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)

        threshold = 0.15
        close_mask = (D < threshold) & (D > 1e-12)

        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        inv_d2 = np.where(close_mask, 1.0 / (D**2 + 1e-9), 0.0)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)

        # Boundary repulsion
        margin = 0.04
        forces[:, 0] += np.where(pts[:, 0] < -0.5 + margin, (margin - (pts[:, 0] + 0.5)) * 5.0, 0.0)
        forces[:, 0] -= np.where(pts[:, 0] > 0.5 - margin, (pts[:, 0] - (0.5 - margin)) * 5.0, 0.0)
        forces[:, 1] += np.where(pts[:, 1] < -0.5 + margin, (margin - (pts[:, 1] + 0.5)) * 5.0, 0.0)
        forces[:, 1] -= np.where(pts[:, 1] > 0.5 - margin, (pts[:, 1] - (0.5 - margin)) * 5.0, 0.0)

        pts = pts + step * forces
        pts = np.clip(pts, -0.5, 0.5)

        step *= decay

        r = _get_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_r, best_pts

def _bfgs_optimize(pts, t_start, time_budget, beta_sequence=None):
    """
    Run BFGS quasi-Newton optimization with increasing beta.
    """
    if beta_sequence is None:
        beta_sequence = [30, 80, 200, 500, 1200, 3000, 8000]

    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    best_r = _get_radius(pts)
    best_x = x0.copy()

    for beta in beta_sequence:
        elapsed = time.time() - t_start
        if elapsed > time_budget - 2.0:
            break

        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g

        try:
            maxiter = 80 if beta < 200 else 150
            res = minimize(
                fun_and_grad,
                x0,
                jac=True,
                method='BFGS',
                options={'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)

            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()

        except Exception:
            pass

    return best_r, best_x.reshape((N, 2))

def _perturb_and_retry(pts, t_start, time_budget, rng):
    """
    Perturb the most constrained points, then re-optimize.
    """
    r_curr = _get_radius(pts)

    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_d = np.min(D) / 2.0

    # Find points involved in the closest pairs
    closest_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            closest_pairs.append((D[i, j], i, j))
    closest_pairs.sort()

    # Perturb points in the 15 closest pairs
    involved = set()
    for d, i, j in closest_pairs[:15]:
        involved.add(i)
        involved.add(j)

    pts_new = pts.copy()
    for idx in list(involved)[:10]:
        pts_new[idx] += rng.normal(0, 0.004, 2)
    pts_new = np.clip(pts_new, -0.5, 0.5)

    # Quick force-directed fix
    r_fd, pts_fd = _force_directed_relax(pts_new, time_limit=min(1.5, time_budget * 0.15))

    # BFGS optimize
    remaining = time_budget - (time.time() - t_start)
    if remaining > 2:
        r_opt, pts_opt = _bfgs_optimize(pts_fd, t_start, remaining)
        if r_opt > r_curr:
            return r_opt, pts_opt

    if r_fd > r_curr:
        return r_fd, pts_fd

    return r_curr, pts

def construct():
    t0 = time.time()
    np.random.seed(123)
    rng = np.random.default_rng(123)

    best_r = 0.0
    best_pts = None

    # Primary hex lattice with tuned spacing
    # For 200 circles in a unit square, hex spacing ~ 1/sqrt(200/sqrt(3)) ≈ 0.074
    # Let's use spacing that gives ~200 circles
    spacings = [0.0730, 0.0725, 0.0720, 0.0715, 0.0710]
    phase_shifts = [0.0, 0.005, -0.005, 0.010, -0.010]

    # Build initial configurations
    initial_configs = []
    for s, px in zip(spacings, phase_shifts):
        pts = _generate_hex_lattice(s, phase_x=px)
        initial_configs.append(pts)

    # Add jittered variants
    for i in range(3):
        base = _generate_hex_lattice(0.0725)
        jittered = base + rng.uniform(-0.003, 0.003, size=base.shape)
        jittered = np.clip(jittered, -0.499, 0.499)
        initial_configs.append(jittered)

    # Random start
    initial_configs.append(rng.uniform(-0.4, 0.4, (N, 2)))

    # Process each initial config with time allocation
    n_configs = len(initial_configs)
    time_per = (_TIME_LIMIT - 15) / n_configs

    for idx, pts_init in enumerate(initial_configs):
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining < time_per * 0.4:
            break

        # Phase 1: Force-directed relaxation
        fd_time = min(2.5, time_per * 0.25)
        r_fd, pts_fd = _force_directed_relax(pts_init, time_limit=fd_time)

        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd

        # Phase 2: BFGS optimization with increasing beta
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        opt_time = min(remaining * 0.55, time_per * 0.55)

        if opt_time > 2.0:
            r_opt, pts_opt = _bfgs_optimize(pts_fd, t0, opt_time)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt

        # Phase 3: Perturbation and retry (only if time allows)
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining > time_per * 0.3 and best_pts is not None:
            r_pert, pts_pert = _perturb_and_retry(best_pts, t0, remaining * 0.6, rng)
            if r_pert > best_r:
                best_r = r_pert
                best_pts = pts_pert

    # Final high-precision polish on the best configuration
    elapsed = time.time() - t0
    remaining = _TIME_LIMIT - 3 - elapsed
    if remaining > 5 and best_pts is not None:
        r_final, pts_final = _bfgs_optimize(
            best_pts, t0, remaining,
            beta_sequence=[2000, 5000, 10000, 20000]
        )
        if r_final > best_r:
            best_r = r_final
            best_pts = pts_final

    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))

    # Ensure all points are strictly inside the square
    best_pts = np.clip(best_pts, -0.499999, 0.499999)

    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
