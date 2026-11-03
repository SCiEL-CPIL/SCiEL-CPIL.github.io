"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n, beta=50.0, penalty=100.0):
    """
    Smooth objective: minimize -sum(r) + penalty for violations.
    v is a vector of length 3*n: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Uses smooth max via softplus for wall and pairwise violations.
    Vectorized pairwise computation.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Wall violations: r > min(x, 1-x, y, 1-y)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    # Pairwise violations: vectorized
    # Compute all pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    # viol[i,j] = r_i + r_j - dist[i,j]
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dist
    # Only upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    viol_upper = viol[iu]
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol_upper) / beta)

    return -np.sum(r) + penalty * (wall_pen + pair_pen)


def _smooth_grad(v, n, beta=50.0, penalty=100.0):
    """
    Gradient of the smooth objective using softplus derivatives.
    Vectorized pairwise computation.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = -np.ones(n)  # from -sum(r)

    # Wall terms
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    # Determine binding wall for each circle
    dists_x = np.stack([x, 1.0 - x], axis=0)  # (2, n)
    dists_y = np.stack([y, 1.0 - y], axis=0)  # (2, n)
    all_dists = np.stack([dists_x, dists_y], axis=1)  # (n, 4)
    min_idx = np.argmin(all_dists, axis=1)  # (n,)

    # For each circle, add gradient contribution based on binding wall
    for i in range(n):
        s = sig_wall[i] * penalty
        if s < 1e-12:
            continue
        idx = min_idx[i]
        grad_r[i] += s
        if idx == 0:  # x
            grad_x[i] -= s
        elif idx == 1:  # 1-x
            grad_x[i] += s
        elif idx == 2:  # y
            grad_y[i] -= s
        else:  # 1-y
            grad_y[i] += s

    # Pairwise terms (vectorized)
    dx = x[:, np.newaxis] - x[np.newaxis, :]  # (n, n)
    dy = y[:, np.newaxis] - y[np.newaxis, :]  # (n, n)
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq + 1e-30)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dist
    sig_pair = 1.0 / (1.0 + np.exp(-beta * viol))  # (n, n)

    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    sig_upper = sig_pair[iu]  # (n_pairs,)
    dx_upper = dx[iu]
    dy_upper = dy[iu]
    dist_upper = dist[iu]

    # d(viol)/dx_i = dx/dist, d(viol)/dx_j = -dx/dist
    # factor = penalty * sig
    factor = penalty * sig_upper

    # Accumulate gradients
    ii = iu[0]
    jj = iu[1]
    np.add.at(grad_x, ii, factor * dx_upper / dist_upper)
    np.add.at(grad_x, jj, -factor * dx_upper / dist_upper)
    np.add.at(grad_y, ii, factor * dy_upper / dist_upper)
    np.add.at(grad_y, jj, -factor * dy_upper / dist_upper)
    np.add.at(grad_r, ii, factor)
    np.add.at(grad_r, jj, factor)

    return np.concatenate([grad_x, grad_y, grad_r])


def _gradient_polish(xy, r, deadline, tol=1e-8, max_iter=500):
    """
    Gradient-based polish using L-BFGS-B on the smooth objective.
    Uses adaptive beta and penalty to improve convergence.
    More stages and higher iteration counts since objective/grad are now vectorized.
    """
    n = len(xy)

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
    for i in range(n):
        bounds.append((0.001, 0.999))  # y
    for i in range(n):
        bounds.append((0.001, 0.5))  # r

    # Adaptive beta/penalty strategy: more stages, tighter
    stages = [
        (8.0, 30.0, 300),
        (20.0, 80.0, 500),
        (50.0, 150.0, 800),
        (100.0, 300.0, 1200),
        (200.0, 500.0, 1500),
    ]
    
    current_xy = xy.copy()
    current_r = r.copy()
    best_total, _ = _eval_packing(current_xy, current_r)

    for beta, penalty, iters in stages:
        if time.time() > deadline - 5:
            break
            
        def obj(v, b=beta, p=penalty):
            if time.time() > deadline:
                return float('inf')
            return _smooth_objective(v, n, b, p)

        def grad(v, b=beta, p=penalty):
            if time.time() > deadline:
                return np.zeros(3 * n)
            return _smooth_grad(v, n, b, p)

        v0 = np.concatenate([current_xy[:, 0], current_xy[:, 1], current_r])
        try:
            result = minimize(obj, v0, 
                              jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': iters, 'ftol': tol, 'gtol': 1e-7,
                                       'maxfun': iters * 5})
        except Exception:
            continue

        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]

        # Validate
        total, valid = _eval_packing(new_xy, new_r)
        if valid and total > best_total + 1e-12:
            best_total = total
            current_xy = new_xy
            current_r = new_r

    return current_xy, current_r


def _exact_local_optimize(xy, r, deadline, max_sweeps=50):
    """
    Exact local optimization: for each circle in sequence, find the position
    that maximizes its radius given all other circles fixed.
    Uses scipy minimize for each circle's 2D position + radius.
    """
    n = len(xy)
    cur_xy = xy.copy()
    cur_r = r.copy()
    best_total, _ = _eval_packing(cur_xy, cur_r)
    best_xy = cur_xy.copy()
    best_r = cur_r.copy()

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 1:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = cur_xy[mask]
            others_r = cur_r[mask]
            
            # Objective: maximize r_i subject to
            # r_i <= min(x, 1-x, y, 1-y)
            # r_i <= dist((x,y), others_k) - r_k for all k
            # We minimize -r_i + penalty for violations
            
            def circle_obj(v, ii=i):
                xi, yi, ri = v
                # Wall clearance
                wall_c = min(xi, 1.0 - xi, yi, 1.0 - yi)
                if ri > wall_c + 1e-12:
                    return 1e6 * (ri - wall_c) - ri
                # Circle clearance
                for k in range(len(others_xy)):
                    d = np.sqrt((xi - others_xy[k,0])**2 + (yi - others_xy[k,1])**2)
                    if ri > d - others_r[k] + 1e-12:
                        return 1e6 * (ri - (d - others_r[k])) - ri
                return -ri
            
            # Start from current position with slightly larger radius
            v0 = np.array([cur_xy[i, 0], cur_xy[i, 1], cur_r[i] + 0.001])
            # Bounds
            bnd = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.49)]
            
            try:
                res = minimize(circle_obj, v0, method='L-BFGS-B', bounds=bnd,
                              options={'maxiter': 200, 'ftol': 1e-12})
            except Exception:
                continue
            
            new_x, new_y, new_r = res.x
            # Validate
            new_r = min(new_r, new_x, 1.0 - new_x, new_y, 1.0 - new_y)
            new_r = max(new_r, 0.001)
            # Check against others
            valid = True
            for k in range(len(others_xy)):
                d = np.sqrt((new_x - others_xy[k,0])**2 + (new_y - others_xy[k,1])**2)
                if new_r > d - others_r[k] - 1e-10:
                    new_r = d - others_r[k] - 1e-10
                    valid = False
            new_r = max(new_r, 0.001)
            
            if new_r > cur_r[i] + 1e-10:
                cur_xy[i] = [new_x, new_y]
                cur_r[i] = new_r
                improved = True
        
        # Evaluate after sweep
        total, valid = _eval_packing(cur_xy, cur_r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = cur_xy.copy()
            best_r = cur_r.copy()
        
        if not improved:
            break

    return best_xy, best_r


def _high_precision_polish(xy, r, deadline):
    """
    Final high-precision polish: tight tolerance, more iterations, high beta.
    Multiple restarts with perturbations to escape local minima.
    Plus exact local optimization pass.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, _ = _eval_packing(best_xy, best_r)

    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))

    # Try multiple high-beta settings with perturbations
    configs = [
        (150.0, 400.0, 0.0),
        (250.0, 600.0, 0.0005),
        (350.0, 800.0, 0.001),
    ]

    rng = np.random.RandomState(42)

    for beta, penalty, pert_scale in configs:
        if time.time() > deadline - 10:
            break

        # Create perturbed starting point
        if pert_scale > 0:
            v0 = np.concatenate([best_xy[:, 0], best_xy[:, 1], best_r])
            v0 = v0 + rng.uniform(-pert_scale, pert_scale, size=len(v0))
            for i in range(n):
                v0[i] = np.clip(v0[i], 0.0001, 0.9999)
                v0[n+i] = np.clip(v0[n+i], 0.0001, 0.9999)
                v0[2*n+i] = np.clip(v0[2*n+i], 0.0001, 0.4999)
        else:
            v0 = np.concatenate([best_xy[:, 0], best_xy[:, 1], best_r])

        def obj(v, b=beta, p=penalty):
            if time.time() > deadline:
                return float('inf')
            return _smooth_objective(v, n, b, p)

        def grad(v, b=beta, p=penalty):
            if time.time() > deadline:
                return np.zeros(3 * n)
            return _smooth_grad(v, n, b, p)

        try:
            result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10,
                                       'maxfun': 100000})
        except Exception:
            continue

        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]

        total, valid = _eval_packing(new_xy, new_r)
        if valid and total > best_total:
            best_total = total
            best_xy = new_xy
            best_r = new_r

    # Exact local optimization pass
    if time.time() < deadline - 5:
        ex_xy, ex_r = _exact_local_optimize(best_xy, best_r, deadline, max_sweeps=30)
        total, valid = _eval_packing(ex_xy, ex_r)
        if valid and total > best_total:
            best_total = total
            best_xy = ex_xy
            best_r = ex_r

    return best_xy, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + local search
    for seed in range(8):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Gradient polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 30:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

        # Phase 4: Second local search pass from polished result
        if best_valid and best_xy is not None and time.time() < deadline - 20:
            ls2_xy, ls2_r, ls2_total, ls2_valid = _best_fit_local_search(
                best_xy.copy(), best_r.copy(), deadline, grid_size=100)
            if ls2_valid and ls2_total > best_total + 1e-10:
                best_total = ls2_total
                best_xy = ls2_xy
                best_r = ls2_r

        # Phase 5: High-precision polish (includes exact local opt)
        if best_valid and best_xy is not None and time.time() < deadline - 3:
            hp_xy, hp_r = _high_precision_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(hp_xy, hp_r)
            if valid and total > best_total:
                best_total = total
                best_xy = hp_xy
                best_r = hp_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
