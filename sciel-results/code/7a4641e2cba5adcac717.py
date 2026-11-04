"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def _eval_packing(xy, r):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side):
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


def _build_contact_graph(xy):
    """Return list of (i, j) pairs from Delaunay triangulation (edges)."""
    try:
        tri = Delaunay(xy)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                i, j = simplex[a], simplex[(a + 1) % 3]
                if i > j:
                    i, j = j, i
                edges.add((i, j))
        return list(edges)
    except Exception:
        # Fallback: connect all pairs within some distance
        edges = []
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                d = np.linalg.norm(xy[i] - xy[j])
                if d < 0.25:
                    edges.append((i, j))
        return edges


def _force_directed_packing(xy, r, deadline, max_iters=20000):
    """
    Force-directed optimization:
    - Repulsive force pushes apart overlapping circles (along center line).
    - Wall forces push circles back inside.
    - 'Squeeze' force: for contact pairs (from Delaunay), if circles are
      nearly touching (dist ~ r_i + r_j), apply a small attractive force
      to pull them together, freeing space for radius growth.
    - Radius growth: each step, try to increase each radius slightly if
      it doesn't cause overlap (checked against nearest neighbors).
    """
    n = len(xy)
    cur_xy = xy.astype(float).copy()
    cur_r = r.astype(float).copy()

    best_xy = cur_xy.copy()
    best_r = cur_r.copy()
    best_total, best_valid = _eval_packing(cur_xy, cur_r)
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Precompute: we'll rebuild contact graph every K iterations
    rebuild_interval = 20
    contact_edges = _build_contact_graph(cur_xy)

    # Velocity for momentum
    vel = np.zeros_like(cur_xy)
    vel_r = np.zeros_like(cur_r)

    lr_pos = 0.003
    lr_r = 0.001
    damping = 0.85
    squeeze_strength = 0.3
    repulse_strength = 1.0
    wall_strength = 1.0

    it = 0
    no_improve = 0
    max_no_improve = 3000

    while it < max_iters:
        if time.time() > deadline:
            break

        # Rebuild contact graph periodically
        if it % rebuild_interval == 0:
            contact_edges = _build_contact_graph(cur_xy)

        # Build force vectors
        force_xy = np.zeros_like(cur_xy)
        force_r = np.zeros_like(cur_r)

        # 1) Repulsive forces from overlaps
        for (i, j) in contact_edges:
            dx = cur_xy[i, 0] - cur_xy[j, 0]
            dy = cur_xy[i, 1] - cur_xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dx, dy = 1e-6, 1e-6
                dist = np.sqrt(dx * dx + dy * dy)
            min_dist = cur_r[i] + cur_r[j]

            if dist < min_dist:
                # Overlap: repel
                overlap = min_dist - dist
                fx = repulse_strength * overlap * dx / dist
                fy = repulse_strength * overlap * dy / dist
                force_xy[i, 0] += fx
                force_xy[i, 1] += fy
                force_xy[j, 0] -= fx
                force_xy[j, 1] -= fy
                # Slightly shrink both
                force_r[i] -= 0.1 * overlap
                force_r[j] -= 0.1 * overlap
            elif dist < min_dist * 1.05:
                # Near-contact: squeeze (attract) to free space
                gap = dist - min_dist
                fx = squeeze_strength * gap * dx / dist
                fy = squeeze_strength * gap * dy / dist
                # Attract: i moves toward j, j moves toward i
                force_xy[i, 0] -= fx
                force_xy[i, 1] -= fy
                force_xy[j, 0] += fx
                force_xy[j, 1] += fy

        # 2) Wall forces
        for i in range(n):
            x, y, ri = cur_xy[i, 0], cur_xy[i, 1], cur_r[i]
            if x < ri:
                force_xy[i, 0] += wall_strength * (ri - x)
                force_r[i] -= 0.5 * (ri - x)
            if 1.0 - x < ri:
                force_xy[i, 0] -= wall_strength * (ri - (1.0 - x))
                force_r[i] -= 0.5 * (ri - (1.0 - x))
            if y < ri:
                force_xy[i, 1] += wall_strength * (ri - y)
                force_r[i] -= 0.5 * (ri - y)
            if 1.0 - y < ri:
                force_xy[i, 1] -= wall_strength * (ri - (1.0 - y))

        # 3) Radius growth: try to grow each circle
        for i in range(n):
            # Check max possible radius given walls
            wall_max = min(cur_xy[i, 0], 1.0 - cur_xy[i, 0],
                           cur_xy[i, 1], 1.0 - cur_xy[i, 1])
            # Check max possible radius given other circles (nearest few)
            circle_max = wall_max
            for j in range(n):
                if i == j:
                    continue
                dx = cur_xy[i, 0] - cur_xy[j, 0]
                dy = cur_xy[i, 1] - cur_xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 1e-12:
                    avail = dist - cur_r[j]
                    if avail < circle_max:
                        circle_max = avail
            # Grow toward the max
            target_r = max(0.005, circle_max)
            if target_r > cur_r[i]:
                force_r[i] += 0.3 * (target_r - cur_r[i])

        # 4) Update positions and radii
        vel = damping * vel + lr_pos * force_xy
        cur_xy += vel
        vel_r = damping * vel_r + lr_r * force_r
        cur_r += vel_r

        # Clamp
        cur_xy[:, 0] = np.clip(cur_xy[:, 0], 0.001, 0.999)
        cur_xy[:, 1] = np.clip(cur_xy[:, 1], 0.001, 0.999)
        cur_r = np.clip(cur_r, 0.005, 0.49)

        # Ensure radii don't exceed wall distance
        for i in range(n):
            wm = min(cur_xy[i, 0], 1.0 - cur_xy[i, 0],
                     cur_xy[i, 1], 1.0 - cur_xy[i, 1])
            if cur_r[i] > wm:
                cur_r[i] = wm

        # Evaluate
        total, valid = _eval_packing(cur_xy, cur_r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = cur_xy.copy()
            best_r = cur_r.copy()
            no_improve = 0
        else:
            no_improve += 1

        # Adaptive learning rate: reduce if stuck
        if no_improve > max_no_improve:
            lr_pos *= 0.9
            lr_r *= 0.9
            squeeze_strength *= 0.95
            no_improve = 0
            if lr_pos < 1e-5:
                break

        it += 1

    return best_xy, best_r, best_total, best_valid


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """Grid-based best-fit local search (kept as a refinement tool)."""
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n, beta=50.0):
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.logaddexp(0.0, beta * viol) / beta

    return -np.sum(r) + 100.0 * (wall_pen + total_viol)


def _smooth_grad(v, n, beta=50.0):
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        if min_idx == 0:
            grad[2*n + i] += 100.0 * s
            grad[i] -= 100.0 * s
        elif min_idx == 1:
            grad[2*n + i] += 100.0 * s
            grad[i] += 100.0 * s
        elif min_idx == 2:
            grad[2*n + i] += 100.0 * s
            grad[n + i] -= 100.0 * s
        else:
            grad[2*n + i] += 100.0 * s
            grad[n + i] += 100.0 * s

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-10:
                continue
            factor = 100.0 * s
            grad[i] += factor * dx / dist
            grad[n + i] += factor * dy / dist
            grad[j] -= factor * dx / dist
            grad[n + j] -= factor * dy / dist
            grad[2*n + i] += factor
            grad[2*n + j] += factor

    grad[2*n:3*n] -= 1.0
    return grad


def _gradient_polish(xy, r, deadline):
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))

    beta = 100.0

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, beta)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n, beta)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 20000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r
    return xy, r


def _high_precision_polish(xy, r, deadline):
    """
    Multi-stage gradient polish with increasing beta for tighter constraints.
    """
    n = len(xy)
    cur_xy = xy.copy()
    cur_r = r.copy()

    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline:
            break
        v0 = np.concatenate([cur_xy[:, 0], cur_xy[:, 1], cur_r])

        bounds = []
        for _ in range(n):
            bounds.append((0.001, 0.999))
        for _ in range(n):
            bounds.append((0.001, 0.999))
        for _ in range(n):
            bounds.append((0.001, 0.5))

        def obj(v):
            if time.time() > deadline:
                return float('inf')
            return _smooth_objective(v, n, beta)

        def grad(v):
            if time.time() > deadline:
                return np.zeros(3 * n)
            return _smooth_grad(v, n, beta)

        try:
            result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                                       'maxfun': 10000})
        except Exception:
            continue

        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]

        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            cur_xy = new_xy
            cur_r = new_r

    return cur_xy, cur_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Multiple grid inits + force-directed optimization
    for side in [5, 6, 7]:
        if time.time() > deadline - 10:
            break
        xy = _grid_init(side)
        r_init = xy[:, 2].copy()
        xy_init = xy[:, :2].copy()

        out_xy, out_r, total, valid = _force_directed_packing(
            xy_init, r_init, deadline - 10)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random inits + force-directed
    rng = np.random.RandomState(0)
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 15:
            break
        xy_rand = rng.uniform(0.1, 0.9, size=(N, 2))
        r_rand = rng.uniform(0.03, 0.06, size=N)

        out_xy, out_r, total, valid = _force_directed_packing(
            xy_rand, r_rand, deadline - 15)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Grid local search refinement
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(
            best_xy.copy(), best_r.copy(), deadline - 10, grid_size=100)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r

    # Phase 4: Force-directed again on the refined solution
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        fd_xy, fd_r, fd_total, fd_valid = _force_directed_packing(
            best_xy.copy(), best_r.copy(), deadline - 8)
        if fd_valid and fd_total > best_total + 1e-10:
            best_total = fd_total
            best_xy = fd_xy
            best_r = fd_r

    # Phase 5: High-precision gradient polish
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        polished_xy, polished_r = _high_precision_polish(best_xy, best_r, deadline)
        total, valid = _eval_packing(polished_xy, polished_r)
        if valid and total > best_total:
            best_total = total
            best_xy = polished_xy
            best_r = polished_r

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
