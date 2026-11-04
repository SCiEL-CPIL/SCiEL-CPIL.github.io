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


def _largest_empty_circle(xy, r, n_grid=200):
    """
    Find the center and radius of the largest empty circle in the unit square
    that does not overlap with existing circles and stays within bounds.
    Uses a dense grid scan + local refinement.
    """
    gx = np.linspace(0.0, 1.0, n_grid)
    gy = np.linspace(0.0, 1.0, n_grid)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    n = len(xy)
    if n > 0:
        # Vectorized: for each candidate, compute min distance to all circles minus their radius
        # Process in chunks to limit memory
        chunk = 10000
        best_r = np.zeros(M)
        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            cx = candidates[start:end, 0:1]  # (chunk, 1)
            cy = candidates[start:end, 1:2]  # (chunk, 1)
            dx = cx - xy[:, 0][np.newaxis, :]  # (chunk, n)
            dy = cy - xy[:, 1][np.newaxis, :]
            dists = np.sqrt(dx * dx + dy * dy)
            clear = np.min(dists - r[np.newaxis, :], axis=1)  # (chunk,)
            best_r[start:end] = np.minimum(wall_clear[start:end], clear)
    else:
        best_r = wall_clear.copy()

    idx = int(np.argmax(best_r))
    best_x, best_y = candidates[idx, 0], candidates[idx, 1]
    best_r = max(best_r[idx], 0.0)

    # Local refinement around the best point
    step = 1.0 / n_grid
    for _ in range(3):
        fine_gx = np.linspace(max(0, best_x - 3 * step), min(1, best_x + 3 * step), 30)
        fine_gy = np.linspace(max(0, best_y - 3 * step), min(1, best_y + 3 * step), 30)
        FGX, FGY = np.meshgrid(fine_gx, fine_gy, indexing='ij')
        fcands = np.column_stack([FGX.ravel(), FGY.ravel()])
        fM = len(fcands)
        fwall = np.minimum(np.minimum(fcands[:, 0], 1.0 - fcands[:, 0]),
                           np.minimum(fcands[:, 1], 1.0 - fcands[:, 1]))
        if n > 0:
            dx = fcands[:, 0:1] - xy[:, 0][np.newaxis, :]
            dy = fcands[:, 1:2] - xy[:, 1][np.newaxis, :]
            dists = np.sqrt(dx * dx + dy * dy)
            fclear = np.min(dists - r[np.newaxis, :], axis=1)
            fr = np.minimum(fwall, fclear)
        else:
            fr = fwall.copy()
        fidx = int(np.argmax(fr))
        if fr[fidx] > best_r + 1e-10:
            best_x, best_y = fcands[fidx, 0], fcands[fidx, 1]
            best_r = fr[fidx]
            step = (3 * step) / 30
        else:
            break

    return best_x, best_y, max(best_r, 0.0)


def _force_directed_relax(xy, r, deadline, max_iter=20000, dt_init=0.005, cooling=0.9995):
    """
    Force-directed relaxation: treat circles as physical particles.
    Forces:
      - Repulsion between overlapping circles (push apart)
      - Attraction from walls (push inward if too close)
      - Growth force on radius (grow as much as possible)
    Update positions and radii simultaneously with damped dynamics.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()

    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    dt = dt_init
    vel_xy = np.zeros_like(xy)
    vel_r = np.zeros_like(r)
    damping = 0.85

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Compute forces
        f_xy = np.zeros_like(xy)
        f_r = np.zeros_like(r)

        # Wall forces: push circles away from walls
        for i in range(n):
            # Left wall
            if xy[i, 0] < r[i]:
                f_xy[i, 0] += 10.0 * (r[i] - xy[i, 0])
            # Right wall
            if 1.0 - xy[i, 0] < r[i]:
                f_xy[i, 0] -= 10.0 * (r[i] - (1.0 - xy[i, 0]))
            # Bottom wall
            if xy[i, 1] < r[i]:
                f_xy[i, 1] += 10.0 * (r[i] - xy[i, 1])
            # Top wall
            if 1.0 - xy[i, 1] < r[i]:
                f_xy[i, 1] -= 10.0 * (r[i] - (1.0 - xy[i, 1]))

        # Pairwise repulsion for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1e-12:
                    # Perturb randomly
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Strong repulsion proportional to overlap
                    force_mag = 50.0 * overlap
                    ux = dx / dist
                    uy = dy / dist
                    f_xy[i, 0] += force_mag * ux
                    f_xy[i, 1] += force_mag * uy
                    f_xy[j, 0] -= force_mag * ux
                    f_xy[j, 1] -= force_mag * uy
                    # Slightly shrink both radii when overlapping
                    f_r[i] -= 5.0 * overlap
                    f_r[j] -= 5.0 * overlap

        # Growth force: each circle wants to grow
        # The max radius for circle i is limited by walls and other circles
        for i in range(n):
            wall_limit = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            circle_limit = wall_limit
            for j in range(n):
                if j == i:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Max radius i can have given j is fixed: dist - r[j]
                avail = dist - r[j]
                if avail < circle_limit:
                    circle_limit = avail
            max_r = max(min(wall_limit, circle_limit), 0.0)
            if max_r > r[i]:
                f_r[i] += 2.0 * (max_r - r[i])
            else:
                # If can't grow, slight shrink to resolve overlaps
                f_r[i] -= 1.0 * max(0.0, r[i] - max_r)

        # Update velocities with damping
        vel_xy = damping * vel_xy + dt * f_xy
        vel_r = damping * vel_r + dt * f_r

        # Update positions and radii
        xy += vel_xy
        r += vel_r

        # Clamp to valid range
        for i in range(n):
            r[i] = max(r[i], 0.001)
            # Keep center within [r_i, 1-r_i]
            r_clip = min(r[i], 0.49)
            xy[i, 0] = np.clip(xy[i, 0], r_clip, 1.0 - r_clip)
            xy[i, 1] = np.clip(xy[i, 1], r_clip, 1.0 - r_clip)
            r[i] = min(r[i], xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            r[i] = max(r[i], 0.001)

        # Cool down
        dt *= cooling

        # Check improvement every 100 iterations
        if it % 100 == 0:
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total + 1e-10:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()

    # Final check
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total:
        best_total = total
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()

    return best_xy, best_r, best_total, best_valid


def _simulated_annealing(xy, r, deadline, max_iter=50000, T_init=0.05, T_final=0.001, cooling=0.9999):
    """
    Simulated annealing: randomly perturb one circle's position and radius,
    accept if it improves or with probability exp(-delta/T).
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    rng = np.random.RandomState(42)

    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    T = T_init
    current_total = best_total if best_valid else -1.0

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Pick a random circle to perturb
        i = rng.randint(n)

        # Save old state
        old_x, old_y, old_r = xy[i, 0], xy[i, 1], r[i]
        old_total = current_total

        # Perturb position
        xy[i, 0] += rng.normal(0, 0.03)
        xy[i, 1] += rng.normal(0, 0.03)
        # Perturb radius
        r[i] *= (1.0 + rng.normal(0, 0.1))
        r[i] = max(r[i], 0.001)

        # Clamp
        r_clip = min(r[i], 0.49)
        xy[i, 0] = np.clip(xy[i, 0], r_clip, 1.0 - r_clip)
        xy[i, 1] = np.clip(xy[i, 1], r_clip, 1.0 - r_clip)
        r[i] = min(r[i], xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        r[i] = max(r[i], 0.001)

        # Evaluate
        new_total, new_valid = _eval_packing(xy, r)
        if not new_valid:
            new_total = -1.0

        delta = new_total - old_total
        if delta > 0 or (T > T_final and rng.random() < np.exp(-delta / T)):
            current_total = new_total
            if new_valid and new_total > best_total:
                best_total = new_total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
        else:
            # Reject
            xy[i, 0] = old_x
            xy[i, 1] = old_y
            r[i] = old_r

        T *= cooling

    return best_xy, best_r, best_total, best_valid


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


def _seeded_init(n_seeds=26):
    """
    Seed circles using largest-empty-circle placement.
    Start with a few large circles, then fill in with largest empty circles.
    """
    xy_list = []
    r_list = []

    # Start with 4-6 large circles in corners and center
    initial_positions = [
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)
    ]
    for x, y in initial_positions[:min(5, n_seeds)]:
        r = 0.22
        xy_list.append([x, y])
        r_list.append(r)

    # Add remaining circles at largest empty spots
    for _ in range(len(initial_positions), n_seeds):
        xy = np.array(xy_list) if xy_list else np.zeros((0, 2))
        r = np.array(r_list) if r_list else np.zeros(0)
        x, y, rad = _largest_empty_circle(xy, r, n_grid=150)
        if rad < 0.01:
            # Fallback: random small circle
            rng = np.random.RandomState(len(xy_list))
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            rad = 0.05
        xy_list.append([x, y])
        r_list.append(max(rad, 0.02))

    return np.array(xy_list), np.array(r_list)


def _gradient_polish(xy, r, deadline, beta=80.0, penalty_weight=200.0):
    """
    Gradient-based polish using L-BFGS-B on a smooth penalty objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        x = v[0:n]
        y = v[n:2*n]
        rr = v[2*n:3*n]

        # Wall violations
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = rr - wall_min
        wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

        # Pairwise violations
        total_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = rr[i] + rr[j] - dist
                total_viol += np.logaddexp(0.0, beta * viol) / beta

        return -np.sum(rr) + penalty_weight * (wall_pen + total_viol)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        x = v[0:n]
        y = v[n:2*n]
        rr = v[2*n:3*n]
        g = np.zeros(3 * n)

        # Wall gradient
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = rr - wall_min
        sig = 1.0 / (1.0 + np.exp(-beta * wall_viol))
        for i in range(n):
            s = sig[i]
            if s < 1e-10:
                continue
            dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
            min_idx = int(np.argmin(dists))
            pw = penalty_weight
            if min_idx == 0:
                g[2*n + i] += pw * s
                g[i] -= pw * s
            elif min_idx == 1:
                g[2*n + i] += pw * s
                g[i] += pw * s
            elif min_idx == 2:
                g[2*n + i] += pw * s
                g[n + i] -= pw * s
            else:
                g[2*n + i] += pw * s
                g[n + i] += pw * s

        # Pairwise gradient
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1e-20:
                    continue
                dist = np.sqrt(dist_sq)
                viol = rr[i] + rr[j] - dist
                s = 1.0 / (1.0 + np.exp(-beta * viol))
                if s < 1e-10:
                    continue
                factor = penalty_weight * s
                g[i] += factor * dx / dist
                g[n + i] += factor * dy / dist
                g[j] -= factor * dx / dist
                g[n + j] -= factor * dy / dist
                g[2*n + i] += factor
                g[2*n + j] += factor

        g[2*n:3*n] -= 1.0
        return g

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 50000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Seeded initialization (largest empty circle)
    if time.time() < deadline:
        xy, r = _seeded_init(N)
        # Force-directed relaxation
        out_xy, out_r, total, valid = _force_directed_relax(xy, r, deadline - 60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Grid initializations + force-directed
    for side in [5, 6, 7]:
        if time.time() > deadline - 40:
            break
        grid = _grid_init(side)
        xy = grid[:, :2].copy()
        r = grid[:, 2].copy()
        out_xy, out_r, total, valid = _force_directed_relax(xy, r, deadline - 40)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Simulated annealing from best so far
    if best_valid and best_xy is not None and time.time() < deadline - 20:
        sa_xy, sa_r, sa_total, sa_valid = _simulated_annealing(
            best_xy.copy(), best_r.copy(), deadline - 20)
        if sa_valid and sa_total > best_total:
            best_total = sa_total
            best_xy = sa_xy
            best_r = sa_r

    # Phase 4: Gradient polish
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        for beta, pw in [(80.0, 200.0), (150.0, 500.0)]:
            if time.time() < deadline - 5:
                pol_xy, pol_r = _gradient_polish(best_xy, best_r, deadline - 5,
                                                 beta=beta, penalty_weight=pw)
                total, valid = _eval_packing(pol_xy, pol_r)
                if valid and total > best_total:
                    best_total = total
                    best_xy = pol_xy
                    best_r = pol_r

    # Phase 5: Final force-directed micro-relaxation
    if best_valid and best_xy is not None and time.time() < deadline - 2:
        micro_xy, micro_r, micro_total, micro_valid = _force_directed_relax(
            best_xy.copy(), best_r.copy(), deadline - 1, max_iter=5000, dt_init=0.001, cooling=0.999)
        if micro_valid and micro_total > best_total:
            best_total = micro_total
            best_xy = micro_xy
            best_r = micro_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        grid = _grid_init(side)
        xy = grid[:, :2].copy()
        r = grid[:, 2].copy()
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
