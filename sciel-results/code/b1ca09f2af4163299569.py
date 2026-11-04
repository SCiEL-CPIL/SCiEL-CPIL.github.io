"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
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


def _largest_empty_circle(xy, r):
    """
    Find the largest empty circle center and radius given existing circles.
    Uses Delaunay triangulation of circle centers (with wall points) to find
    candidate empty-circle centers at triangle circumcenters and midpoints to walls.
    Returns (cx, cy, max_r).
    """
    n = len(xy)
    if n == 0:
        return 0.5, 0.5, 0.5

    # Candidate centers: triangle circumcenters + wall-midpoint candidates
    # We use the circle centers as points for Delaunay
    pts = xy.copy()

    # Add boundary "virtual" points far away to help capture wall-adjacent voids
    # Actually, a better approach: check circumcenters of triangles AND
    # check points along the walls at optimal positions.

    best_r = 0.0
    best_cx, best_cy = 0.5, 0.5

    # Method: For each triangle in Delaunay, compute circumcenter and check clearance
    if n >= 3:
        try:
            tri = Delaunay(pts)
            for simplex in tri.simplices:
                p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                # Circumcenter
                ax, ay = p1
                bx, by = p2
                cx, cy = p3
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-15:
                    continue
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
                      (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
                      (cx**2 + cy**2) * (bx - ax)) / d

                # Check if circumcenter is inside the square
                if ux < 0 or ux > 1 or uy < 0 or uy > 1:
                    continue

                # Clearance to all circles
                dx = ux - xy[:, 0]
                dy = uy - xy[:, 1]
                dists = np.sqrt(dx**2 + dy**2) - r
                wall_clear = min(ux, 1 - ux, uy, 1 - uy)
                clr = min(np.min(dists), wall_clear)
                if clr > best_r:
                    best_r = clr
                    best_cx, best_cy = ux, uy
        except Exception:
            pass

    # Also check wall-adjacent candidates: for each wall, the best spot
    # is where the distance to nearest circles is maximized.
    # Simple heuristic: check a grid along each wall
    for wall in range(4):
        for t in np.linspace(0.05, 0.95, 20):
            if wall == 0:  # bottom: y=0
                cx, cy = t, 0.0
            elif wall == 1:  # top: y=1
                cx, cy = t, 1.0
            elif wall == 2:  # left: x=0
                cx, cy = 0.0, t
            else:  # right: x=1
                cx, cy = 1.0, t

            dx = cx - xy[:, 0]
            dy = cy - xy[:, 1]
            dists = np.sqrt(dx**2 + dy**2) - r
            wall_clear = min(cx, 1 - cx, cy, 1 - cy)
            clr = min(np.min(dists) if len(dists) > 0 else 1.0, wall_clear)
            if clr > best_r:
                best_r = clr
                best_cx, best_cy = cx, cy

    return best_cx, best_cy, max(best_r, 0.0)


def _force_directed_relax(xy, r, deadline, n_iter=3000, dt_init=0.02, dt_decay=0.998):
    """
    Force-directed relaxation: circles repel each other and walls push inward.
    Radii are held fixed during relaxation; positions are adjusted.
    Uses a damped velocity Verlet-like scheme.
    """
    n = len(xy)
    pos = xy.copy().astype(float)
    vel = np.zeros((n, 2))
    dt = dt_init
    best_pos = pos.copy()
    best_score = 0.0

    # Compute initial score (sum of radii is constant, but we track validity)
    for it in range(n_iter):
        if time.time() > deadline:
            break
        dt *= dt_decay
        if dt < 1e-6:
            break

        forces = np.zeros((n, 2))

        # Circle-circle repulsion: force proportional to overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    # Jitter
                    dx, dy = 0.01, 0.01
                    dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Repulsive force: stronger for larger overlap
                    f = overlap * 10.0
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Wall repulsion: push inward if too close
        for i in range(n):
            ri = r[i]
            # Left wall
            if pos[i, 0] - ri < 0:
                forces[i, 0] += (0 - (pos[i, 0] - ri)) * 20.0
            # Right wall
            if 1.0 - pos[i, 0] - ri < 0:
                forces[i, 0] -= (1.0 - pos[i, 0] - ri) * 20.0
            # Bottom wall
            if pos[i, 1] - ri < 0:
                forces[i, 1] += (0 - (pos[i, 1] - ri)) * 20.0
            # Top wall
            if 1.0 - pos[i, 1] - ri < 0:
                forces[i, 1] -= (1.0 - pos[i, 1] - ri) * 20.0

        # Update velocity with damping
        vel = 0.8 * vel + dt * forces
        # Limit velocity
        speed = np.sqrt(np.sum(vel**2, axis=1, keepdims=True))
        max_speed = 0.05
        mask = speed > max_speed
        vel[mask] *= max_speed / speed[mask]

        # Update positions
        pos += vel

        # Clamp to square (hard constraint)
        pos[:, 0] = np.clip(pos[:, 0], 0.01, 0.99)
        pos[:, 1] = np.clip(pos[:, 1], 0.01, 0.99)

        # Check validity periodically
        if it % 50 == 0 or it == n_iter - 1:
            # Ensure positions respect radii for validity check
            test_pos = pos.copy()
            test_r = r.copy()
            for i in range(n):
                test_pos[i, 0] = np.clip(test_pos[i, 0], test_r[i], 1.0 - test_r[i])
                test_pos[i, 1] = np.clip(test_pos[i, 1], test_r[i], 1.0 - test_r[i])
            total, valid = _eval_packing(test_pos, test_r)
            if valid and total > best_score:
                best_score = total
                best_pos = test_pos.copy()

    return best_pos, best_score


def _grow_circles_into_voids(xy, r, deadline, n_grows=50):
    """
    Iteratively find the largest empty circle and either:
    (a) grow an existing circle toward it if close, or
    (b) use it as a guide for local optimization.
    Here we use it to nudge the nearest circle toward the void center
    and grow its radius.
    """
    n = len(xy)
    pos = xy.copy().astype(float)
    rad = r.copy().astype(float)
    best_total = 0.0
    best_pos = pos.copy()
    best_rad = rad.copy()

    for grow_iter in range(n_grows):
        if time.time() > deadline:
            break

        # Find largest empty circle
        ecx, ecy, er = _largest_empty_circle(pos, rad)

        # Find the circle closest to the empty circle center
        dists = np.sqrt((pos[:, 0] - ecx)**2 + (pos[:, 1] - ecy)**2)
        closest = int(np.argmin(dists))

        # Nudge the closest circle toward the void center
        direction = np.array([ecx - pos[closest, 0], ecy - pos[closest, 1]])
        dist_to_void = np.linalg.norm(direction)
        if dist_to_void > 1e-8:
            direction /= dist_to_void

        # Try to move and grow
        step = min(0.05, dist_to_void * 0.5)
        new_pos = pos[closest].copy()
        new_pos[0] += direction[0] * step
        new_pos[1] += direction[1] * step

        # Compute max radius at new position
        dx = new_pos[0] - pos
        dy = new_pos[1] - pos
        dists_all = np.sqrt(dx**2 + dy**2)
        dists_all[closest] = 1e9  # Ignore self
        wall_clear = min(new_pos[0], 1 - new_pos[0], new_pos[1], 1 - new_pos[1])
        circle_clear = np.min(dists_all - rad)
        max_r = max(0.001, min(wall_clear, circle_clear))

        # Accept if it increases total
        old_total = np.sum(rad)
        new_total = old_total - rad[closest] + max_r
        if new_total > old_total + 1e-10:
            pos[closest] = new_pos
            rad[closest] = max_r

            # Check validity
            total, valid = _eval_packing(pos, rad)
            if valid and total > best_total:
                best_total = total
                best_pos = pos.copy()
                best_rad = rad.copy()

    return best_pos, best_rad, best_total


def _gradient_polish_positions(xy, r, deadline, max_iter=2000, tol=1e-10):
    """
    Gradient polish of positions only (radii fixed) to eliminate overlaps.
    Minimizes sum of squared overlaps + wall violations.
    """
    n = len(xy)

    def overlap_objective(v):
        pos = v.reshape(n, 2)
        # Wall violations
        wall_viol = 0.0
        for i in range(n):
            ri = r[i]
            wall_viol += max(0, ri - pos[i, 0])**2
            wall_viol += max(0, ri - (1 - pos[i, 0]))**2
            wall_viol += max(0, ri - pos[i, 1])**2
            wall_viol += max(0, ri - (1 - pos[i, 1]))**2
        # Pairwise overlaps
        pair_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    pair_viol += overlap**2
        return wall_viol + pair_viol

    v0 = xy.flatten()
    bounds = [(0.001, 0.999)] * (2 * n)

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return overlap_objective(v)

    try:
        result = minimize(obj, v0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'maxfun': 5000})
        new_pos = result.x.reshape(n, 2)
        # Clamp
        for i in range(n):
            ri = r[i]
            new_pos[i, 0] = np.clip(new_pos[i, 0], ri, 1 - ri)
            new_pos[i, 1] = np.clip(new_pos[i, 1], ri, 1 - ri)
        total, valid = _eval_packing(new_pos, r)
        if valid:
            return new_pos
    except Exception:
        pass
    return xy


def _simulated_annealing_positions(xy, r, deadline, n_iter=5000, T_init=0.05, T_min=0.001):
    """
    Simulated annealing on positions only (radii fixed).
    Accepts moves that increase overlap, with probability exp(-delta/T).
    """
    n = len(xy)
    pos = xy.copy().astype(float)
    best_pos = pos.copy()
    best_score = 0.0

    # Compute current overlap score (lower is better)
    def overlap_score(p):
        s = 0.0
        for i in range(n):
            ri = r[i]
            s += max(0, ri - p[i, 0])**2
            s += max(0, ri - (1 - p[i, 0]))**2
            s += max(0, ri - p[i, 1])**2
            s += max(0, ri - (1 - p[i, 1]))**2
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                ov = r[i] + r[j] - dist
                if ov > 0:
                    s += ov**2
        return s

    cur_score = overlap_score(pos)
    T = T_init
    cooling = (T_min / T_init) ** (1.0 / n_iter)

    for it in range(n_iter):
        if time.time() > deadline:
            break
        T *= cooling
        if T < T_min:
            break

        # Propose a random move: pick a random circle and nudge it
        i = np.random.randint(n)
        delta = np.array([np.random.uniform(-0.05, 0.05),
                          np.random.uniform(-0.05, 0.05)])
        new_pos = pos.copy()
        new_pos[i, 0] += delta[0]
        new_pos[i, 1] += delta[1]
        # Clamp
        ri = r[i]
        new_pos[i, 0] = np.clip(new_pos[i, 0], ri, 1 - ri)
        new_pos[i, 1] = np.clip(new_pos[i, 1], ri, 1 - ri)

        new_score = overlap_score(new_pos)
        delta_s = new_score - cur_score

        if delta_s < 0 or np.random.random() < np.exp(-delta_s / T):
            pos = new_pos
            cur_score = new_score
            # Check validity
            total, valid = _eval_packing(pos, r)
            if valid and total > best_score:
                best_score = total
                best_pos = pos.copy()

    return best_pos, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Multiple initializations
    inits = []

    # Grid inits
    for side in [5, 6, 7]:
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
        positions = positions[:N]
        inits.append((np.array([p[:2] for p in positions]), np.array([p[2] for p in positions])))

    # Random inits
    for seed in range(10):
        rng = np.random.RandomState(seed)
        xy = rng.uniform(0.1, 0.9, size=(N, 2))
        r = rng.uniform(0.02, 0.06, size=N)
        inits.append((xy, r))

    # Phase 2: For each init, run force-directed + void-growing + SA
    for init_idx, (xy0, r0) in enumerate(inits):
        if time.time() > deadline - 10:
            break

        # Step A: Force-directed relaxation
        pos, _ = _force_directed_relax(xy0, r0, deadline - 5)

        # Step B: Gradient polish to fix any residual overlaps
        pos = _gradient_polish_positions(pos, r0, deadline - 5)

        # Step C: Void-growing (grow circles into empty spaces)
        pos, rad, total = _grow_circles_into_voids(pos, r0, deadline - 5, n_grows=30)
        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = pos
            best_r = rad

        # Step D: Simulated annealing for further position optimization
        if time.time() < deadline - 10:
            pos2, score2 = _simulated_annealing_positions(best_xy, best_r, deadline - 5, n_iter=3000)
            total2, valid2 = _eval_packing(pos2, best_r)
            if valid2 and total2 > best_total:
                best_total = total2
                best_xy = pos2

        # Step E: Another round of void-growing with updated positions
        if time.time() < deadline - 8:
            pos3, rad3, total3 = _grow_circles_into_voids(best_xy, best_r, deadline - 5, n_grows=40)
            if total3 > best_total:
                best_total = total3
                best_valid = True
                best_xy = pos3
                best_r = rad3

    # Phase 3: Final gradient polish on the best result
    if best_valid and best_xy is not None and time.time() < deadline - 3:
        final_pos = _gradient_polish_positions(best_xy, best_r, deadline - 2, max_iter=3000)
        total_f, valid_f = _eval_packing(final_pos, best_r)
        if valid_f and total_f > best_total:
            best_total = total_f
            best_xy = final_pos

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
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
