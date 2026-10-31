"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines a Voronoi-based geometric model (radii = min distance to cell edges,
guaranteeing non-overlap by construction) with greedy best-fit seeding,
direct gradient ascent on the seed coordinates, and a hard-constraint
grow-and-push refinement that carves space for larger circles."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-8):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalty. Vectorized."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    left = np.maximum(0.0, radii - pos[:, 0])
    right = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bot = np.maximum(0.0, radii - pos[:, 1])
    top = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    bpen = np.sum(left * left) + np.sum(right * right) + np.sum(bot * bot) + np.sum(top * top)

    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = np.maximum(0.0, min_dist - dist)
    np.fill_diagonal(viol, 0.0)
    open_pen = 0.5 * np.sum(viol * viol)

    return -np.sum(radii) + 100.0 * (bpen + open_pen)


# ---------------- Voronoi-based model ----------------

def _voronoi_radii(seeds):
    """
    Given N seed points, compute the radius of each circle as the minimum
    distance from the seed to the edges of its Voronoi cell (including
    the unit-square boundary). This guarantees non-overlap by construction.
    """
    n = len(seeds)
    # Augment with boundary points to handle the square boundary
    # We use a set of points far outside to create bounding edges
    boundary_pts = np.array([
        [-10, -10], [-10, 10], [10, -10], [10, 10],
        [-10, 0], [-10, 0.25], [-10, 0.5], [-10, 0.75], [-10, 1.0],
        [10, 0], [10, 0.25], [10, 0.5], [10, 0.75], [10, 1.0],
        [0, -10], [0.25, -10], [0.5, -10], [0.75, -10], [1.0, -10],
        [0, 10], [0.25, 10], [0.5, 10], [0.75, 10], [1.0, 10],
    ])
    pts = np.vstack([seeds, boundary_pts])
    try:
        vor = Voronoi(pts)
    except Exception:
        # Fallback: use distance to nearest neighbor / boundary
        radii = np.zeros(n)
        for i in range(n):
            min_d = min(seeds[i, 0], 1 - seeds[i, 0], seeds[i, 1], 1 - seeds[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(seeds[i] - seeds[j])
                    min_d = min(min_d, d / 2)
            radii[i] = min_d * 0.999
        return radii

    radii = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Unbounded region; compute distance to boundary manually
            min_d = min(seeds[i, 0], 1 - seeds[i, 0], seeds[i, 1], 1 - seeds[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(seeds[i] - seeds[j])
                    min_d = min(min_d, d / 2)
            radii[i] = min_d * 0.999
            continue
        min_d = np.inf
        for edge_idx in region:
            edge = vor.ridges[edge_idx]
            p1 = vor.vertices[edge[0]]
            p2 = vor.vertices[edge[1]]
            # Distance from seeds[i] to line segment p1-p2
            d = _point_segment_dist(seeds[i], p1, p2)
            if d < min_d:
                min_d = d
        # Also check distance to unit square boundary
        boundary_d = min(seeds[i, 0], 1 - seeds[i, 0], seeds[i, 1], 1 - seeds[i, 1])
        radii[i] = min(min_d, boundary_d) * 0.999

    return radii


def _point_segment_dist(p, a, b):
    """Distance from point p to line segment a-b."""
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-15)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def _voronoi_objective(seeds, deadline):
    """Objective for optimizing seed positions: maximize sum of Voronoi radii."""
    if time.time() > deadline:
        return 1e10
    radii = _voronoi_radii(seeds)
    return -np.sum(radii)


def _grow_and_push(pos, radii, deadline, max_iter=5000):
    """
    Hard-constraint grow-and-push: iteratively grow the smallest circle,
    push overlapping circles apart, and clamp to boundaries.
    This carves space for larger circles rather than just nudging in a penalty landscape.
    """
    pos = pos.copy()
    radii = radii.copy()
    n = len(radii)

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Find the smallest circle to grow
        min_idx = np.argmin(radii)
        # Compute max possible radius at current position
        max_r = min(pos[min_idx, 0], 1.0 - pos[min_idx, 0],
                    pos[min_idx, 1], 1.0 - pos[min_idx, 1])
        # Also limited by nearest neighbor
        for j in range(n):
            if j == min_idx:
                continue
            dx = pos[min_idx, 0] - pos[j, 0]
            dy = pos[min_idx, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        max_r = max(max_r, 0.001)

        if radii[min_idx] >= max_r * 0.999:
            # Can't grow this one; try a random one
            min_idx = np.random.randint(n)
            max_r = min(pos[min_idx, 0], 1.0 - pos[min_idx, 0],
                        pos[min_idx, 1], 1.0 - pos[min_idx, 1])
            for j in range(n):
                if j == min_idx:
                    continue
                dx = pos[min_idx, 0] - pos[j, 0]
                dy = pos[min_idx, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            max_r = max(max_r, 0.001)
            if radii[min_idx] >= max_r * 0.999:
                # All circles at local max; try to push apart
                pass
            else:
                radii[min_idx] = max_r * 0.999

        # Push overlapping circles apart
        for _ in range(20):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist - 1e-10:
                        if dist < 1e-10:
                            # Overlapping at same point; push apart randomly
                            angle = np.random.uniform(0, 2 * np.pi)
                            push = (min_dist - dist) / 2 + 0.001
                            pos[i, 0] += push * np.cos(angle)
                            pos[i, 1] += push * np.sin(angle)
                            pos[j, 0] -= push * np.cos(angle)
                            pos[j, 1] -= push * np.sin(angle)
                        else:
                            push = (min_dist - dist) / 2 + 0.0005
                            dx_n = dx / dist
                            dy_n = dy / dist
                            pos[i, 0] += push * dx_n
                            pos[i, 1] += push * dy_n
                            pos[j, 0] -= push * dx_n
                            pos[j, 1] -= push * dy_n
                        moved = True

            # Clamp to boundaries
            for i in range(n):
                pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
                pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
                # If clamped, shrink radius if needed
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                if radii[i] > max_r:
                    radii[i] = max_r * 0.999

            if not moved:
                break

    return pos, radii


def _greedy_best_fit(deadline, seed=42):
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    Places circles one by one, each at the position that maximizes its radius.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    rng = np.random.RandomState(seed)

    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.22

    for i in range(1, N):
        if time.time() > deadline:
            break
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False

        # Sample many random points
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r

            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True

        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003

        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97

    return pos, radii


def _make_hexagonal(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < n:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        for col in range(5):
            if k >= n:
                break
            x = start_x + col * dx
            y = row * dy + r
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
    return pos, radii


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision L-BFGS-B polish with tight bounds and tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _voronoi_gradient_ascent(seeds, deadline, max_iter=2000):
    """
    Gradient ascent on seed coordinates to maximize sum of Voronoi radii.
    Uses numerical gradients.
    """
    seeds = seeds.copy()
    n = len(seeds)
    lr = 0.005
    best_score = np.sum(_voronoi_radii(seeds))
    best_seeds = seeds.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Compute numerical gradient
        grad = np.zeros_like(seeds)
        eps = 0.001
        for i in range(n):
            for d in range(2):
                seeds_plus = seeds.copy()
                seeds_plus[i, d] += eps
                seeds_plus = np.clip(seeds_plus, 0.01, 0.99)
                f_plus = -np.sum(_voronoi_radii(seeds_plus))

                seeds_minus = seeds.copy()
                seeds_minus[i, d] -= eps
                seeds_minus = np.clip(seeds_minus, 0.01, 0.99)
                f_minus = -np.sum(_voronoi_radii(seeds_minus))

                grad[i, d] = (f_plus - f_minus) / (2 * eps)

        # Gradient ascent step
        seeds = seeds - lr * grad
        seeds = np.clip(seeds, 0.01, 0.99)

        # Update best
        radii = _voronoi_radii(seeds)
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_seeds = seeds.copy()

        # Decay learning rate
        lr *= 0.999

    return best_seeds


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # ---------- Track A: Voronoi-based optimization ----------
    if time.time() < deadline - 60:
        # Start from greedy best-fit
        pos_gf, radii_gf = _greedy_best_fit(deadline, seed=42)
        update_best(pos_gf, radii_gf)

        # Use greedy positions as seeds for Voronoi optimization
        seeds = pos_gf.copy()
        if time.time() < deadline - 50:
            seeds_opt = _voronoi_gradient_ascent(seeds, deadline, max_iter=1500)
            radii_v = _voronoi_radii(seeds_opt)
            update_best(seeds_opt, radii_v)

        # Second Voronoi run from hexagonal seeds
        pos_hex, radii_hex = _make_hexagonal(N)
        update_best(pos_hex, radii_hex)
        if time.time() < deadline - 40:
            seeds_hex = pos_hex.copy()
            seeds_opt2 = _voronoi_gradient_ascent(seeds_hex, deadline, max_iter=1000)
            radii_v2 = _voronoi_radii(seeds_opt2)
            update_best(seeds_opt2, radii_v2)

    # ---------- Track B: Grow-and-push from best configuration ----------
    if best_pos is not None and time.time() < deadline - 30:
        pos_gp, radii_gp = _grow_and_push(best_pos, best_radii, deadline, max_iter=3000)
        if _is_valid(pos_gp, radii_gp):
            s = np.sum(radii_gp)
            if s > best_score:
                best_score = s
                best_pos = pos_gp.copy()
                best_radii = radii_gp.copy()

    # ---------- Track C: Direct L-BFGS-B from multiple seeds ----------
    configs = []
    pos_gf, radii_gf = _greedy_best_fit(deadline, seed=123)
    configs.append((pos_gf, radii_gf))
    configs.append(_make_hexagonal(N))
    if best_pos is not None:
        configs.append((best_pos, best_radii))

    for pos, radii in configs:
        if time.time() > deadline - 20:
            break
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 400, 'ftol': 1e-11})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # ---------- Track D: Second grow-and-push with different seed ----------
    if best_pos is not None and time.time() < deadline - 15:
        rng = np.random.RandomState(789)
        pos_perturbed = best_pos + rng.uniform(-0.02, 0.02, (N, 2))
        pos_perturbed = np.clip(pos_perturbed, 0.01, 0.99)
        pos_gp2, radii_gp2 = _grow_and_push(pos_perturbed, best_radii.copy(), deadline, max_iter=2000)
        if _is_valid(pos_gp2, radii_gp2):
            s2 = np.sum(radii_gp2)
            if s2 > best_score:
                best_score = s2
                best_pos = pos_gp2.copy()
                best_radii = radii_gp2.copy()

    # ---------- Final polish of overall best ----------
    if best_pos is not None and time.time() < deadline - 5:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # Fallback
    if best_pos is None:
        pos, radii = _greedy_best_fit(deadline, seed=999)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
