"""Combined program: force-directed / Gauss-Newton on Delaunay contact graph with penalty fallback."""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(arr):
    """Check that circles are inside [0,1]^2 and non-overlapping."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r < 0) or np.any(x - r < -1e-9) or np.any(y - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True


def _initial_hex(n):
    """Generate initial positions in a hexagonal packing pattern scaled to unit square."""
    row_counts = [5, 5, 5, 5, 6]
    total = sum(row_counts)
    if total != n:
        side = int(np.ceil(np.sqrt(n)))
        pts = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
        return np.array(pts[:n])

    d = 0.18
    vert_spacing = d * np.sqrt(3) / 2
    pts = []
    row_idx = 0
    for row in row_counts:
        offset = 0 if row_idx % 2 == 0 else d / 2
        total_width = (row - 1) * d
        start_x = 0.5 - total_width / 2 + offset
        for col in range(row):
            x = start_x + col * d
            y = 0.5 + (row_idx - 2) * vert_spacing
            pts.append((x, y))
        row_idx += 1
    return np.array(pts[:n])


def _delaunay_edges(points):
    """Get Delaunay edges from points."""
    if len(points) < 3:
        return list(zip(range(len(points)), range(1, len(points))))
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    for _ in range(100):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    return p


def _score_valid(p):
    return np.sum(p[:, 2])


def _gn_solve(start_xy, start_r, time_limit=15.0):
    """
    Gauss-Newton style solver on Delaunay contact graph.
    Variables: x, y, r for each circle.
    Objective: maximize sum(r)
    Constraints:
      - Wall: x >= r, y >= r, x + r <= 1, y + r <= 1
      - Contact (Delaunay edges): ||p_i - p_j|| >= r_i + r_j
    We treat this as: at each step, compute residual violations, form J^T J,
    solve for delta, line-search.
    """
    n = N
    xy = start_xy.copy()
    r = start_r.copy()
    t_start = time.time()

    # We'll do a simple iterative scheme:
    # 1. Compute Delaunay edges
    # 2. For each edge, if overlapping, push apart
    # 3. For walls, if violating, shrink radius or push inward
    # 4. Grow radii where possible (local gradient ascent on sum(r))

    for iteration in range(500):
        if time.time() - t_start > time_limit:
            break

        # Enforce wall constraints
        for i in range(n):
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            if r[i] > wall_dist:
                r[i] = max(wall_dist, 1e-6)

        # Get Delaunay edges
        edges = _delaunay_edges(xy)

        # Resolve overlaps on Delaunay edges
        for (i, j) in edges:
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            dist_safe = max(dist, 1e-10)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # Push apart along the connecting vector
                ux = dx / dist_safe
                uy = dy / dist_safe
                # Move each circle by half the overlap (weighted by radius)
                total_r = r[i] + r[j]
                if total_r > 1e-10:
                    wi = r[i] / total_r
                    wj = r[j] / total_r
                else:
                    wi = wj = 0.5
                push = overlap * 0.5
                xy[i, 0] += ux * push * wi
                xy[i, 1] += uy * push * wi
                xy[j, 0] -= ux * push * wj
                xy[j, 1] -= uy * push * wj

        # Re-enforce walls after pushing
        for i in range(n):
            xy[i, 0] = np.clip(xy[i, 0], 0.0, 1.0)
            xy[i, 1] = np.clip(xy[i, 1], 0.0, 1.0)
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            if r[i] > wall_dist:
                r[i] = max(wall_dist, 1e-6)

        # Try to grow radii: for each circle, compute max possible radius
        # given current positions and neighbors
        for i in range(n):
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Circle i can have radius up to (dist - r[j]) if we keep j fixed
                possible = (dist - r[j])
                if possible < max_r:
                    max_r = possible
            # Grow toward max_r with a damping factor
            target = max(1e-6, max_r)
            r[i] = r[i] + 0.3 * (target - r[i])
            r[i] = np.clip(r[i], 1e-6, 0.5)

    params = np.empty(3 * N)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    return params


def _penalty_objective(params, penalty=1000.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0):
    """Analytic gradient of penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    n = N
    g = np.zeros_like(params)

    g[2::3] = -1.0

    for i in range(n):
        if r[i] - x[i] > 0:
            g[3 * i] += penalty * (-1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3 * i] += penalty * (+1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3 * i + 1] += penalty * (-1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3 * i + 1] += penalty * (+1.0)
            g[3 * i + 2] += penalty * (+1.0)

    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            g[3 * i] += penalty * (-adx[k])
            g[3 * j] += penalty * (+adx[k])
            g[3 * i + 1] += penalty * (-ady[k])
            g[3 * j + 1] += penalty * (+ady[k])
            g[3 * i + 2] += penalty * (+1.0)
            g[3 * j + 2] += penalty * (+1.0)

    return g


def _hex_initial(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    p = points[indices]
    radii = np.full(N, 0.05)
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3 * i] = 0.0
        hi[3 * i] = 1.0
        lo[3 * i + 1] = 0.0
        hi[3 * i + 1] = 1.0
        lo[3 * i + 2] = 1e-6
        hi[3 * i + 2] = 0.5
    return (lo, hi)


def _shake_and_refine(p, steps=3000, time_limit=10.0, seed=42):
    """
    Refine a valid configuration using random perturbations (shake and solve).
    """
    current_score = _score_valid(p)
    best_p = p.copy()
    best_score = current_score
    rng = np.random.RandomState(seed)
    start = time.time()
    sigma = 0.015

    for _ in range(steps):
        if time.time() - start > time_limit:
            break

        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, sigma * 0.5, size=N)
        p_new = _enforce_validity(p_new)
        score = _score_valid(p_new)
        if score > best_score:
            best_score = score
            best_p = p_new.copy()
            # Reduce sigma as we improve
            sigma = max(0.005, sigma * 0.99)
        else:
            sigma = min(0.03, sigma * 1.01)

    return best_p, best_score


def construct():
    t_start = time.time()
    deadline = t_start + 140  # Safety margin before 150s kill

    best_arr = None
    best_score = -np.inf

    # Generate multiple initial configurations
    initializers = [
        ('hex', _hex_initial(0)),
        ('grid', _grid_initial(0)),
        ('hex2', _hex_initial(42)),
    ]

    for name, params_init in initializers:
        if time.time() > deadline - 25:
            break

        p = _enforce_validity(params_init)
        score = _score_valid(p)

        # Phase 1: Gauss-Newton / force-directed on Delaunay
        p_gn = _gn_solve(p[:, :2], p[:, 2], time_limit=10.0)
        p_gn = _enforce_validity(p_gn)
        score_gn = _score_valid(p_gn)
        if score_gn > score:
            p = p_gn
            score = score_gn

        # Phase 2: Penalty-based L-BFGS-B refinement
        if time.time() < deadline - 15:
            from scipy.optimize import minimize
            lo, hi = _bounds()
            try:
                result = minimize(
                    _penalty_objective, p.flatten(),
                    jac=_penalty_grad, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                p_opt = _enforce_validity(result.x)
                score_opt = _score_valid(p_opt)
                if score_opt > score:
                    p = p_opt
                    score = score_opt
            except Exception:
                pass

        # Phase 3: Shake and refine
        if time.time() < deadline - 5:
            p_shake, score_shake = _shake_and_refine(p, steps=2000, time_limit=8.0, seed=hash(name) % 2**31)
            if score_shake > score:
                p = p_shake
                score = score_shake

        if score > best_score:
            best_score = score
            best_arr = p.copy()

    # Fallback to grid if optimization failed
    if best_arr is None or not _validate(best_arr):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])

    return best_arr

# EVOLVE-BLOCK-END
