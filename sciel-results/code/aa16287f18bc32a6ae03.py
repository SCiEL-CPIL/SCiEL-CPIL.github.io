"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 115.0  # safety margin before 120s kill


def _min_triangle_area(points):
    """Compute minimum triangle area among all C(n,3) triples."""
    n = len(points)
    if n < 3:
        return 0.0
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 0.0
    min_ta = _min_triangle_area(points)
    return min_ta / hull_a


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _initial_polygon(n):
    """Regular n-gon centered at origin, unit radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _initial_polygon_plus_center(n):
    """Regular (n-1)-gon plus center point."""
    if n <= 1:
        return np.zeros((n, 2))
    m = n - 1
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    return np.vstack([pts, center])


def _initial_hex_cluster(n):
    """Hexagonal-like cluster: center, ring at r=1, ring at r=1.8."""
    pts = [np.array([0.0, 0.0])]
    # Inner ring: 6 points
    for i in range(6):
        a = i * np.pi / 3
        pts.append(np.array([np.cos(a), np.sin(a)]))
    # Outer ring: 6 points
    for i in range(6):
        a = i * np.pi / 3 + np.pi / 6
        pts.append(np.array([1.8 * np.cos(a), 1.8 * np.sin(a)]))
    pts = np.array(pts[:n])
    return pts


def _initial_grid(n):
    """Staggered grid initial configuration."""
    points = []
    cols = 4
    rows = 4
    for r in range(rows):
        for c in range(cols):
            if len(points) >= n:
                break
            x = c / (cols - 1)
            y = r / (rows - 1)
            if r % 2 == 1:
                x += 0.5 / (cols - 1)
            points.append([x, y])
    while len(points) < n:
        points.append([0.5, 0.5])
    return np.array(points[:n])


def _optimize_lbfgsb(start_points, deadline):
    """Optimize using L-BFGS-B."""
    best_points = start_points.copy()
    best_score = _score(best_points)

    def objective(x):
        points = x.reshape(-1, 2)
        return _negative_score(points)

    x0 = start_points.flatten()
    bounds = [(-3, 3)] * (2 * N)

    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.fun < _negative_score(x0):
            cand = result.x.reshape(-1, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_points = cand
                best_score = cand_score
    except Exception:
        pass

    return best_points, best_score


def _optimize_nelder_mead(start_points, deadline):
    """Optimize using Nelder-Mead."""
    best_points = start_points.copy()
    best_score = _score(best_points)

    def objective(x):
        points = x.reshape(-1, 2)
        return _negative_score(points)

    x0 = start_points.flatten()

    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        if result.fun < _negative_score(x0):
            cand = result.x.reshape(-1, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_points = cand
                best_score = cand_score
    except Exception:
        pass

    return best_points, best_score


def _subgradient_polish(points, n_steps=50, deadline=None):
    """
    Subgradient ascent: repeatedly find the min-area triangle and push
    its vertices apart to increase that area.
    """
    points = points.copy()
    best_score = _score(points)
    best_points = points.copy()
    lr = 0.02

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 2.0:
            break

        n = len(points)
        min_area = np.inf
        min_idx = (0, 1, 2)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a = points[i]
                    b = points[j]
                    c = points[k]
                    area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) -
                                     (b[1] - a[1]) * (c[0] - a[0]))
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)

        if min_area < 1e-14:
            points = points + np.random.default_rng().normal(0, 0.01, points.shape)
            continue

        i, j, k = min_idx
        a = points[i]
        b = points[j]
        c = points[k]

        cross_val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        sign = np.sign(cross_val) if cross_val != 0 else 1.0

        # Gradient of 0.5*|cross| w.r.t. each vertex
        # d/d a: 0.5 * sign * (-(c1-a1), (c0-a0))  ... let me re-derive
        # cross = (b0-a0)(c1-a1) - (b1-a1)(c0-a0)
        # d(cross)/da0 = -(c1-a1), d(cross)/da1 = (c0-a0)
        # d(cross)/db0 = (c1-a1),  d(cross)/db1 = -(c0-a0)
        # d(cross)/dc0 = -(b1-a1), d(cross)/dc1 = (b0-a0)
        # area = 0.5 * sign * cross (locally)
        # grad_a = 0.5*sign*(-(c1-a1), (c0-a0))
        # grad_b = 0.5*sign*((c1-a1), -(c0-a0))
        # grad_c = 0.5*sign*(-(b1-a1), (b0-a0))

        ga = 0.5 * sign * np.array([-(c[1] - a[1]), (c[0] - a[0])])
        gb = 0.5 * sign * np.array([(c[1] - a[1]), -(c[0] - a[0])])
        gc = 0.5 * sign * np.array([-(b[1] - a[1]), (b[0] - a[0])])

        # We want to INCREASE the area, so move along +gradient
        # But we also want to keep the hull roughly the same size
        # Apply step
        points[i] += lr * ga
        points[j] += lr * gb
        points[k] += lr * gc

        # Normalize: center and scale to keep hull area ~1
        center = np.mean(points, axis=0)
        points -= center
        hull_a = _hull_area(points)
        if hull_a > 1e-10:
            scale = np.sqrt(1.0 / hull_a)
            points *= scale

        cur_score = _score(points)
        if cur_score > best_score:
            best_score = cur_score
            best_points = points.copy()

        # Decay learning rate
        lr *= 0.97

    return best_points, best_score


def construct():
    """Construct 13 points maximizing min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_points = None
    best_score = 0.0

    # Generate diverse initial configurations
    initials = []
    initials.append(_initial_polygon(N))
    initials.append(_initial_polygon_plus_center(N))
    initials.append(_initial_hex_cluster(N))
    initials.append(_initial_grid(N))

    # Add perturbed versions
    rng = np.random.RandomState(42)
    for base in initials[:2]:
        initials.append(base + rng.normal(0, 0.08, size=(N, 2)))

    # Phase 1: L-BFGS-B from each initial
    for init_pts in initials:
        if time.time() > deadline - 15.0:
            break
        pts, score = _optimize_lbfgsb(init_pts, deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 2: Nelder-Mead from best found and top initials
    candidates = [best_points] if best_points is not None else []
    candidates.append(initials[0])
    candidates.append(initials[1])

    for cand in candidates:
        if time.time() > deadline - 20.0:
            break
        pts, score = _optimize_nelder_mead(cand, deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 3: Subgradient polish from best
    if best_points is not None and time.time() < deadline - 5.0:
        pts, score = _subgradient_polish(best_points, n_steps=80, deadline=deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 4: Final L-BFGS-B polish
    if best_points is not None and time.time() < deadline - 3.0:
        pts, score = _optimize_lbfgsb(best_points, deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    if best_points is None:
        return initials[0]

    return best_points
