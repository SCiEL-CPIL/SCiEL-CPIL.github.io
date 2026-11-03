"""Rewritten program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_score(points):
    """Compute the maximum radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_pair_r = min_pair / 2.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair_r, min_boundary)

def _soft_min_surrogate(points, beta=100.0):
    """
    Compute a smooth surrogate for the min of pairwise distances/2 and boundary distances.
    """
    n = len(points)
    if n < 2:
        return 0.0
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    flat_vals = pair_vals.flatten()
    flat_vals = flat_vals[~np.isinf(flat_vals)]
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    all_vals = np.concatenate([flat_vals, boundary_vals])
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    return soft_min

def _objective_smooth(x, beta=100.0):
    """Objective for smooth optimization: negative soft-min radius + boundary penalty."""
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 100.0 * np.sum(boundary_violation)
    radius = _soft_min_surrogate(pts, beta)
    return -radius + penalty

def _ring_initialization():
    """Generate initial configuration with concentric rings: 1+6+12+18+13=50."""
    points = []
    points.append([0.0, 0.0])
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 0.50
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    r3 = 0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    r4 = 0.90
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    return np.array(points)

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 0.24
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.97

def _random_initial(seed=0):
    """Random initialization inside unit circle."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 0.95:
                points.append([x, y])
                break
    return np.array(points)

def _polish_smooth(points, start_time, deadline, min_time_remaining=5.0):
    """
    High-precision polish using smooth surrogate with increasing beta.
    """
    current = points.copy()
    current_score = _compute_score(current)
    betas = [30, 60, 120, 240, 480, 960]
    for beta in betas:
        if time.time() > deadline - min_time_remaining:
            break
        def obj(x):
            return _objective_smooth(x, beta)
        try:
            result = minimize(
                obj, current.flatten(), method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            new_points = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0 + 1e-10:
                    mask = norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= norms[mask, np.newaxis] * 1.00001
                    new_score = _compute_score(new_points)
                    if new_score > current_score:
                        current = new_points
                        current_score = new_score
        except Exception:
            pass
    return current, current_score

def _polish_exact(points, start_time, deadline, min_time_remaining=3.0):
    """
    Final polish using direct optimization of the exact objective with penalty method.
    """
    current = points.copy()
    current_score = _compute_score(current)
    penalty_weights = [10, 50, 200, 1000, 5000]
    for pw in penalty_weights:
        if time.time() > deadline - min_time_remaining:
            break
        def exact_obj(x):
            pts = x.reshape(N, 2)
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_r = min_pair / 2.0
            norms = np.linalg.norm(pts, axis=1)
            min_boundary = 1.0 - np.max(norms)
            r = min(min_pair_r, min_boundary)
            violation = 0.0
            if min_pair_r < r:
                violation += (r - min_pair_r)
            if min_boundary < r:
                violation += (r - min_boundary)
            return -r + pw * violation**2
        try:
            result = minimize(
                exact_obj, current.flatten(), method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            new_points = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0 + 1e-10:
                    mask = norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= norms[mask, np.newaxis] * 1.00001
                    new_score = _compute_score(new_points)
                    if new_score > current_score:
                        current = new_points
                        current_score = new_score
        except Exception:
            pass
    return current, current_score

def _simulated_annealing(points, start_time, deadline, min_time_remaining=10.0):
    """
    Simulated annealing with adaptive step size.
    """
    rng = np.random.default_rng(42)
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    T = 0.005
    T_min = 1e-10
    cooling = 0.9999
    max_iter = 300000
    norms = np.linalg.norm(current_points, axis=1)
    for iteration in range(max_iter):
        if time.time() > deadline - min_time_remaining:
            break
        idx = rng.integers(0, N)
        scale = max(T, 1e-7) * 0.15
        delta = rng.normal(0, scale, size=2)
        new_points = current_points.copy()
        new_points[idx] += delta
        norm_idx = np.linalg.norm(new_points[idx])
        if norm_idx > 1.0:
            new_points[idx] = new_points[idx] / norm_idx * 0.999
        diffs = new_points - new_points[idx]
        dists = np.linalg.norm(diffs, axis=1)
        dists[idx] = np.inf
        min_pair = np.min(dists)
        min_pair_r = min_pair / 2.0
        new_norms = norms.copy()
        new_norms[idx] = np.linalg.norm(new_points[idx])
        new_min_boundary = 1.0 - np.max(new_norms)
        new_score = min(min_pair_r, new_min_boundary)
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-12)):
            current_points = new_points
            current_score = new_score
            norms = new_norms
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        T *= cooling
        if T < T_min:
            break
    return best_points, best_score

def _parametric_points(radii, angles):
    """Build 50 points from ring parameters: 1+6+12+18+13=50."""
    ring_sizes = [1, 6, 12, 18, 13]
    points = []
    points.append([0.0, 0.0])
    for ring_idx in range(1, 5):
        size = ring_sizes[ring_idx]
        r = radii[ring_idx - 1]
        angle_offset = angles[ring_idx - 1]
        for i in range(size):
            angle = 2 * np.pi * i / size + angle_offset
            points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)

def _parametric_objective(params):
    """Objective for parametric optimization: negative soft-min radius."""
    radii = params[:4]
    angles = params[4:8]
    points = _parametric_points(radii, angles)
    norms = np.linalg.norm(points, axis=1)
    if np.max(norms) > 1.0:
        return 100.0
    radius = _soft_min_surrogate(points, beta=50.0)
    return -radius

def _optimize_parametric(start_time, deadline, min_time_remaining=25.0):
    """Optimize ring radii and angles using L-BFGS from multiple starts."""
    bounds = [
        (0.15, 0.40), (0.35, 0.65), (0.55, 0.85), (0.75, 0.98),
        (0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi),
    ]
    # Multiple starting points
    rng = np.random.default_rng(42)
    starts = [np.array([0.25, 0.50, 0.75, 0.90, 0.0, np.pi/12, 0.0, np.pi/13])]
    for _ in range(5):
        s = np.array([
            rng.uniform(0.20, 0.35), rng.uniform(0.40, 0.60),
            rng.uniform(0.60, 0.80), rng.uniform(0.80, 0.95),
            rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi),
            rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
        ])
        starts.append(s)
    best_points = None
    best_score = -np.inf
    for s in starts:
        if time.time() > deadline - min_time_remaining:
            break
        try:
            result = minimize(
                _parametric_objective, s, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            best_params = result.x
            pts = _parametric_points(best_params[:4], best_params[4:8])
            sc = _compute_score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        except Exception:
            pass
    if best_points is not None and time.time() < deadline - min_time_remaining + 10:
        best_points, best_score = _polish_smooth(best_points, start_time, deadline, min_time_remaining=min_time_remaining - 10)
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    # Phase 1: Parametric optimization (low-dimensional, fast)
    param_points, param_score = _optimize_parametric(start_time, deadline, min_time_remaining=25.0)
    if param_points is not None and param_score > best_score:
        best_score = param_score
        best_points = param_points.copy()

    # Phase 2: Try multiple initializations
    initializations = [
        ("ring", _ring_initialization()),
        ("hex", _hexagonal_initial()),
        ("rand0", _random_initial(0)),
        ("rand1", _random_initial(1)),
        ("rand2", _random_initial(2)),
    ]
    for name, init_points in initializations:
        if time.time() > deadline - 60:
            break
        score = _compute_score(init_points)
        if score > best_score:
            best_score = score
            best_points = init_points.copy()

    # Phase 3: Simulated annealing
    if time.time() < deadline - 30:
        sa_points, sa_score = _simulated_annealing(best_points, start_time, deadline, min_time_remaining=30.0)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()

    # Phase 4: Smooth polish
    if time.time() < deadline - 15:
        best_points, best_score = _polish_smooth(best_points, start_time, deadline, min_time_remaining=15.0)

    # Phase 5: Exact polish
    if time.time() < deadline - 3:
        best_points, best_score = _polish_exact(best_points, start_time, deadline, min_time_remaining=3.0)

    # Final safety check: ensure all points are inside unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] /= norms[mask, np.newaxis] * 1.00001

    return best_points
# EVOLVE-BLOCK-END
