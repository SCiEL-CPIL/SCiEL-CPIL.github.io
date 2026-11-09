# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _all_triangle_indices(n):
    """Precompute all C(n,3) triangle index triples."""
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return None

def _score(pts):
    """Compute the score: min_triangle_area / hull_area."""
    try:
        n = len(pts)
        tri_idx = _all_triangle_indices(n)
        areas = _triangle_areas(pts, tri_idx)
        min_area = areas.min()
        hull_area = _hull_area(pts)
        if hull_area is None or hull_area <= 0:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _penalty_objective(x, tri_idx, n):
    """
    Objective function for L-BFGS-B: minimize negative min-area penalty.
    x is flattened (n, 2) points.
    We use a smooth approximation: -min_area + lambda * (hull_area - 1)^2
    to keep hull area near 1 while maximizing min triangle area.
    """
    pts = x.reshape(n, 2)
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    
    # Smooth min using soft-min
    beta = 20.0
    shifted = -beta * areas
    max_shifted = shifted.max()
    soft_min = -(max_shifted + np.log(np.exp(shifted - max_shifted).sum())) / beta
    
    # Hull area penalty to keep area ~ 1
    hull_area = _hull_area(pts)
    if hull_area is None:
        return 1e6  # invalid
    hull_penalty = (hull_area - 1.0) ** 2
    
    return -soft_min + 0.1 * hull_penalty

def _penalty_gradient(x, tri_idx, n):
    """
    Gradient of the penalty objective.
    """
    pts = x.reshape(n, 2)
    areas = _triangle_areas(pts, tri_idx)
    
    # Soft-min gradient
    beta = 20.0
    shifted = -beta * areas
    max_shifted = shifted.max()
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / exp_shifted.sum()
    
    # Signed areas for gradient direction
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    
    signed = (a[:, 0] * (b[:, 1] - c[:, 1]) +
              b[:, 0] * (c[:, 1] - a[:, 1]) +
              c[:, 0] * (a[:, 1] - b[:, 1]))
    
    signs = np.sign(signed)
    signs[signs == 0] = 1.0
    
    grad = np.zeros_like(pts)
    
    da = 0.5 * signs[:, None] * np.stack([-(b[:, 1] - c[:, 1]), b[:, 0] - c[:, 0]], axis=1)
    db = 0.5 * signs[:, None] * np.stack([-(c[:, 1] - a[:, 1]), c[:, 0] - a[:, 0]], axis=1)
    dc = 0.5 * signs[:, None] * np.stack([-(a[:, 1] - b[:, 1]), a[:, 0] - b[:, 0]], axis=1)
    
    w = weights[:, None]
    np.add.at(grad, tri_idx[:, 0], w * da)
    np.add.at(grad, tri_idx[:, 1], w * db)
    np.add.at(grad, tri_idx[:, 2], w * dc)
    
    # Gradient of -soft_min is -grad of soft_min, which is -sum(w_i * d(area_i)/d(pts))
    # But we want to MINIMIZE -soft_min, so gradient is -grad
    obj_grad = -grad
    
    # Hull area gradient (approximate via finite differences if needed)
    # For now, skip hull gradient and rely on penalty being small
    
    return obj_grad.ravel()

def _regular_polygon(n, r=1.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return r * np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _spiral_pattern(n, r=1.0):
    """Generate n points in a spiral pattern."""
    angles = np.linspace(0, 4*np.pi, n, endpoint=False)
    radii = np.linspace(0.3, r, n)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

def _concentric_rings(n):
    """Generate points on concentric rings."""
    # 3 points on inner ring, 10 on outer ring
    pts = []
    # Inner ring: 3 points
    for k in range(3):
        angle = 2 * np.pi * k / 3
        pts.append([0.4 * np.cos(angle), 0.4 * np.sin(angle)])
    # Outer ring: 10 points
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    return np.array(pts)

def _halton_sequence(n, dims=2):
    """Generate n Halton sequence points in [0,1]^dims."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, dims))
    for i in range(n):
        for j in range(dims):
            points[i, j] = halton(i + 1, j + 2)
    return points

def _optimize_lbfgsb(pts, tri_idx, n, time_limit=120.0):
    """
    Optimize using L-BFGS-B on the penalty objective.
    """
    start_time = time.time()
    x0 = pts.ravel()
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Try multiple beta values for the soft-min
    for beta in [15.0, 25.0, 40.0]:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x):
            pts_x = x.reshape(n, 2)
            areas = _triangle_areas(pts_x, tri_idx)
            shifted = -beta * areas
            max_shifted = shifted.max()
            soft_min = -(max_shifted + np.log(np.exp(shifted - max_shifted).sum())) / beta
            hull_area = _hull_area(pts_x)
            if hull_area is None:
                return 1e6
            hull_penalty = (hull_area - 1.0) ** 2
            return -soft_min + 0.1 * hull_penalty
        
        # Use numerical gradient (L-BFGS-B can handle this)
        result = minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 1e6:
            new_pts = result.x.reshape(n, 2)
            if np.all(np.isfinite(new_pts)):
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    x0 = new_pts.ravel()  # warm start
    
    return best_pts, best_score

def _local_refine(pts, tri_idx, n, time_limit=60.0):
    """
    Local refinement using coordinate descent on individual points.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Normalize hull area to 1
    hull_area = _hull_area(best_pts)
    if hull_area and hull_area > 0:
        best_pts = best_pts / np.sqrt(hull_area)
        best_score = _score(best_pts)
    
    # Coordinate descent: move each point slightly in various directions
    step_sizes = [0.01, 0.005, 0.001]
    
    for ss in step_sizes:
        if time.time() - start_time > time_limit:
            break
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                # Try 8 directions
                for dx in [-ss, 0, ss]:
                    for dy in [-ss, 0, ss]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = best_pts.copy()
                        trial[i, 0] += dx
                        trial[i, 1] += dy
                        trial_score = _score(trial)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_pts = trial.copy()
                            improved = True
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    n = 13
    tri_idx = _all_triangle_indices(n)
    
    deadline = time.time() + 600  # Safety margin before 630s kill
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Regular polygon
    starts.append(_regular_polygon(n, r=1.0))
    
    # 2. Spiral pattern
    starts.append(_spiral_pattern(n, r=1.0))
    
    # 3. Concentric rings
    starts.append(_concentric_rings(n))
    
    # 4. Halton sequence
    starts.append(_halton_sequence(n))
    
    # 5. Random starts with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        rng = np.random.default_rng(seed)
        starts.append(rng.random((n, 2)))
    
    # 6. Points near circle with perturbation
    for seed in [10, 11, 12, 13, 14]:
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.uniform(-0.15, 0.15, n)
        radii = 1.0 + rng.uniform(-0.15, 0.15, n)
        pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        starts.append(pts)
    
    best_pts = None
    best_score = -1.0
    
    # Optimize from each start
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        # Scale start to have hull area ~1 for consistency
        hull_area = _hull_area(start_pts)
        if hull_area and hull_area > 0:
            start_pts = start_pts / np.sqrt(hull_area)
        
        # Time per optimization
        remaining_starts = len(starts) - i
        time_per_opt = min(80.0, (deadline - time.time() - 30) / max(1, remaining_starts))
        if time_per_opt < 10.0:
            break
        
        # L-BFGS-B optimization
        pts, score = _optimize_lbfgsb(start_pts, tri_idx, n, time_limit=time_per_opt * 0.6)
        
        # Local refinement
        if time.time() < deadline - 30:
            pts, score = _local_refine(pts, tri_idx, n, time_limit=time_per_opt * 0.4)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final polish on the best solution
    if best_pts is not None and time.time() < deadline - 30:
        time_left = deadline - time.time() - 10
        best_pts, best_score = _optimize_lbfgsb(best_pts, tri_idx, n, time_limit=time_left * 0.5)
        if time.time() < deadline - 10:
            best_pts, best_score = _local_refine(best_pts, tri_idx, n, time_limit=time_left * 0.5)
    
    # Normalize: scale so hull area is exactly 1
    if best_pts is None:
        best_pts = _regular_polygon(n, r=1.0)
    
    hull_area = _hull_area(best_pts)
    if hull_area and hull_area > 0:
        best_pts = best_pts / np.sqrt(hull_area)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _regular_polygon(n, r=1.0)
    
    return best_pts

# EVOLVE-BLOCK-END
