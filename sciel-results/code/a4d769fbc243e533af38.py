"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.spatial import Voronoi
import time

N = 26
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill


def _voronoi_inradii(points):
    """
    Compute the inradius of each Voronoi cell for points in the unit square.
    The inradius is the largest radius such that a circle centered at the point
    stays within the unit square AND doesn't cross any Voronoi edge.
    """
    n = len(points)
    # Pad with boundary points to capture the square boundary
    # We need points on the boundary so Voronoi cells are bounded by the square
    # Strategy: add many points just outside the square
    eps = 0.1
    boundary_pts = []
    # Bottom
    boundary_pts.extend([(i * 0.1 - eps, -eps) for i in range(12)])
    # Top
    boundary_pts.extend([(i * 0.1 - eps, 1 + eps) for i in range(12)])
    # Left
    boundary_pts.extend([(-eps, i * 0.1 - eps) for i in range(12)])
    # Right
    boundary_pts.extend([(1 + eps, i * 0.1 - eps) for i in range(12)])
    
    all_pts = np.vstack([points, np.array(boundary_pts)])
    
    try:
        vor = Voronoi(all_pts)
    except Exception:
        return np.zeros(n)
    
    radii = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Unbounded region - shouldn't happen with padding
            radii[i] = 0.0
            continue
        
        # For each edge in the region, compute distance from point to edge
        min_dist = np.inf
        for k in range(len(region)):
            v1 = vor.vertices[region[k]]
            v2 = vor.vertices[region[(k + 1) % len(region)]]
            # Distance from point i to line segment v1-v2
            d = _point_to_segment_dist(points[i], v1, v2)
            min_dist = min(min_dist, d)
        
        # Also check distance to square boundaries
        x, y = points[i]
        min_dist = min(min_dist, x, 1 - x, y, 1 - y)
        
        radii[i] = min_dist
    
    return radii


def _point_to_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    if np.dot(ab, ab) < 1e-15:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def _gradient_ascent(points, max_iter=2000, lr_init=0.01, lr_decay=0.999):
    """
    Gradient ascent on sum of Voronoi inradii.
    Uses finite differences for gradients.
    """
    points = points.copy().astype(float)
    n = len(points)
    total = 0.0
    best_points = points.copy()
    best_total = 0.0
    
    radii = _voronoi_inradii(points)
    total = np.sum(radii)
    best_total = total
    best_points = points.copy()
    
    lr = lr_init
    step = 0
    stalled = 0
    
    while step < max_iter:
        # Compute gradient via finite differences
        grad = np.zeros_like(points)
        h = 1e-5
        for i in range(n):
            for d in range(2):
                old = points[i, d]
                points[i, d] = old + h
                r_plus = _voronoi_inradii(points)
                points[i, d] = old - h
                r_minus = _voronoi_inradii(points)
                points[i, d] = old
                grad[i, d] = (np.sum(r_plus) - np.sum(r_minus)) / (2 * h)
        
        # Update
        points += lr * grad
        
        # Keep points inside [0.01, 0.99] to avoid boundary issues
        points = np.clip(points, 0.01, 0.99)
        
        radii = _voronoi_inradii(points)
        total = np.sum(radii)
        
        if total > best_total:
            best_total = total
            best_points = points.copy()
            stalled = 0
        else:
            stalled += 1
        
        lr *= lr_decay
        step += 1
        
        if stalled > 200:
            break
    
    return best_points, best_total


def _local_refinement(points):
    """
    Final polish: for each circle, try to push it away from neighbors.
    Iterative pairwise separation.
    """
    points = points.copy().astype(float)
    n = len(points)
    
    for iteration in range(500):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # If too close, push apart
                if dist < 0.05:
                    push = (0.05 - dist) * 0.5
                    direction = d / dist
                    points[i] += push * direction
                    points[j] -= push * direction
                    moved = True
        
        # Keep in bounds
        points = np.clip(points, 0.01, 0.99)
        
        if not moved:
            break
    
    return points


def _grid_init():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(cells[:N])


def _random_init(seed=42):
    """Random initialization with minimum spacing."""
    rng = np.random.RandomState(seed)
    points = []
    min_dist = 0.08
    attempts = 0
    while len(points) < N and attempts < 10000:
        p = rng.uniform(0.05, 0.95, 2)
        if all(np.linalg.norm(p - q) > min_dist for q in points):
            points.append(p)
        attempts += 1
    
    if len(points) < N:
        # Fill remaining randomly
        while len(points) < N:
            points.append(rng.uniform(0.05, 0.95, 2))
    
    return np.array(points)


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Multiple initializations
    inits = [
        _grid_init(),
        _random_init(42),
        _random_init(123),
        _random_init(456),
        _random_init(789),
    ]
    
    for init_idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 30:
            break
        
        # Gradient ascent
        try:
            refined_pts, score = _gradient_ascent(init_pts, max_iter=1000, lr_init=0.005, lr_decay=0.9995)
        except Exception:
            continue
        
        if score > best_score:
            best_score = score
            best_points = refined_pts.copy()
        
        # Local refinement
        if best_points is not None:
            refined = _local_refinement(best_points)
            radii = _voronoi_inradii(refined)
            score2 = np.sum(radii)
            if score2 > best_score:
                best_score = score2
                best_points = refined.copy()
    
    # If we still have time, do another round of gradient ascent from best
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < DEADLINE_SECONDS - 20:
        try:
            refined_pts, score = _gradient_ascent(best_points, max_iter=500, lr_init=0.002, lr_decay=0.9998)
            if score > best_score:
                best_score = score
                best_points = refined_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to grid
        best_points = _grid_init()
    
    # Compute final radii
    radii = _voronoi_inradii(best_points)
    
    # Safety: ensure no overlap and within bounds
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_points[i, 0]
        result[i, 1] = best_points[i, 1]
        result[i, 2] = radii[i]
    
    # Verify and fix any issues
    for i in range(N):
        x, y, r = result[i]
        # Clamp to unit square
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 1e-6)
        result[i, 2] = r
    
    # Final overlap check and shrink if needed
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i, :2] - result[j, :2])
            min_r = max(0, (d - 1e-6) / 2)
            if result[i, 2] + result[j, 2] > d:
                # Shrink both
                total_r = result[i, 2] + result[j, 2]
                scale = d / total_r * 0.999
                result[i, 2] *= scale
                result[j, 2] *= scale
    
    return result
