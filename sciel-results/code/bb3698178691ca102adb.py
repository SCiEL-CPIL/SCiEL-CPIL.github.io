"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            diffs = pts - pts[i]
            crosses = np.abs(diffs[:, 0] * v1[1] - diffs[:, 1] * v1[0])
            crosses[i] = float('inf')
            crosses[j] = float('inf')
            local_min = np.min(crosses)
            if local_min < min_area:
                min_area = local_min
    return 0.5 * min_area


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a


def _neg_score(pts):
    return -_score(pts)


def _regular_polygon_init(n, center, radius, rotation=0.0):
    """Create points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    pts = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return pts


def _two_ring_init(n_inner, n_outer, center, r_inner, r_outer, rot_inner=0.0, rot_outer=0.0):
    """Create points on two concentric rings."""
    pts = []
    if n_inner > 0:
        pts.append(_regular_polygon_init(n_inner, center, r_inner, rot_inner))
    if n_outer > 0:
        pts.append(_regular_polygon_init(n_outer, center, r_outer, rot_outer))
    if pts:
        return np.vstack(pts)
    return np.array([center])


def _optimize_from_init(pts0, deadline):
    """Run L-BFGS-B optimization from initial point."""
    x0 = pts0.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        if time.time() > deadline:
            return 0.0
        return _neg_score(pts)
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'gtol': 1e-10})
        return result.x.reshape(-1, 2), _score(result.x.reshape(-1, 2))
    except Exception:
        return pts0, _score(pts0)


def _local_search(pts, deadline, n_iter=50, step_size=0.01):
    """Simple local search with random perturbations."""
    best_pts = pts.copy()
    best_score = _score(pts)
    
    rng = np.random.RandomState(123)
    
    for _ in range(n_iter):
        if time.time() > deadline:
            break
        
        # Try moving each point slightly
        for i in range(N):
            if time.time() > deadline:
                break
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = best_pts.copy()
                    new_pts[i, 0] += dx
                    new_pts[i, 1] += dy
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts.copy()
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_pts = None
    best_score = -1.0
    
    # Generate diverse initializations
    inits = []
    center = np.array([0.5, 0.5])
    
    # Configuration 1: 1 center + 12 outer
    for radius in [0.4, 0.45, 0.5]:
        for rot in [0.0, np.pi / 12]:
            pts = _two_ring_init(0, 12, center, 0, radius, 0, rot)
            inits.append(pts)
    
    # Configuration 2: 1 center + 6 inner + 6 outer
    for r_in in [0.2, 0.25]:
        for r_out in [0.4, 0.45, 0.5]:
            for rot_in in [0.0, np.pi / 6]:
                for rot_out in [0.0, np.pi / 12]:
                    pts = _two_ring_init(6, 6, center, r_in, r_out, rot_in, rot_out)
                    inits.append(pts)
    
    # Configuration 3: 1 center + 4 inner + 8 outer
    for r_in in [0.2, 0.25]:
        for r_out in [0.45, 0.5]:
            pts = _two_ring_init(4, 8, center, r_in, r_out, 0, np.pi / 8)
            inits.append(pts)
    
    # Configuration 4: 1 center + 5 inner + 7 outer
    for r_in in [0.2, 0.25]:
        for r_out in [0.45, 0.5]:
            pts = _two_ring_init(5, 7, center, r_in, r_out, 0, np.pi / 14)
            inits.append(pts)
    
    # Configuration 5: 1 center + 3 inner + 9 outer
    for r_in in [0.2, 0.25]:
        for r_out in [0.45, 0.5]:
            pts = _two_ring_init(3, 9, center, r_in, r_out, 0, np.pi / 9)
            inits.append(pts)
    
    # Configuration 6: 1 center + 2 inner + 10 outer
    for r_in in [0.2, 0.25]:
        for r_out in [0.45, 0.5]:
            pts = _two_ring_init(2, 10, center, r_in, r_out, 0, np.pi / 10)
            inits.append(pts)
    
    # Configuration 7: 1 center + 1 inner + 11 outer
    for r_in in [0.2, 0.25]:
        for r_out in [0.45, 0.5]:
            pts = _two_ring_init(1, 11, center, r_in, r_out, 0, np.pi / 11)
            inits.append(pts)
    
    # Add some perturbed versions for diversity
    rng = np.random.RandomState(42)
    for base in inits[:10]:
        perturbed = base + rng.randn(13, 2) * 0.02
        inits.append(perturbed)
    
    # Try each initialization
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        try:
            # First try L-BFGS-B
            optimized_pts, score = _optimize_from_init(init_pts, deadline)
            if score > best_score:
                best_score = score
                best_pts = optimized_pts
            
            # Then try local search from the optimized point
            if time.time() < deadline:
                ls_pts, ls_score = _local_search(optimized_pts, deadline, n_iter=20, step_size=0.005)
                if ls_score > best_score:
                    best_score = ls_score
                    best_pts = ls_pts
        except Exception:
            continue
    
    # If no good result, fall back to best initialization
    if best_pts is None:
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
