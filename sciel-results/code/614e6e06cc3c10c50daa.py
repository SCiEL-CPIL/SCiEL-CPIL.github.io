"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas for points of shape (n,2)."""
    n = len(pts)
    areas = np.zeros((n * (n-1) * (n-2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # area = 0.5 * |cross(pj-pi, pk-pi)|
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                areas[idx] = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = np.array(pts)
    n = len(pts)
    if n < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    return areas.min() / hull_area

def _neg_score_smooth(pts, beta=100.0):
    """Smooth surrogate: -softmin(triangle_areas) / hull_area.
    softmin(a) = -(1/beta) * log(sum(exp(-beta * a_i)))
    This is differentiable and approximates min as beta -> inf."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas(pts)
    # softmin = -(1/beta) * log(sum(exp(-beta * areas)))
    # To avoid overflow, use log-sum-exp trick
    min_a = areas.min()
    log_sum = min_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - min_a))))
    softmin_val = log_sum
    return softmin_val / hull_area

def _construct_grid(seed_offset=0):
    """Structured grid-like initial configuration."""
    # 14 points: try a 4x4 grid minus corners, or 2x7, or other patterns
    # Let's try a distorted grid
    rng = np.random.RandomState(42 + seed_offset)
    # Start with a 4x4 grid, remove 2 points to get 14
    xs = np.linspace(0, 1, 4)
    ys = np.linspace(0, 1, 4)
    points = []
    for i in range(4):
        for j in range(4):
            points.append([xs[i], ys[j]])
    # Remove 2 random interior points to get 14
    indices = list(range(16))
    rng.shuffle(indices)
    remove = indices[:2]
    keep = [i for i in range(16) if i not in remove]
    pts = np.array([points[i] for i in keep])
    # Add small random perturbation
    pts += rng.uniform(-0.05, 0.05, size=pts.shape)
    return pts

def _construct_polygon(seed_offset=0):
    """Points on a regular polygon with some interior points."""
    rng = np.random.RandomState(100 + seed_offset)
    # 10 points on a circle, 4 interior
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    outer = np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
    # 4 interior points in a small grid
    inner = np.array([
        [0.35, 0.35], [0.65, 0.35],
        [0.35, 0.65], [0.65, 0.65]
    ])
    pts = np.vstack([outer, inner])
    pts += rng.uniform(-0.03, 0.03, size=pts.shape)
    return pts

def _construct_two_rings(seed_offset=0):
    """Two concentric rings of points."""
    rng = np.random.RandomState(200 + seed_offset)
    # 7 points on outer ring, 7 on inner ring
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([0.5 + 0.48 * np.cos(angles_outer), 0.5 + 0.48 * np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    inner = np.column_stack([0.5 + 0.25 * np.cos(angles_inner), 0.5 + 0.25 * np.sin(angles_inner)])
    pts = np.vstack([outer, inner])
    pts += rng.uniform(-0.03, 0.03, size=pts.shape)
    return pts

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_pts = None
    best_score = -1.0
    
    # Generate initial candidates
    initial_configs = []
    for i in range(5):
        initial_configs.append(_construct_grid(i))
    for i in range(5):
        initial_configs.append(_construct_polygon(i))
    for i in range(5):
        initial_configs.append(_construct_two_rings(i))
    
    # Also add a random one
    rng = np.random.RandomState(N)
    initial_configs.append(rng.uniform(0.1, 0.9, size=(N, 2)))
    
    # Evaluate initial scores
    for pts in initial_configs:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Local optimization from best few starts
    top_starts = []
    scored = [(_score(pts), i) for i, pts in enumerate(initial_configs)]
    scored.sort(key=lambda x: -x[0])
    top_starts = [initial_configs[i] for _, i in scored[:6]]
    
    for start_pts in top_starts:
        if time.time() > deadline - 20.0:
            break
        
        # Try different beta values for the smooth surrogate
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10.0:
                break
            
            def objective(x, b=beta):
                pts = x.reshape(N, 2)
                return _neg_score_smooth(pts, b)
            
            x0 = start_pts.flatten()
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-10})
                if time.time() > deadline - 5.0:
                    break
                opt_pts = res.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final polish: use exact min (not smooth) with tighter tolerance
    if best_pts is not None and time.time() < deadline - 5.0:
        def objective_exact(x):
            pts = x.reshape(N, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-12:
                return 1e10
            areas = _triangle_areas(pts)
            return -areas.min() / hull_area
        
        x0 = best_pts.flatten()
        try:
            res = minimize(objective_exact, x0, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-12})
            opt_pts = res.x.reshape(N, 2)
            s = _score(opt_pts)
            if s > best_score:
                best_score = s
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return grid
        best_pts = _construct_grid(0)
    
    return best_pts
