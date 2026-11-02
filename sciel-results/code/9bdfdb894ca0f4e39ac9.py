"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

N = 13


def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                ax = pts[j, 0] - pts[i, 0]
                ay = pts[j, 1] - pts[i, 1]
                bx = pts[k, 0] - pts[i, 0]
                by = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape((N, 2))
    return -_score_points(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_score = 0.0
    best_pts = None

    # Strategy 1: Use a structured initial guess - regular polygon-like arrangement
    # Place points on a regular polygon plus center
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_outer = 1.0
    # Mix of outer and inner points for better coverage
    # Try several structured configurations
    
    structured_configs = []
    
    # Config 1: Regular 13-gon
    pts1 = np.column_stack([np.cos(angles) * r_outer, np.sin(angles) * r_outer])
    structured_configs.append(pts1)
    
    # Config 2: Two concentric rings - 7 outer, 6 inner
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    inner = np.column_stack([np.cos(angles6) * 0.5, np.sin(angles6) * 0.5])
    pts2 = np.vstack([outer, inner])
    structured_configs.append(pts2)
    
    # Config 3: 5 outer, 8 inner in two sub-rings
    angles5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer5 = np.column_stack([np.cos(angles5), np.sin(angles5)])
    inner8 = np.column_stack([np.cos(angles8) * 0.4, np.sin(angles8) * 0.4])
    pts3 = np.vstack([outer5, inner8])
    structured_configs.append(pts3)
    
    # Config 4: Grid-like arrangement
    grid_pts = []
    # 4x4 grid minus one corner, centered
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_pts.append([i - 1.5, j - 1.5])
    pts4 = np.array(grid_pts)
    structured_configs.append(pts4)
    
    # Config 5: Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = []
    for i in range(N):
        r = 0.9 * np.sqrt((i + 0.5) / N)
        theta = i * golden_angle
        fib_pts.append([r * np.cos(theta), r * np.sin(theta)])
    pts5 = np.array(fib_pts)
    structured_configs.append(pts5)
    
    # Evaluate all structured configs
    for cfg in structured_configs:
        score = _score_points(cfg)
        if score > best_score:
            best_score = score
            best_pts = cfg.copy()
    
    # Check time
    if time.time() > deadline:
        return best_pts if best_pts is not None else structured_configs[0]

    # Strategy 2: Differential evolution from best structured config
    # Normalize to unit area for optimization
    if best_pts is not None:
        try:
            hull = ConvexHull(best_pts)
            hull_area = hull.volume
            if hull_area > 0:
                # Center and scale
                center = best_pts.mean(axis=0)
                scaled = (best_pts - center)
                # Scale so hull area is 1
                scale_factor = 1.0 / np.sqrt(hull_area)
                initial = (scaled * scale_factor).flatten()
            else:
                initial = best_pts.flatten()
        except Exception:
            initial = best_pts.flatten()
    else:
        initial = structured_configs[0].flatten()

    # Run differential evolution with time limit
    try:
        result = differential_evolution(
            _neg_score,
            bounds=[(-2, 2)] * (2 * N),
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='sobol'
        )
        
        if result.success or result.fun < -best_score:
            candidate_pts = result.x.reshape((N, 2))
            candidate_score = _score_points(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
    except Exception:
        pass

    if time.time() > deadline:
        return best_pts if best_pts is not None else structured_configs[0]

    # Strategy 3: Local refinement with multiple restarts
    if best_pts is not None:
        from scipy.optimize import minimize
        
        # Multiple local optimizations from perturbed starting points
        rng = np.random.default_rng(42)
        for restart in range(10):
            if time.time() > deadline:
                break
            
            if restart == 0:
                start = best_pts.copy()
            else:
                # Perturb best solution
                noise = rng.normal(0, 0.1, size=(N, 2))
                start = best_pts + noise
            
            # Optimize
            res = minimize(
                _neg_score,
                start.flatten(),
                method='L-BFGS-B',
                bounds=[(-3, 3)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.fun < -best_score:
                candidate_pts = res.x.reshape((N, 2))
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()

    # Final check
    if best_pts is None:
        best_pts = structured_configs[0]
    
    return best_pts
