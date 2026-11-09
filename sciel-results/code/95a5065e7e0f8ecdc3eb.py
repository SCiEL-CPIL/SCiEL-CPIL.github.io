# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

# Precompute all triangle index combinations (286 triangles for 13 points)
N_POINTS = 13
N_TRI = 286
_TRI_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)), dtype=np.int32)

def _triangle_areas(points):
    """Compute areas of all 286 triangles (vectorized)."""
    p1 = points[_TRI_IDX[:, 0]]
    p2 = points[_TRI_IDX[:, 1]]
    p3 = points[_TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def compute_score(points):
    """Compute the Heilbronn score: A_min / A_hull"""
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 0:
        return 0.0
    areas = _triangle_areas(points)
    return float(np.min(areas) / A_hull)

def _soft_min_score(points, beta=50.0):
    """Differentiable surrogate: soft-min of areas / hull area.
    Higher beta → closer to true min. Returns negative (for minimization)."""
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    if A_hull <= 0:
        return 1e10
    areas = _triangle_areas(points)
    # Soft min: -log(sum(exp(-beta*a)))/beta ≈ min(a) for large beta
    m = np.min(areas)
    soft_min = m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta
    return -soft_min / A_hull

def _nelder_mead(points, start_time, deadline, max_iter=3000):
    """Nelder-Mead optimization on the soft-min surrogate."""
    x0 = points.flatten()
    
    def obj(x):
        if time.time() > deadline - 5:
            return 1e10
        pts = x.reshape(N_POINTS, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        return _soft_min_score(pts, beta=50.0)
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True,
            }
        )
        return result.x.reshape(N_POINTS, 2)
    except Exception:
        return points

def _nelder_mead_hard(points, start_time, deadline, max_iter=2000):
    """Nelder-Mead on the hard objective (actual min area / hull area)."""
    x0 = points.flatten()
    
    def obj(x):
        if time.time() > deadline - 5:
            return 1e10
        pts = x.reshape(N_POINTS, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = compute_score(pts)
        return -s if s > 0 else 1e10
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-7,
                'fatol': 1e-9,
                'adaptive': True,
            }
        )
        return result.x.reshape(N_POINTS, 2)
    except Exception:
        return points

def _delaunay_repair(points, n_iter=200):
    """Move vertices of small triangles toward adjacent centroids to increase min area."""
    from scipy.spatial import Delaunay
    
    for _ in range(n_iter):
        try:
            tri = Delaunay(points)
        except Exception:
            break
        
        # Compute areas of all triangles
        areas = _triangle_areas(points)
        
        # Find the smallest triangles
        n_small = min(10, len(areas))
        small_idx = np.argpartition(areas, n_small)[:n_small]
        
        # For each small triangle, move its vertices slightly away
        moved = False
        for si in small_idx:
            t = _TRI_IDX[si]
            a, b, c = t
            # Centroid of the triangle
            centroid = (points[a] + points[b] + points[c]) / 3.0
            # Move each vertex away from centroid (increases area)
            for idx in [a, b, c]:
                direction = points[idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    points[idx] += 0.005 * direction / norm
            moved = True
        
        if not moved:
            break
    
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points maximizing A_min / A_hull.
    Uses Nelder-Mead with soft-min surrogate, Delaunay repair, and multi-start.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    def time_left():
        return deadline - time.time()
    
    # --- Initial configurations ---
    
    def make_config1():
        """Regular 12-gon + center"""
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([outer, center])
    
    def make_config2():
        """Two concentric rings (7 outer, 6 inner)"""
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        outer = np.column_stack([np.cos(a1), np.sin(a1)])
        inner = np.column_stack([0.55 * np.cos(a2), 0.55 * np.sin(a2)])
        return np.vstack([outer, inner])
    
    def make_config3():
        """Three rings (5, 4, 4)"""
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p3 = np.column_stack([0.3 * np.cos(a3), 0.3 * np.sin(a3)])
        return np.vstack([p1, p2, p3])
    
    def make_config4():
        """1 center, 6 at r=0.5, 6 at r=1.0"""
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.5 * np.cos(a1), 0.5 * np.sin(a1)])
        p3 = np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([p1, p2, p3])
    
    def make_config5():
        """Perturbed regular 13-gon"""
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        rng = np.random.default_rng(42)
        points += rng.normal(0, 0.03, size=points.shape)
        return points
    
    def make_config6():
        """1 center, 3 at r=0.3, 3 at r=0.6, 6 at r=1.0"""
        a1 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
        a3 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.3 * np.cos(a1), 0.3 * np.sin(a1)])
        p3 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p4 = np.column_stack([np.cos(a3), np.sin(a3)])
        return np.vstack([p1, p2, p3, p4])
    
    initial_configs = [
        make_config1(),
        make_config2(),
        make_config3(),
        make_config4(),
        make_config5(),
        make_config6(),
    ]
    
    # Add random starts
    rng = np.random.default_rng(7)
    for _ in range(4):
        pts = rng.uniform(-1, 1, size=(N_POINTS, 2))
        initial_configs.append(pts)
    
    best_points = None
    best_score = 0.0
    
    # --- Phase 1: Evaluate and refine each initial config ---
    for config in initial_configs:
        if time_left() < 30:
            break
        
        points = config.copy()
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Delaunay repair
        if time_left() > 30:
            points = _delaunay_repair(points, n_iter=100)
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # Nelder-Mead with soft-min surrogate
        if time_left() > 60:
            points = _nelder_mead(points, start_time, deadline, max_iter=2000)
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # --- Phase 2: Refine best with hard objective Nelder-Mead ---
    if best_points is not None and time_left() > 60:
        refined = _nelder_mead_hard(best_points, start_time, deadline, max_iter=2000)
        score = compute_score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
    
    # --- Phase 3: More Nelder-Mead rounds from best ---
    if best_points is not None and time_left() > 120:
        for _ in range(3):
            if time_left() < 30:
                break
            refined = _nelder_mead(best_points, start_time, deadline, max_iter=1500)
            score = compute_score(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
            
            # Also try hard objective
            if time_left() > 60:
                refined2 = _nelder_mead_hard(refined, start_time, deadline, max_iter=1000)
                score2 = compute_score(refined2)
                if score2 > best_score:
                    best_score = score2
                    best_points = refined2.copy()
    
    # --- Phase 4: Final polish with small perturbations around best ---
    if best_points is not None and time_left() > 30:
        rng_final = np.random.default_rng(123)
        for _ in range(50):
            if time_left() < 10:
                break
            new_pts = best_points + rng_final.normal(0, 0.02, size=(N_POINTS, 2))
            s = compute_score(new_pts)
            if s > best_score:
                best_score = s
                best_points = new_pts.copy()
    
    # Ensure validity
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = make_config1()
    
    try:
        hull = ConvexHull(best_points)
    except Exception:
        best_points = make_config1()
    
    return best_points

# EVOLVE-BLOCK-END
