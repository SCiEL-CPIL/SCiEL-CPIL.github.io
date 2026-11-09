import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _min_triangle_area(points):
    """Vectorized minimum triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 0.0

def _score(points):
    """Score: min triangle area / convex hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_a = _min_triangle_area(points)
    if min_a == float('inf'):
        return 0.0
    return min_a / hull_a

def _negative_score(flat_points):
    """Negative score for minimization."""
    pts = flat_points.reshape(N, 2)
    return -_score(pts)

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles) + center[0]
    y = radius * np.sin(angles) + center[1]
    return np.column_stack((x, y))

def _make_12gon_center():
    """12-gon on unit circle + center point."""
    pts = _regular_polygon(12, radius=1.0)
    center = np.array([[0.0, 0.0]])
    return np.vstack([pts, center])

def _optimize_from_start(start_pts, max_iter=3000, deadline=None):
    """Optimize from a starting configuration using Nelder-Mead."""
    try:
        res = minimize(
            _negative_score,
            start_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if deadline is not None and time.time() >= deadline:
            return None
        return res.x.reshape(N, 2)
    except Exception:
        return None

def _powell_refine(start_pts, max_iter=1000, deadline=None):
    """Powell refinement."""
    try:
        res = minimize(
            _negative_score,
            start_pts.flatten(),
            method='Powell',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if deadline is not None and time.time() >= deadline:
            return None
        return res.x.reshape(N, 2)
    except Exception:
        return None

def _gradient_ascent(points, n_iter=500, step_size=0.01, deadline=None):
    """
    Gradient ascent on the score function using finite differences.
    This helps escape local optima that Nelder-Mead gets stuck in.
    """
    pts = points.copy()
    for it in range(n_iter):
        if deadline is not None and time.time() >= deadline:
            break
        
        current_score = _score(pts)
        
        # Compute gradient via finite differences
        grad = np.zeros_like(pts)
        h = 1e-5
        for i in range(N):
            for d in range(2):
                pts_plus = pts.copy()
                pts_plus[i, d] += h
                score_plus = _score(pts_plus)
                
                pts_minus = pts.copy()
                pts_minus[i, d] -= h
                score_minus = _score(pts_minus)
                
                grad[i, d] = (score_plus - score_minus) / (2 * h)
        
        # Normalize gradient to avoid steps that are too large
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Take a step
        pts += step_size * grad
        
        # Decay step size slowly
        step_size *= 0.999
    
    return pts

def _simulated_annealing(points, n_iter=5000, t_start=0.1, t_end=0.001, deadline=None):
    """
    Simulated annealing to escape local optima.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(n_iter):
        if deadline is not None and time.time() >= deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Propose a random perturbation
        perturbation = np.random.randn(N, 2) * t
        new_pts = pts + perturbation
        
        new_score = _score(new_pts)
        current_score = _score(pts)
        
        # Accept or reject
        if new_score > current_score or np.random.rand() < np.exp((new_score - current_score) / max(t, 1e-10)):
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with 12-gon + center (better topology for N=13)
    initial_points = _make_12gon_center()
    best_score = _score(initial_points)
    best_points = initial_points.copy()
    
    # Also try regular 13-gon as alternative
    reg13 = _regular_polygon(13, radius=1.0)
    s13 = _score(reg13)
    if s13 > best_score:
        best_score = s13
        best_points = reg13.copy()
    
    # Optimize from best initial using Nelder-Mead
    if time.time() < deadline - 5:
        opt = _optimize_from_start(best_points, max_iter=5000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    # Multi-restart with different strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Perturbed 12-gon + center with Nelder-Mead
    for trial in range(5):
        if time.time() >= deadline - 10:
            break
        noise_scale = 0.05 + trial * 0.03
        start_pts = _make_12gon_center() + rng.normal(0, noise_scale, (N, 2))
        opt = _optimize_from_start(start_pts, max_iter=2000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    # Strategy 2: Powell refinement on current best
    if time.time() < deadline - 15:
        refined = _powell_refine(best_points, max_iter=1000, deadline=deadline)
        if refined is not None:
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        
        # Optimize from refined with Nelder-Mead
        if time.time() < deadline - 5:
            opt = _optimize_from_start(best_points, max_iter=3000, deadline=deadline)
            if opt is not None:
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_points = opt.copy()
    
    # Strategy 3: Simulated annealing to escape local optima
    if time.time() < deadline - 20:
        sa_result = _simulated_annealing(best_points, n_iter=3000, t_start=0.05, t_end=0.001, deadline=deadline)
        s = _score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()
        
        # Refine with Nelder-Mead after SA
        if time.time() < deadline - 5:
            opt = _optimize_from_start(best_points, max_iter=3000, deadline=deadline)
            if opt is not None:
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_points = opt.copy()
    
    # Strategy 4: Gradient ascent to escape local optima
    if time.time() < deadline - 15:
        ga_result = _gradient_ascent(best_points, n_iter=300, step_size=0.005, deadline=deadline)
        s = _score(ga_result)
        if s > best_score:
            best_score = s
            best_points = ga_result.copy()
        
        # Refine with Nelder-Mead after gradient ascent
        if time.time() < deadline - 5:
            opt = _optimize_from_start(best_points, max_iter=3000, deadline=deadline)
            if opt is not None:
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_points = opt.copy()
    
    # Strategy 5: Random restarts with larger perturbations
    for trial in range(3):
        if time.time() >= deadline - 10:
            break
        start_pts = best_points + rng.normal(0, 0.15 + trial * 0.05, (N, 2))
        opt = _optimize_from_start(start_pts, max_iter=2000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    # Final refinement: Powell then Nelder-Mead
    if time.time() < deadline - 10:
        refined = _powell_refine(best_points, max_iter=1000, deadline=deadline)
        if refined is not None:
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        
        opt = _optimize_from_start(best_points, max_iter=2000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    return best_points
