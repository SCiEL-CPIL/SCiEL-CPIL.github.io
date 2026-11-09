"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 110.0  # safety margin below 120s limit


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_a = _hull_area(points)
    if hull_a <= 1e-12:
        return 0.0
    return min_ta / hull_a


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _objective(params):
    """Objective function for optimizer: negative of score."""
    points = params.reshape(N, 2)
    return _negative_score(points)


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    return pts


def construct():
    start_time = time.time()
    
    # Strategy: Explore the topology where 12 points form a convex hull 
    # and 1 point is inside (likely near the center).
    # This is distinct from the 13-gon topology.
    
    best_score = -1.0
    best_points = None

    # Initial guess: 12-gon on unit circle, center point at origin
    outer_12 = _regular_polygon(12, radius=1.0)
    center_pt = np.array([[0.0, 0.0]])
    points = np.vstack([outer_12, center_pt])
    
    # Ensure the center point is strictly inside or on the boundary
    # For 12-gon, origin is inside.
    
    # We will optimize the positions of all 13 points.
    # To maintain the "12 outer, 1 inner" structure, we can just let the optimizer work.
    # However, Nelder-Mead might move the center point out or outer points in.
    # A more robust approach for this specific topology is to fix the outer 12 as a regular polygon
    # (or optimize their radius/rotation) and optimize the center point.
    # But the problem allows any convex region. The hull is defined by the outer points.
    
    # Let's try a hybrid: 
    # 1. Optimize the center point with the outer 12 fixed as a regular 12-gon.
    # 2. Optimize the outer 12 points with the center point fixed.
    # 3. Joint optimization with Nelder-Mead.
    
    def score_current(pts):
        return _score(pts)
    
    # Phase 1: Optimize center point with fixed 12-gon
    # The score is invariant to rotation and scaling, but depends on shape.
    # A regular 12-gon is a strong candidate for the hull.
    # Let's just run a local search from this configuration using Nelder-Mead.
    
    x0 = points.flatten()
    
    # Nelder-Mead is derivative-free and can handle the non-smoothness of min-area/hull-area better than L-BFGS
    # in some cases, especially if we suspect discrete topological changes.
    
    try:
        # Use a tight initial simplex
        res = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        candidate = res.x.reshape(N, 2)
        c_score = score_current(candidate)
        
        if c_score > best_score:
            best_score = c_score
            best_points = candidate.copy()
            
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        if best_points is not None:
            return best_points
        return points
    
    # Phase 2: Try a different topology: 11-gon + 2 inner points?
    # Or just perturb the 12+1 solution more aggressively.
    
    # Let's try optimizing the radius of the 12-gon and the position of the center point
    # explicitly, as this reduces the dimensionality and might be faster/more robust.
    
    def _obj_2d(params):
        r = params[0]
        cx = params[1]
        cy = params[2]
        
        outer = _regular_polygon(12, radius=r)
        inner = np.array([[cx, cy]])
        pts = np.vstack([outer, inner])
        
        # Ensure inner is inside hull? 
        # If inner moves outside, the hull changes to 13 points.
        # The score function handles this naturally.
        
        return -_score(pts)
    
    # Initial guess: r=1, cx=0, cy=0
    x0_2d = np.array([1.0, 0.0, 0.0])
    
    try:
        res_2d = minimize(
            _obj_2d,
            x0_2d,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8}
        )
        
        r_opt = res_2d.x[0]
        cx_opt = res_2d.x[1]
        cy_opt = res_2d.x[2]
        
        outer = _regular_polygon(12, radius=r_opt)
        inner = np.array([[cx_opt, cy_opt]])
        candidate_2d = np.vstack([outer, inner])
        
        c_score_2d = score_current(candidate_2d)
        
        if c_score_2d > best_score:
            best_score = c_score_2d
            best_points = candidate_2d.copy()
            
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        if best_points is not None:
            return best_points
        return points

    # Phase 3: Full joint optimization from the best result found so far
    # Using Nelder-Mead again with a slightly perturbed start to escape local minima
    
    if best_points is not None:
        x0_full = best_points.flatten()
        # Add small noise
        rng = np.random.RandomState(42)
        x0_full += rng.normal(0, 0.01, x0_full.shape)
        
        try:
            res_full = minimize(
                _objective,
                x0_full,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            candidate_full = res_full.x.reshape(N, 2)
            c_score_full = score_current(candidate_full)
            
            if c_score_full > best_score:
                best_score = c_score_full
                best_points = candidate_full.copy()
                
        except Exception:
            pass
            
    # Final fallback
    if best_points is None:
        best_points = points
        
    return best_points
