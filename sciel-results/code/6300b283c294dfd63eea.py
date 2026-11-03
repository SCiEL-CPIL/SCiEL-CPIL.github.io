"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
from itertools import combinations
import sys

def _hull_area(points):
    """Compute convex hull area of a set of 2D points."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of points."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return float('inf')
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score: min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    if min_area == float('inf') or min_area <= 0:
        return 0.0
    return min_area / hull_area

def _objective_neg(points_flat):
    """Negative of the score for minimization."""
    points = points_flat.reshape(-1, 2)
    score = _score(points)
    return -score

def _initialize_points():
    """Initialize points in a reasonable configuration."""
    # Start with points on a regular 14-gon, then add some perturbation
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Add small random perturbation
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.1, size=points.shape)
    # Normalize to unit area
    hull_area = _hull_area(points)
    if hull_area > 0:
        scale = np.sqrt(1.0 / hull_area)
        points *= scale
    return points

def construct():
    """Construct 14 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = 110.0  # 10 seconds safety margin
    
    # Initialize
    best_points = _initialize_points()
    best_score = _score(best_points)
    
    # Local optimization using Nelder-Mead
    x0 = best_points.flatten()
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            _objective_neg,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        candidate_points = result.x.reshape(-1, 2)
        candidate_score = _score(candidate_points)
        
        if candidate_score > best_score:
            best_points = candidate_points
            best_score = candidate_score
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline:
        return best_points
    
    # Try multiple restarts with different initializations
    rng = np.random.default_rng(123)
    for restart in range(5):
        if time.time() - start_time > deadline:
            break
        
        # Different initial configurations
        if restart == 0:
            # Regular polygon with larger perturbation
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
            points += rng.normal(0, 0.3, size=points.shape)
        elif restart == 1:
            # Two rings
            angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            angles_outer = np.linspace(np.pi/7, 2 * np.pi, 7, endpoint=False)
            inner = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
            outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            points = np.vstack([inner, outer])
            points += rng.normal(0, 0.1, size=points.shape)
        elif restart == 2:
            # Three rings
            angles1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
            angles2 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
            angles3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            ring1 = np.column_stack([0.3 * np.cos(angles1), 0.3 * np.sin(angles1)])
            ring2 = np.column_stack([0.6 * np.cos(angles2), 0.6 * np.sin(angles2)])
            ring3 = np.column_stack([np.cos(angles3), np.sin(angles3)])
            points = np.vstack([ring1, ring2, ring3])
            points += rng.normal(0, 0.05, size=points.shape)
        elif restart == 3:
            # Random initialization
            points = rng.uniform(-1, 1, size=(N, 2))
        else:
            # Use best points with small perturbation
            points = best_points + rng.normal(0, 0.05, size=best_points.shape)
        
        # Normalize to unit area
        hull_area = _hull_area(points)
        if hull_area > 0:
            scale = np.sqrt(1.0 / hull_area)
            points *= scale
        
        # Local optimization
        try:
            result = minimize(
                _objective_neg,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            candidate_points = result.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
