"""Optimized program for Heilbronn convex regions n=14."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - 
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    points = np.array(points)
    # Sort by angle around centroid
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    order = np.argsort(angles)
    points = points[order]
    
    # Graham scan
    hull = []
    for p in points:
        while len(hull) >= 2:
            p1 = hull[-2]
            p2 = hull[-1]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    
    # Compute area of hull
    hull = np.array(hull)
    n_hull = len(hull)
    if n_hull < 3:
        return 0.0
    area = 0.0
    for i in range(n_hull):
        j = (i + 1) % n_hull
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return 0.5 * abs(area)

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _soft_min_triangle_area(points, beta=1000.0):
    """Smooth surrogate for min triangle area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - 
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                areas.append(area)
    areas = np.array(areas)
    # Use log-sum-exp for soft min
    max_a = np.max(areas)
    log_sum = max_a + np.log(np.sum(np.exp(-areas / beta)))
    return -log_sum  # This is -beta * log(sum(exp(-area/beta))) ~ -beta * log(K) + min_area

def _neg_score_smooth(params, beta=1000.0):
    """Negative of smooth score for minimization."""
    points = np.array(params).reshape(N, 2)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    min_tri_smooth = _soft_min_triangle_area(points, beta)
    # The soft min approximates min_tri, so score ~ min_tri / hull_area
    # We want to maximize this, so minimize -min_tri/hull_area
    return -min_tri_smooth / hull_area

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Start with regular polygon and optimize
    def regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Initialize with regular polygon
    points = regular_polygon(N)
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Strategy 2: Differential evolution on the full 28 parameters
    # But this is expensive, so let's try a few local optimizations from different starts
    
    # Multiple restarts with local optimization
    rng = np.random.RandomState(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        if restart == 0:
            x0 = regular_polygon(N).flatten()
        elif restart == 1:
            # Slightly perturbed regular polygon
            x0 = (regular_polygon(N) + 0.1 * rng.randn(N, 2)).flatten()
        elif restart == 2:
            # Random points in unit square, then scale
            x0 = (rng.rand(N, 2) - 0.5).flatten()
        elif restart == 3:
            # Elliptical distribution
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            x0 = np.column_stack([1.0 * np.cos(angles), 0.8 * np.sin(angles)]).flatten()
        else:
            # Another perturbation
            x0 = (regular_polygon(N) + 0.2 * rng.randn(N, 2)).flatten()
        
        # Local optimization using Nelder-Mead on the exact score
        def neg_score_exact(params):
            pts = np.array(params).reshape(N, 2)
            return -_score(pts)
        
        try:
            result = minimize(neg_score_exact, x0, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.success or True:  # Accept even if not fully converged
                new_score = -result.fun
                if new_score > best_score:
                    best_score = new_score
                    best_points = result.x.reshape(N, 2).copy()
        except Exception:
            pass
        
        # If we have time, try a second pass with tighter tolerance
        if time.time() - start_time > deadline - 20:
            break
        
        # Polish with even tighter Nelder-Mead
        def neg_score_exact_tight(params):
            pts = np.array(params).reshape(N, 2)
            return -_score(pts)
        
        try:
            result2 = minimize(neg_score_exact_tight, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result2.success or True:
                new_score2 = -result2.fun
                if new_score2 > best_score:
                    best_score = new_score2
                    best_points = result2.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Strategy 3: Try Fourier parameterization with 3 coefficients
    # r(theta) = 1 + c1*cos(theta) + c2*sin(theta) + c3*cos(2theta)
    # This maintains convexity if coefficients are small enough
    if time.time() - start_time < deadline - 10:
        def fourier_points(coeffs):
            c1, c2, c3 = coeffs
            thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
            r = 1.0 + c1 * np.cos(thetas) + c2 * np.sin(thetas) + c3 * np.cos(2*thetas)
            # Ensure r > 0 for convexity
            r = np.maximum(r, 0.5)
            return np.column_stack([r * np.cos(thetas), r * np.sin(thetas)])
        
        def neg_score_fourier(params):
            pts = fourier_points(params)
            return -_score(pts)
        
        # Optimize the 3 Fourier coefficients
        x0_fourier = np.array([0.0, 0.0, 0.0])
        try:
            result_f = minimize(neg_score_fourier, x0_fourier, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result_f.success or True:
                new_score_f = -result_f.fun
                if new_score_f > best_score:
                    best_score = new_score_f
                    best_points = fourier_points(result_f.x).copy()
        except Exception:
            pass
    
    # Final polish on the best points found
    if time.time() - start_time < deadline - 5 and best_points is not None:
        def neg_score_final(params):
            pts = np.array(params).reshape(N, 2)
            return -_score(pts)
        
        try:
            result_final = minimize(neg_score_final, best_points.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 50000, 'xatol': 1e-13, 'fatol': 1e-15})
            if result_final.success or True:
                new_score_final = -result_final.fun
                if new_score_final > best_score:
                    best_score = new_score_final
                    best_points = result_final.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Ensure we return something valid
    if best_points is None:
        best_points = regular_polygon(N)
    
    return best_points

# EVOLVE-BLOCK-END
