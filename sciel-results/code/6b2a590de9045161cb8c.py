"""Heilbronn convex regions: 13 points to maximize minimum triangle area."""
import time
import numpy as np
from itertools import combinations

N = 13
DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

def _min_triangle_area(pts):
    """Minimum area over all C(13,3)=286 triangles."""
    n = len(pts)
    min_a = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = 0.5 * abs(pts[j, 0] * (pts[k, 1] - pts[i, 1]) +
                              pts[k, 0] * (pts[i, 1] - pts[j, 1]) +
                              pts[i, 0] * (pts[j, 1] - pts[k, 1]))
                if a < min_a:
                    min_a = a
    return min_a

def _hull_area(pts):
    """Area of convex hull via scipy."""
    from scipy.spatial import ConvexHull
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1e-10

def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha

def _fourier_to_points(params, m=3):
    """
    params: array of shape (2*(m+1) + 1,)
      - params[0]: center x
      - params[1]: center y
      - params[2]: scale (radius)
      - params[3:]: Fourier coefficients for radius modulation
        r(θ) = R * (1 + a1*cos(θ) + b1*sin(θ) + a2*cos(2θ) + b2*sin(2θ) + a3*cos(3θ) + b3*sin(3θ))
    Returns N points on the polygon (evenly spaced angles).
    """
    cx, cy, R = params[0], params[1], params[2]
    if R < 1e-6:
        R = 1e-6
    # Fourier coefficients for harmonics 1..m
    coeffs = params[3:]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    for h in range(1, m + 1):
        a = coeffs[2 * (h - 1)]
        b = coeffs[2 * (h - 1) + 1]
        radii += a * np.cos(h * angles) + b * np.sin(h * angles)
    # Ensure radii are positive (clamp to avoid self-intersection)
    radii = np.maximum(radii, 0.1)
    x = cx + R * radii * np.cos(angles)
    y = cy + R * radii * np.sin(angles)
    return np.column_stack([x, y])

def _params_to_score(params, m=3):
    pts = _fourier_to_points(params, m)
    return _score(pts)

def construct():
    m = 3
    n_params = 3 + 2 * m  # cx, cy, R, then 2*m Fourier coeffs
    best_score = 0.0
    best_params = np.zeros(n_params)
    best_params[2] = 1.0  # initial radius
    
    # Initial random starts
    rng = np.random.RandomState(42)
    
    # Start with regular polygon
    init = np.zeros(n_params)
    init[2] = 1.0
    best_params = init.copy()
    best_score = _params_to_score(best_params, m)
    
    # CMA-ES-like optimization using scipy minimize with multiple starts
    from scipy.optimize import minimize
    
    n_starts = 8
    for s in range(n_starts):
        if time.time() > DEADLINE:
            break
        # Generate initial parameters
        if s == 0:
            p0 = np.zeros(n_params)
            p0[2] = 1.0
        else:
            p0 = np.zeros(n_params)
            p0[2] = 1.0
            # Small random perturbations to Fourier coefficients
            p0[3:] = rng.normal(0, 0.1, size=2 * m)
        
        # Maximize score = minimize -score
        def neg_score(p):
            return -_params_to_score(p, m)
        
        try:
            res = minimize(neg_score, p0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if time.time() > DEADLINE:
                break
            sc = -res.fun
            if sc > best_score:
                best_score = sc
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final refinement with tighter tolerances
    if time.time() < DEADLINE - 5:
        try:
            def neg_score(p):
                return -_params_to_score(p, m)
            res = minimize(neg_score, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            sc = -res.fun
            if sc > best_score:
                best_score = sc
                best_params = res.x.copy()
        except Exception:
            pass
    
    pts = _fourier_to_points(best_params, m)
    
    # Verify convexity and return
    return pts

# EVOLVE-BLOCK-END
