"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _fourier_boundary(theta, a0, coeffs):
    """Compute radial boundary r(theta) from Fourier coefficients.
    coeffs = [a1, b1, a2, b2, a3, b3]
    """
    r = np.full_like(theta, a0, dtype=float)
    for k in range(1, 4):
        r += coeffs[2*(k-1)] * np.cos(k * theta) + coeffs[2*(k-1)+1] * np.sin(k * theta)
    return r

def _build_points(params):
    """Build 14 points from parameters.
    params[0] = a0 (base radius)
    params[1:7] = Fourier coefficients [a1,b1,a2,b2,a3,b3]
    params[7:21] = angle offsets for 14 points (each in [-0.5/N, 0.5/N] * 2pi)
    """
    a0 = params[0]
    coeffs = params[1:7]
    offsets = params[7:21]
    
    # Base angles evenly spaced
    base_angles = 2.0 * np.pi * np.arange(N) / N
    # Add offsets (offsets are in radians, small)
    angles = base_angles + offsets
    
    # Compute boundary radius at each angle
    r = _fourier_boundary(angles, a0, coeffs)
    
    # Ensure positive radius (clip to small positive)
    r = np.maximum(r, 0.1)
    
    # Convert to Cartesian
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    return np.column_stack([x, y])

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 0.0

def _score(params):
    """Compute score: min triangle area / convex hull area."""
    if _time_remaining() < 5.0:
        return -1.0  # Signal timeout
    
    points = _build_points(params)
    
    # Check for degeneracy
    if np.any(~np.isfinite(points)):
        return -1.0
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return -1.0
    
    min_area = _min_triangle_area(points)
    
    if min_area < 1e-15:
        return -1.0
    
    return min_area / hull_area

def _objective(params):
    """Objective for minimization (negate score)."""
    s = _score(params)
    if s < 0:
        return 1e6
    return -s

def construct():
    """Construct 14 points maximizing min triangle area / hull area."""
    # Parameter bounds
    # a0: base radius, should be around 1.0 for unit area
    # For a circle of radius R, area = pi*R^2 = 1 => R = 1/sqrt(pi) ~ 0.564
    # With Fourier terms, a0 should be close to this
    a0_bounds = (0.4, 1.0)
    
    # Fourier coefficients: keep them small to maintain convexity
    # r(theta) = a0 + sum, need r > 0 always
    # If a0 >= 0.5 and |coeffs| small, we're safe
    coeff_bounds = [(-0.15, 0.15)] * 6
    
    # Angle offsets: small perturbations from even spacing
    # Max offset = half the angular spacing = pi/N
    offset_bounds = [(-np.pi / N, np.pi / N)] * N
    
    bounds = [a0_bounds] + coeff_bounds + offset_bounds
    
    # Initial guess: circle with even spacing
    best_params = np.array([1.0/np.sqrt(np.pi)] + [0.0]*6 + [0.0]*N)
    best_score = _score(best_params)
    
    # Run differential evolution with time guard
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=True,
            callback=lambda xk, convergence: (None if _time_remaining() > 10.0 else None),
            # We'll handle timeout via the objective returning large values
        )
        
        if result.success or result.fun < 1e5:
            s = _score(result.x)
            if s > best_score:
                best_score = s
                best_params = result.x
    except Exception:
        pass
    
    # If time remains, try a few more random restarts or local refinement
    if _time_remaining() > 15.0:
        # Try a few random perturbations around best
        for _ in range(50):
            if _time_remaining() < 10.0:
                break
            perturbed = best_params + np.random.normal(0, 0.02, size=len(best_params))
            # Clip to bounds
            for i, (lo, hi) in enumerate(bounds):
                perturbed[i] = np.clip(perturbed[i], lo, hi)
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_params = perturbed
    
    # Final: build points from best_params
    points = _build_points(best_params)
    
    # Ensure points are finite
    if not np.all(np.isfinite(points)):
        # Fallback: regular 14-gon
        angles = 2.0 * np.pi * np.arange(N) / N
        r = 1.0 / np.sqrt(np.pi)
        points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    return points
# EVOLVE-BLOCK-END
