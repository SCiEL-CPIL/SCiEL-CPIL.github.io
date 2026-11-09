# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

_N = 14


def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2


def _generate_tetrahedral_points(rv, rm, rc):
    """Generate 14 points based on tetrahedral symmetry orbits."""
    sqrt3 = np.sqrt(3)
    
    # Orbit 1: 4 Vertices of regular tetrahedron
    v_coords = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / sqrt3
    v_points = rv * v_coords
    
    # Orbit 2: 6 Edge Midpoints
    m_coords = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    m_points = rm * m_coords
    
    # Orbit 3: 4 Face Centroids
    c_coords = np.array([
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, -1, -1]
    ]) / sqrt3
    c_points = rc * c_coords
    
    return np.vstack([v_points, m_points, c_points])


def _neg_score_radii(params):
    """Objective: negative score for minimization of radii."""
    rv, rm, rc = params
    # Ensure positive radii for stability, though Nelder-Mead can handle signs
    # If negative, it just flips the direction, which is equivalent due to symmetry
    # But we want magnitudes.
    rv = abs(rv)
    rm = abs(rm)
    rc = abs(rc)
    
    if rv == 0 and rm == 0 and rc == 0:
        return 1.0
        
    points = _generate_tetrahedral_points(rv, rm, rc)
    if not np.all(np.isfinite(points)):
        return 1.0
        
    return -_score(points)


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses a high-symmetry combinatorial template (Tetrahedron Vertices, 
    Edge Midpoints, Face Centroids) and optimizes the 3 radial scaling factors.
    """
    n = _N
    deadline = time.time() + 580.0  # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Initial guesses for radii (rv, rm, rc)
    # 1. Equal radii
    # 2. Vertices larger
    # 3. Midpoints larger
    # 4. Centroids larger
    # 5. Random perturbations
    initial_guesses = [
        [1.0, 1.0, 1.0],
        [1.2, 1.0, 0.8],
        [1.0, 1.2, 0.8],
        [1.0, 1.0, 1.2],
        [1.5, 1.0, 0.5],
        [0.5, 1.5, 1.0],
        [1.0, 0.5, 1.5],
        [1.1, 0.9, 1.1],
        [0.9, 1.1, 0.9],
        [1.0, 1.1, 0.9]
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        initial_guesses.append(rng.uniform(0.5, 1.5, 3))
    
    for idx, guess in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            result = minimize(
                _neg_score_radii,
                guess,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'maxfev': 10000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                }
            )
            
            rv, rm, rc = result.x
            rv, rm, rc = abs(rv), abs(rm), abs(rc)
            
            points = _generate_tetrahedral_points(rv, rm, rc)
            
            if np.all(np.isfinite(points)):
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
        except Exception:
            continue

    # Fallback if optimization failed
    if best_points is None:
        best_points = _generate_tetrahedral_points(1.0, 1.0, 1.0)
    
    # Ensure shape and validity
    if best_points.shape != (n, 3) or not np.all(np.isfinite(best_points)):
        # Last resort: Fibonacci sphere
        golden_angle = np.pi * (3 - np.sqrt(5))
        thetas = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        phis = golden_angle * np.arange(n)
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        best_points = np.column_stack([x, y, z])

    return best_points


# EVOLVE-BLOCK-END
