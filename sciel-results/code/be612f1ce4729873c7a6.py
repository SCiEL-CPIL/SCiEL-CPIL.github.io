# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 40s safety margin for 630s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is a scalar: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    """
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 0.0
    a_min = _min_triangle_area(points)
    return a_min / a_hull

def _objective(points_flat):
    """
    Objective function for minimization: -score.
    Uses exact metric.
    """
    points = points_flat.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e6
    s = _score(points)
    if s <= 1e-12:
        return 1e6
    return -s

def _generate_regular_polygon(n=13, radius=1.0):
    """
    Generate a regular n-gon.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_hex_lattice(n=13, radius=1.2):
    """
    Generate points from a hexagonal lattice clipped to a circle, then scale.
    This is often close to optimal for packing-like problems.
    """
    points = []
    # Generate a grid of hex lattice points
    # Hex lattice spacing 1
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            if x*x + y*y <= radius*radius:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # If not enough points, expand radius
        return _generate_regular_polygon(n)
    
    # If more than n, select the n closest to center? 
    # Or select a subset that maximizes min distance?
    # For simplicity, take the first n after sorting by distance from center
    # This creates a "filled" hexagon shape
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n]]
    
    # Scale to unit radius
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        points = points / max_r
    
    return points

def _generate_centered_rings(n=13, seed=0):
    """
    Generate points in concentric rings.
    For n=13, a common configuration is 1 center + 6 inner + 6 outer, or similar.
    We will try a few specific distributions and randomize slightly.
    """
    rng = np.random.default_rng(seed)
    
    # Configuration 1: 1 center, 6 ring1, 6 ring2
    if n == 13:
        center = np.array([[0.0, 0.0]])
        
        # Ring 1: 6 points at radius r1
        r1 = 0.5
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi)
        ring1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Ring 2: 6 points at radius r2
        r2 = 1.0
        angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi)
        # Offset angles to interleave if possible
        angles2 += np.pi/6 
        ring2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        points = np.vstack([center, ring1, ring2])
        
    elif n == 12:
        # 6+6
        r1 = 0.6
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi)
        ring1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        r2 = 1.0
        angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi)
        ring2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        points = np.vstack([ring1, ring2])
        
    else:
        # Fallback: Random
        return rng.random((n, 2)) * 2 - 1

    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_score = -1.0
    best_points = np.random.default_rng(0).random((n, 2))
    
    # List of initial configurations
    # We want diverse starting points in the "basins" of attraction
    initializers = []
    
    # 1. Regular polygon
    initializers.append(_generate_regular_polygon(n))
    
    # 2. Hex lattice based
    initializers.append(_generate_hex_lattice(n))
    
    # 3. Centered rings with various seeds
    for seed in range(10):
        initializers.append(_generate_centered_rings(n, seed=seed))
        
    # 4. Random points
    rng = np.random.default_rng(42)
    for _ in range(5):
        initializers.append(rng.random((n, 2)) * 2 - 1)

    # 5. Perturbed regular polygon
    base_poly = _generate_regular_polygon(n)
    for seed in range(5):
        noise = np.random.default_rng(seed).normal(0, 0.1, (n, 2))
        initializers.append(base_poly + noise)

    current_idx = 0
    max_restarts = len(initializers) * 3 # Allow multiple attempts per initializer if time permits
    
    while time.time() < _DEADLINE and current_idx < max_restarts:
        if time.time() > _DEADLINE - 5:
            break
            
        # Select initializer
        init_idx = current_idx % len(initializers)
        init_points = initializers[init_idx].copy()
        
        # Add small random perturbation to avoid exact symmetry degeneracies if needed
        # But for Nelder-Mead, symmetric starts are fine as long as simplex is asymmetric
        # We will let Nelder-Mead handle the breaking of symmetry.
        
        try:
            init_flat = init_points.flatten()
            
            # Run Nelder-Mead on the EXACT objective
            # Nelder-Mead is robust for non-smooth functions if the simplex is well-sized
            # We use maxiter to limit time per restart
            # ftol/gtol help with convergence
            res = minimize(
                _objective, 
                init_flat, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10,
                    'adaptive': True,
                    'disp': False
                }
            )
            
            res_points = res.x.reshape(n, 2)
            
            # Validate
            if not np.all(np.isfinite(res_points)):
                current_idx += 1
                continue
                
            res_score = _score(res_points)
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                
        except Exception:
            pass
            
        current_idx += 1
        
    return best_points

# EVOLVE-BLOCK-END
