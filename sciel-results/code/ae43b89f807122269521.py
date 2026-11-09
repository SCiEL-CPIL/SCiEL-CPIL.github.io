# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2


def _neg_score(x):
    """Negative score for minimization."""
    points = x.reshape(14, 3)
    return -_score(points)


def _make_icosahedron():
    """Create 12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _initial_guess():
    """Create initial guess: 12 icosahedron vertices + 2 points along z-axis."""
    ico = _make_icosahedron()
    # Add 2 points along z-axis (top and bottom)
    # The icosahedron has points at various z values; add points at (0,0,±1)
    # But we need to check if this improves min distance
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    points = np.vstack([ico, p1, p2])
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses structured initial guess (icosahedron + 2 axial points) with local optimization.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit
    
    # Initial guess
    points = _initial_guess()
    x0 = points.flatten()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Try optimization from the initial guess
    try:
        result = minimize(
            _neg_score,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() < deadline:
            candidate = result.x.reshape(14, 3)
            s = _score(candidate)
            if s > best_score:
                best_points = candidate
                best_score = s
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Generate random points on a sphere with some jitter
        angles = np.random.uniform(0, 2 * np.pi, 14)
        cos_theta = np.random.uniform(-1, 1, 14)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        
        # Start from a perturbed icosahedron-like structure
        ico = _make_icosahedron()
        noise_scale = np.random.uniform(0.1, 0.5)
        x0 = (ico + noise_scale * np.random.randn(12, 3)).flatten()
        # Add 2 random points on sphere
        for i in range(2):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            x0 = np.append(x0, [x, y, z])
        
        try:
            result = minimize(
                _neg_score,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if time.time() > deadline - 10.0:
                break
            candidate = result.x.reshape(14, 3)
            s = _score(candidate)
            if s > best_score:
                best_points = candidate
                best_score = s
        except Exception:
            continue
    
    # Final polish: high-precision optimization on the best point found
    if time.time() < deadline - 15.0:
        try:
            x0 = best_points.flatten()
            result = minimize(
                _neg_score,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            candidate = result.x.reshape(14, 3)
            s = _score(candidate)
            if s > best_score:
                best_points = candidate
                best_score = s
        except Exception:
            pass
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to initial guess
        best_points = _initial_guess()
    
    return best_points


# EVOLVE-BLOCK-END
