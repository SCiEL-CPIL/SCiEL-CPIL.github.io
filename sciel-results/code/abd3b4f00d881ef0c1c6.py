# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _min_dist(points: np.ndarray) -> float:
    """Return the minimum pairwise distance."""
    dists = pdist(points)
    return np.min(dists)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n points on a unit sphere using Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points


def _repulsion_optimize(points: np.ndarray, n_iter: int = 2000, lr: float = 0.01) -> np.ndarray:
    """
    Optimize points on the unit sphere to maximize minimum distance using
    repulsion forces (Tammes problem approach).
    """
    points = points.copy()
    n = len(points)
    
    for it in range(n_iter):
        # Compute repulsion forces
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random nudge if points coincide
                    diff = np.random.randn(3) * 0.01
                    dist = np.linalg.norm(diff)
                # Force proportional to 1/dist^2 (Coulomb-like)
                force_mag = 1.0 / (dist * dist)
                force_vec = diff / dist * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Update positions
        points += lr * forces
        
        # Project back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.99
    
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.

    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 14
    d = 3
    
    # Start with Fibonacci sphere initialization
    best_points = _fibonacci_sphere(n)
    best_score = _evaluate(best_points)
    
    # Run repulsion optimization
    points = _repulsion_optimize(best_points, n_iter=5000, lr=0.005)
    score = _evaluate(points)
    if score > best_score:
        best_points = points
        best_score = score
    
    # Check time
    if time.time() > deadline:
        return best_points
    
    # Try multiple restarts with different initializations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Different initializations
        if restart % 3 == 0:
            # Fibonacci with slight perturbation
            np.random.seed(restart * 7 + 1)
            init = _fibonacci_sphere(n) + np.random.randn(n, d) * 0.1
        elif restart % 3 == 1:
            # Random on sphere
            np.random.seed(restart * 13 + 2)
            init = np.random.randn(n, d)
            init = init / np.linalg.norm(init, axis=1, keepdims=True)
        else:
            # Random with different seed
            np.random.seed(restart * 23 + 3)
            init = np.random.randn(n, d)
            init = init / np.linalg.norm(init, axis=1, keepdims=True)
        
        # Optimize
        points = _repulsion_optimize(init, n_iter=3000, lr=0.003)
        score = _evaluate(points)
        if score > best_score:
            best_points = points
            best_score = score
    
    # Final polish with smaller steps
    if time.time() < deadline - 60:
        points = _repulsion_optimize(best_points, n_iter=5000, lr=0.001)
        score = _evaluate(points)
        if score > best_score:
            best_points = points
    
    # Ensure points are finite
    if not np.all(np.isfinite(best_points)):
        # Fallback to Fibonacci sphere
        best_points = _fibonacci_sphere(n)
    
    return best_points


# EVOLVE-BLOCK-END
