# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.special import logsumexp

# Deadline guard
_DEADLINE = 625.0  # seconds, safety margin before 630s kill


def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if not np.all(np.isfinite(dists)):
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _soft_min(dists: np.ndarray, beta: float) -> float:
    """Soft minimum via log-sum-exp (more negative beta = sharper)."""
    # soft_min = -logsumexp(-dists * beta) / beta
    return -logsumexp(-dists * beta) / beta


def _soft_max(dists: np.ndarray, beta: float) -> float:
    """Soft maximum via log-sum-exp."""
    return logsumexp(dists * beta) / beta


def _smooth_objective(points: np.ndarray, beta: float) -> float:
    """
    Smooth surrogate: maximize (soft_min / soft_max)^2.
    We maximize the log of this for numerical stability:
    2 * (log(soft_min) - log(soft_max))
    """
    dists = pdist(points)
    sm = _soft_min(dists, beta)
    sx = _soft_max(dists, beta)
    if sm <= 0 or sx <= 0:
        return -1e10
    return 2.0 * (np.log(sm) - np.log(sx))


def _gradient(points: np.ndarray, beta: float, eps: float = 1e-4) -> np.ndarray:
    """
    Numerical gradient of the smooth objective with respect to points.
    points: (14, 3)
    Returns: (14, 3) gradient array.
    """
    n, d = points.shape
    grad = np.zeros_like(points)
    base_val = _smooth_objective(points, beta)
    
    for i in range(n):
        for j in range(d):
            orig = points[i, j]
            points[i, j] = orig + eps
            val_plus = _smooth_objective(points, beta)
            points[i, j] = orig - eps
            val_minus = _smooth_objective(points, beta)
            points[i, j] = orig
            grad[i, j] = (val_plus - val_minus) / (2.0 * eps)
    
    return grad


def _local_optimize(points: np.ndarray, max_iters: int = 2000, 
                    beta_start: float = 10.0, beta_end: float = 500.0,
                    step_start: float = 0.1, step_end: float = 0.001,
                    deadline: float = 625.0) -> np.ndarray:
    """
    Local optimization using gradient ascent on the smooth surrogate
    with beta annealing.
    """
    points = points.copy()
    n = len(points)
    best_points = points.copy()
    best_score = _evaluate(points)
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Anneal beta
        frac = it / max(max_iters - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        # Anneal step size
        step = step_start * (step_end / step_start) ** frac
        
        # Compute gradient
        grad = _gradient(points, beta)
        
        # Gradient ascent step with normalization
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad_normalized = grad / grad_norm
        
        # Try the step
        new_points = points + step * grad_normalized
        
        # Check validity
        if not np.all(np.isfinite(new_points)):
            break
        
        # Evaluate both smooth and true objective
        new_true_score = _evaluate(new_points)
        
        if new_true_score > best_score:
            best_score = new_true_score
            best_points = new_points.copy()
        
        # Accept step if smooth objective improves or true score improves
        old_smooth = _smooth_objective(points, beta)
        new_smooth = _smooth_objective(new_points, beta)
        
        if new_smooth > old_smooth or new_true_score > _evaluate(points):
            points = new_points
        else:
            # Try smaller step
            new_points = points + step * 0.1 * grad_normalized
            if np.all(np.isfinite(new_points)):
                new_true_score = _evaluate(new_points)
                if new_true_score > _evaluate(points):
                    points = new_points
                    if new_true_score > best_score:
                        best_score = new_true_score
                        best_points = new_points.copy()
    
    # Final polish with high beta
    if time.time() < deadline - 10:
        points = best_points.copy()
        for it in range(500):
            if time.time() > deadline - 5:
                break
            beta = 1000.0
            step = 0.0005
            grad = _gradient(points, beta)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad_normalized = grad / grad_norm
            new_points = points + step * grad_normalized
            if not np.all(np.isfinite(new_points)):
                break
            new_true_score = _evaluate(new_points)
            old_true_score = _evaluate(points)
            if new_true_score >= old_true_score:
                points = new_points
                if new_true_score > best_score:
                    best_score = new_true_score
                    best_points = new_points.copy()
            else:
                break
    
    return best_points


def _scale_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Scale points so that max distance is 1."""
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 0 and np.isfinite(d_max):
        points = points / d_max
    return points


def _generate_initial_points(seed: int) -> np.ndarray:
    """Generate initial points using a deterministic seed."""
    np.random.seed(seed)
    points = np.random.randn(14, 3)
    return points


def _try_vertex_configuration() -> np.ndarray:
    """
    Try a configuration based on icosahedron vertices plus center and one extra point.
    Icosahedron has 12 vertices. We need 14 points.
    Add center and one point on the surface.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Add center
    center = np.array([0.0, 0.0, 0.0])
    
    # Add one more point - try a point opposite to one vertex
    extra = -vertices[0]
    
    points = np.vstack([vertices, center.reshape(1, -1), extra.reshape(1, -1)])
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    
    # Try multiple initial configurations
    best_points = None
    best_score = -1.0
    
    # Initial configurations to try
    configs = []
    
    # 1. Icosahedron-based
    configs.append(_try_vertex_configuration())
    
    # 2. Random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        configs.append(_generate_initial_points(seed))
    
    # 3. Sphere-based: points on unit sphere
    np.random.seed(999)
    sphere_points = np.random.randn(14, 3)
    norms = np.linalg.norm(sphere_points, axis=1, keepdims=True)
    configs.append(sphere_points / norms)
    
    # 4. Cube-based: 8 corners + 6 face centers
    cube = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
        [0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]
    ], dtype=float)
    configs.append(cube)
    
    # Optimize each configuration
    for i, config in enumerate(configs):
        if time.time() > _DEADLINE - 30:
            break
        
        # Scale to reasonable range
        config = _scale_to_unit_sphere(config.copy())
        # Add small perturbation to avoid degeneracy
        np.random.seed(i * 1000 + 7)
        config += np.random.randn(14, 3) * 0.01
        
        # Local optimize
        optimized = _local_optimize(
            config, 
            max_iters=800,
            beta_start=5.0, 
            beta_end=300.0,
            step_start=0.05, 
            step_end=0.0005,
            deadline=_DEADLINE - 10
        )
        
        score = _evaluate(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() < _DEADLINE - 20:
        best_points = _local_optimize(
            best_points,
            max_iters=1000,
            beta_start=100.0,
            beta_end=2000.0,
            step_start=0.01,
            step_end=0.0001,
            deadline=_DEADLINE - 5
        )
    
    if best_points is None:
        # Fallback
        np.random.seed(42)
        best_points = np.random.randn(14, 3)
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        np.random.seed(42)
        best_points = np.random.randn(14, 3)
    
    return best_points


# EVOLVE-BLOCK-END
