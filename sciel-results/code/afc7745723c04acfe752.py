"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _covering_radius(points, grid_cache=None):
    """
    Compute covering radius in degrees using dense grid + convex hull facet circumcenters.
    """
    points = _normalize(points)
    
    if grid_cache is None:
        M = 4000
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    else:
        grid = grid_cache

    dots = grid @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    min_angles = np.arccos(max_dots)
    cr_grid = np.max(min_angles) * 180.0 / np.pi

    # Check convex hull facet circumcenters
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        max_cr = cr_grid
        for simplex in hull.simplices:
            facet_pts = points[simplex]
            p1, p2, p3 = facet_pts[0], facet_pts[1], facet_pts[2]
            v = np.cross(np.cross(p1, p2), np.cross(p1, p3))
            norm = np.linalg.norm(v)
            if norm < 1e-15:
                continue
            cc = v / norm
            dots_cc = np.clip(points @ cc, -1.0, 1.0)
            min_dot_cc = np.max(dots_cc)
            angle_cc = np.arccos(min_dot_cc)
            d1 = np.arccos(np.clip(np.dot(cc, p1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(cc, p2), -1, 1))
            d3 = np.arccos(np.clip(np.dot(cc, p3), -1, 1))
            if abs(d1 - d2) < 1e-5 and abs(d1 - d3) < 1e-5:
                other_dots = dots_cc.copy()
                other_dots[simplex] = -1.0
                max_other_dot = np.max(other_dots)
                if max_other_dot <= np.dot(cc, p1) + 1e-8:
                    cr_facet = d1 * 180.0 / np.pi
                    max_cr = max(max_cr, cr_facet)
        return max_cr
    except Exception:
        return cr_grid


def _lse_covering_radius(points, grid, beta=200.0):
    """
    Smooth log-sum-exp approximation of covering radius.
    Returns (value, gradient) where value is in radians.
    For each grid point y, the distance to nearest point is -logsumexp(beta * dots).
    The covering radius is max over y, approximated by logsumexp(beta * angles).
    """
    dots = grid @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each grid point, the max dot product gives min angle
    # angle = arccos(max_dot)
    # We want to maximize the minimum angle (minimize max angle)
    # Use: f = logsumexp over grid points of exp(beta * angle)
    # This is smooth and differentiable
    
    max_dots = np.max(dots, axis=1)  # (M,)
    angles = np.arccos(max_dots)  # (M,)
    
    # Log-sum-exp for numerical stability
    a = beta * angles
    a_max = np.max(a)
    lse = a_max + np.log(np.sum(np.exp(a - a_max)))
    value = lse / beta  # approximate max angle
    
    # Gradient: d/dp_i of lse
    # d lse / d angle_m = beta * exp(beta * angle_m) / sum(exp(beta * angle_j))
    # d angle_m / d p_i = -1/sqrt(1-dot^2) * d dot_m / d p_i for the nearest point
    # This is complex; use the max-dot approach
    
    # Simpler: gradient of max over m of angle_m
    # The active grid points (near max) contribute
    weights = np.exp(a - a_max)  # (M,)
    weights = weights / np.sum(weights)  # softmax weights
    
    # For each grid point m, the nearest point index
    nearest_idx = np.argmax(dots, axis=1)  # (M,)
    
    # Gradient w.r.t. points
    grad = np.zeros_like(points)
    for m in range(len(grid)):
        w = weights[m]
        if w < 1e-12:
            continue
        i = nearest_idx[m]
        dot = dots[m, i]
        # d angle / d dot = -1/sqrt(1-dot^2)
        denom = np.sqrt(np.maximum(1e-15, 1.0 - dot * dot))
        # d dot_m / d p_i = grid[m]
        grad[i] -= w * grid[m] / denom
    
    # Project to tangent space
    grad_dot_p = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - grad_dot_p * points
    
    return value, grad


def _repulsion_grad(points, power=2):
    """
    Gradient of repulsion energy to maintain spread.
    E = sum_{i<j} (1 - p_i.p_j)^(-power)
    """
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    denom = np.maximum(1.0 - dots, 1e-10)
    weights = 1.0 / (denom ** (power + 1))
    np.fill_diagonal(weights, 0.0)
    Wsum = weights.sum(axis=1)
    grad = -(power + 1) * (weights @ points - Wsum[:, None] * points)
    grad_dot_p = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - grad_dot_p * points
    return grad


def _fibonacci_spiral(n, offset=0.5, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi, n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(points, start_time, deadline, grid):
    """
    Optimize from a starting configuration using LSE-smoothed covering radius gradient.
    """
    points = _normalize(points.copy())
    best_points = points.copy()
    best_cr = _covering_radius(points, grid)
    
    alpha = 0.05
    patience = 0
    max_patience = 500
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Compute LSE gradient
        value, grad = _lse_covering_radius(points, grid, beta=150.0)
        
        # Add small repulsion to maintain spread
        rep = _repulsion_grad(points, power=2)
        grad = grad + 0.005 * rep
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Normalize per-point gradient
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms = np.where(grad_norms < 1e-15, 1.0, grad_norms)
        grad_norm = grad / grad_norms
        
        # Take step
        new_points = points + alpha * grad_norm
        new_points = _normalize(new_points)
        
        new_cr = _covering_radius(new_points, grid)
        
        if new_cr < best_cr - 1e-10:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            alpha = min(alpha * 1.02, 0.1)
            patience = 0
        else:
            alpha *= 0.95
            patience += 1
            if patience > max_patience:
                break
    
    return best_points, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Pre-compute grid
    M = 4000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Multi-start optimization
    best_points = None
    best_cr = np.inf
    
    # Start 1: Fibonacci spiral
    points0 = _fibonacci_spiral(N, seed=0)
    p, cr = _optimize_from_start(points0, start_time, deadline, grid)
    if cr < best_cr:
        best_cr = cr
        best_points = p
    
    # Start 2: Different Fibonacci offset
    if time.time() < deadline - 30:
        points1 = _fibonacci_spiral(N, seed=42)
        p, cr = _optimize_from_start(points1, start_time, deadline, grid)
        if cr < best_cr:
            best_cr = cr
            best_points = p
    
    # Start 3: Random
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        points2 = rng.normal(size=(N, 3))
        points2 = _normalize(points2)
        p, cr = _optimize_from_start(points2, start_time, deadline, grid)
        if cr < best_cr:
            best_cr = cr
            best_points = p
    
    # Start 4: Another random
    if time.time() < deadline - 20:
        rng = np.random.default_rng(777)
        points3 = rng.normal(size=(N, 3))
        points3 = _normalize(points3)
        p, cr = _optimize_from_start(points3, start_time, deadline, grid)
        if cr < best_cr:
            best_cr = cr
            best_points = p
    
    # Final polish on best
    if time.time() < deadline - 5 and best_points is not None:
        best_points = _optimize_from_start(best_points, start_time, deadline, grid)[0]
    
    # Ensure validity
    best_points = _normalize(best_points)
    
    # Check for coincident points
    dots = best_points @ best_points.T
    np.fill_diagonal(dots, 0.0)
    if np.max(dots) > 0.99999:
        for i in range(N):
            for j in range(i + 1, N):
                if dots[i, j] > 0.99999:
                    noise = np.random.normal(0, 0.01, 3)
                    best_points[j] += noise
                    best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
