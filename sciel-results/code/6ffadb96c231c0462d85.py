import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    The covering radius is the maximum circumradius of the convex hull facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
            
        d = np.abs(np.dot(normal, p0)) / norm_normal
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)
        
        if alpha > max_radius:
            max_radius = alpha
            
    return max_radius

def _facet_data(points):
    """
    Compute convex hull and return per-facet data:
    - indices: (M, 3) array of vertex indices
    - d: (M,) array of cos(alpha) for each facet
    - alphas: (M,) array of alpha for each facet
    - worst_idx: index of the worst facet
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, -1
    
    simplices = hull.simplices
    M = len(simplices)
    
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        simplices = simplices[valid]
        norms = norms[valid]
        normals = normals[valid]
        p0 = p0[valid]
        M = len(simplices)
    
    if M == 0:
        return None, None, None, -1
    
    d = np.abs(np.dot(normals, p0, axis=1)) / norms
    d = np.clip(d, 0.0, 1.0)
    alphas = np.arccos(d)
    
    worst_idx = np.argmax(alphas)
    
    return simplices, d, alphas, worst_idx

def _gradient_step(points, step_size=0.01, temperature=0.02):
    """
    Compute a gradient-like step for minimizing the smooth max of facet circumradii.
    Uses softmax weights over facets to compute weighted normal directions.
    """
    simplices, d, alphas, worst_idx = _facet_data(points)
    
    if simplices is None:
        return np.zeros_like(points)
    
    M = len(simplices)
    
    # Softmax weights over facets
    max_alpha = np.max(alphas)
    weights = np.exp((alphas - max_alpha) / temperature)
    weights /= np.sum(weights)
    
    # For each facet, the "gradient" direction to reduce its circumradius
    # is to move the three vertices away from the facet normal direction
    # (i.e., toward the facet normal, which increases d and decreases alpha)
    # Actually, to decrease alpha = arccos(d), we need to increase d.
    # d = |n . p0| / ||n||. Moving p0 in the direction of n increases d.
    # So the gradient direction for each vertex of a facet is toward the facet normal.
    
    # Compute facet normals (normalized)
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(norms, 1e-10)[:, np.newaxis]
    
    # Ensure normals point outward (away from origin)
    # For a convex hull of points on the unit sphere, the outward normal
    # should satisfy n . p0 > 0 (since p0 is on the sphere and the facet is on the hull)
    signs = np.sign(np.dot(normals, p0, axis=1))
    signs = np.where(signs == 0, 1.0, signs)
    normals *= signs[:, np.newaxis]
    
    # Gradient direction for each point: weighted sum of facet normals for facets containing that point
    grad = np.zeros_like(points)
    for i in range(3):
        idx = simplices[:, i]
        # Scatter-add weighted normals
        np.add.at(grad, idx, weights[:, np.newaxis] * normals)
    
    # Normalize gradient to unit length per point to control step size
    grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norms = np.maximum(grad_norms, 1e-10)
    grad = grad / grad_norms
    
    return step_size * grad

def _hill_climb(points, max_iter=200, base_step=0.02, decay=0.995, temperature=0.02):
    """
    Hill climbing optimization using gradient steps.
    """
    points = points.copy()
    current_radius = _covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    for it in range(max_iter):
        step_size = base_step * (decay ** it)
        grad = _gradient_step(points, step_size=1.0, temperature=temperature)
        
        # Move in the direction that reduces the objective
        # The gradient points in the direction that increases d (reduces alpha)
        # So we move along +grad to reduce the covering radius
        new_points = points + grad
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points /= norms
        
        new_radius = _covering_radius(new_points)
        
        if new_radius < current_radius:
            points = new_points
            current_radius = new_radius
            if current_radius < best_radius:
                best_points = points.copy()
                best_radius = current_radius
        else:
            # Try smaller step
            small_grad = grad * 0.1
            new_points = points + small_grad
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points /= norms
            new_radius = _covering_radius(new_points)
            
            if new_radius < current_radius:
                points = new_points
                current_radius = new_radius
                if current_radius < best_radius:
                    best_points = points.copy()
                    best_radius = current_radius
            else:
                break
    
    return best_points, best_radius

def _repulsive_relaxation(points, steps=2000, alpha=0.03):
    """
    Relax points using repulsive forces to spread them out.
    """
    points = points.copy()
    for _ in range(steps):
        # Compute all pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_dist_cubed, 0.0)
        forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
        
        points += alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
    
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # 2. Repulsive Relaxation
    if time.time() < deadline - 40:
        points = _repulsive_relaxation(points, steps=2000, alpha=0.03)
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_points = points.copy()
            best_radius = current_radius
    
    # 3. Hill climbing with gradient steps
    if time.time() < deadline - 20:
        optimized, opt_radius = _hill_climb(best_points, max_iter=300, base_step=0.015, decay=0.997, temperature=0.02)
        if opt_radius < best_radius:
            best_points = optimized
            best_radius = opt_radius
    
    # 4. Multiple restarts with perturbations
    np.random.seed(42)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        
        # Perturbation strategies
        strategy = restart_count % 4
        if strategy == 0:
            noise = np.random.normal(0, 0.03, size=best_points.shape)
        elif strategy == 1:
            noise = np.random.normal(0, 0.08, size=best_points.shape)
        elif strategy == 2:
            # Random rotation
            theta = np.random.uniform(0, 2 * np.pi)
            axis = np.random.normal(size=3)
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            noise = best_points @ R.T - best_points
        else:
            # Swap two points with small perturbation
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            noise = np.zeros_like(best_points)
            noise[idx1] = best_points[idx2] - best_points[idx1]
            noise[idx2] = best_points[idx1] - best_points[idx2]
            noise += np.random.normal(0, 0.02, size=best_points.shape)
        
        perturbed = best_points + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed /= norms
        
        # Quick relaxation
        if time.time() < deadline - 10:
            relaxed = _repulsive_relaxation(perturbed, steps=800, alpha=0.03)
            rad = _covering_radius(relaxed)
        else:
            relaxed = perturbed
            rad = _covering_radius(relaxed)
        
        if rad < best_radius:
            best_points = relaxed
            best_radius = rad
            
            # Refine with hill climbing if we have time
            if time.time() < deadline - 10:
                optimized, opt_radius = _hill_climb(best_points, max_iter=150, base_step=0.01, decay=0.997, temperature=0.02)
                if opt_radius < best_radius:
                    best_points = optimized
                    best_radius = opt_radius
    
    return best_points
