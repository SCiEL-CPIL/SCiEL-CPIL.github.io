import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
DEADLINE = 140.0  # Leave 10s margin for 150s limit


def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def _covering_radius_exact(points):
    """Compute exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return _covering_radius_grid(points)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            continue
        
        n_unit = n / n_norm
        d = np.dot(n_unit, v1)
        d_abs = abs(d)
        if d_abs > 1.0:
            d_abs = 1.0
        
        angle = np.arccos(d_abs)
        if angle > max_radius:
            max_radius = angle
    
    return np.degrees(max_radius)


def _covering_radius_grid(points, n_theta=360, n_phi=720):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid_points), chunk_size):
        end = min(start + chunk_size, len(grid_points))
        chunk = grid_points[start:end]
        dots = np.clip(chunk @ points.T, -1.0, 1.0)
        max_dots = dots.max(axis=1)
        angles = np.arccos(max_dots)
        if angles.max() > max_dist:
            max_dist = angles.max()
    
    return np.degrees(max_dist)


def _flatten(points):
    """Flatten Nx3 points to 1D vector."""
    return points.ravel()


def _unflatten(x):
    """Unflatten 1D vector to Nx3 points and normalize."""
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _soft_max_coverage(points, grid_points, temperature=0.05):
    """
    Compute a smooth surrogate for the max angular distance using soft-max.
    For each grid point, compute angular distance to all points, then soft-max.
    Return the max over grid points of the soft-max value, and the gradient w.r.t. points.
    """
    # Compute dot products: (M, N) where M = len(grid_points)
    dots = np.clip(grid_points @ points.T, -1.0 + 1e-12, 1.0 - 1e-12)
    # Angular distances
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-max over N dimension for each grid point
    # f_i = -T * log(sum_j exp(-angles_ij / T))
    # This approximates min_j angles_ij (the nearest point distance)
    # We want to maximize the min, so we minimize the negative soft-min
    # Actually: soft_min(x) = -T * log(sum exp(-x_j/T))
    # We want to minimize max_i soft_min(angles_i)
    
    neg_angles_over_T = -angles / temperature
    # For numerical stability, subtract max
    max_neg = np.max(neg_angles_over_T, axis=1, keepdims=True)
    exp_terms = np.exp(neg_angles_over_T - max_neg)
    log_sum = np.log(np.sum(exp_terms, axis=1)) + max_neg.ravel()
    soft_min = -temperature * log_sum  # (M,)
    
    # The objective is max_i soft_min_i
    # For gradient, we need d(max_i soft_min_i)/d(points)
    # Use another soft-max to smooth the outer max
    outer_T = 0.05
    neg_soft_min_over_T = -soft_min / outer_T
    max_neg_outer = np.max(neg_soft_min_over_T, keepdims=True)
    exp_outer = np.exp(neg_soft_min_over_T - max_neg_outer)
    weights = exp_outer / np.sum(exp_outer)  # (M,)
    
    objective = -outer_T * (np.log(np.sum(exp_outer)) + max_neg_outer.ravel()[0])
    
    # Gradient computation
    # d(soft_min_i)/d(points_k) = -1/T * sum_j [exp(-angles_ij/T) / sum_l exp(-angles_il/T)] * d(angles_ij)/d(points_k)
    # d(angles_ij)/d(points_k) = -dot(grid_i, points_k) / (sqrt(1-dot^2) * ||points_k||)
    # Since points are on unit sphere, ||points_k|| = 1
    # So d(angles_ij)/d(points_k) = -dot_i_k / sqrt(1 - dot_i_k^2)
    
    # For each grid point i and each point k:
    # d(soft_min_i)/d(points_k) = -1/T * w_ij * (-dot_i_k / sqrt(1 - dot_i_k^2))
    # where w_ij = exp(-angles_ij/T) / sum_l exp(-angles_il/T)
    
    # w_ij = exp_terms[i,j] / sum(exp_terms[i,:])
    sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
    W = exp_terms / sum_exp  # (M, N)
    
    # d(angles_ij)/d(points_k) for k = j: -dot_i_j / sqrt(1 - dot_i_j^2)
    # For k != j: 0 (since angles_ij only depends on points_j)
    # So d(soft_min_i)/d(points_j) = -1/T * W[i,j] * (-dots[i,j] / sqrt(1 - dots[i,j]^2))
    # = 1/T * W[i,j] * dots[i,j] / sqrt(1 - dots[i,j]^2)
    
    # But we need to be careful: the points are constrained to the unit sphere.
    # The gradient on the sphere is the tangential component.
    
    # Compute per-point gradient
    # grad_k = sum_i weights_i * (1/T) * W[i,k] * dots[i,k] / sqrt(1 - dots[i,k]^2)
    
    # For numerical stability, clip the sqrt argument
    sqrt_arg = 1.0 - dots * dots
    sqrt_arg = np.maximum(sqrt_arg, 1e-12)
    sqrt_vals = np.sqrt(sqrt_arg)  # (M, N)
    
    # Per-element contribution: W[i,k] * dots[i,k] / sqrt_vals[i,k]
    contrib = W * dots / sqrt_vals  # (M, N)
    
    # Weight by outer weights and temperature
    # d(objective)/d(points_k) = sum_i weights_i * (1/T) * contrib[i,k]
    # But objective = -outer_T * log_sum_outer, and d(objective)/d(soft_min_i) = -weights_i
    # So d(objective)/d(points_k) = sum_i (-weights_i) * d(soft_min_i)/d(points_k)
    # = sum_i (-weights_i) * (1/T) * contrib[i,k]
    
    grad_2d = -weights[:, np.newaxis] * (1.0 / temperature) * contrib  # (M, N)
    grad_points = np.sum(grad_2d, axis=0)  # (N,) per coordinate... wait, this is per point index
    
    # Actually, contrib[i,k] is the derivative w.r.t. the k-th point's position
    # But points are 3D, so we need to think in terms of the 3D vector
    
    # Let me redo this more carefully.
    # points is (N, 3), dots is (M, N) where dots[i,k] = grid_points[i] . points[k]
    # angles[i,k] = arccos(dots[i,k])
    # d(angles[i,k])/d(points[k]) = -grid_points[i] / sqrt(1 - dots[i,k]^2)  (3D vector)
    
    # soft_min_i = -T * log(sum_k exp(-angles[i,k]/T))
    # d(soft_min_i)/d(points[k]) = -T * (1/sum_k exp(-angles[i,k]/T)) * d/d(points[k]) [exp(-angles[i,k]/T)]
    # = -T * (1/sum_k exp(-angles[i,k]/T)) * exp(-angles[i,k]/T) * (-1/T) * d(angles[i,k])/d(points[k])
    # = (exp(-angles[i,k]/T) / sum_k exp(-angles[i,k]/T)) * d(angles[i,k])/d(points[k])
    # = W[i,k] * (-grid_points[i] / sqrt(1 - dots[i,k]^2))
    
    # So d(soft_min_i)/d(points[k]) = -W[i,k] / sqrt(1 - dots[i,k]^2) * grid_points[i]
    
    # objective = -outer_T * log(sum_i exp(-soft_min_i / outer_T))
    # d(objective)/d(points[k]) = sum_i d(objective)/d(soft_min_i) * d(soft_min_i)/d(points[k])
    # d(objective)/d(soft_min_i) = -outer_T * (1/sum_i exp(-soft_min_i/outer_T)) * exp(-soft_min_i/outer_T) * (-1/outer_T)
    # = exp(-soft_min_i/outer_T) / sum_i exp(-soft_min_i/outer_T)
    # = weights_i
    
    # So d(objective)/d(points[k]) = sum_i weights_i * (-W[i,k] / sqrt(1 - dots[i,k]^2)) * grid_points[i]
    
    # This is a 3D vector for each k
    grad_3d = np.zeros((N, 3))
    for k in range(N):
        # grad_3d[k] = sum_i weights_i * (-W[i,k] / sqrt_vals[i,k]) * grid_points[i]
        coeff = weights * (-W[:, k] / sqrt_vals[:, k])  # (M,)
        grad_3d[k] = coeff @ grid_points  # (3,)
    
    # Project gradient onto tangent plane (remove radial component)
    for k in range(N):
        radial = np.dot(grad_3d[k], points[k]) * points[k]
        grad_3d[k] -= radial
    
    return objective, grad_3d


def _make_grid(n_theta=180, n_phi=360):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def _gradient_optimize(points, grid_points, max_iters=200, time_budget=30.0):
    """Optimize using gradient descent on the smooth coverage objective."""
    points = points.copy()
    start_time = time.time()
    
    temperature = 0.03
    outer_T = 0.03
    lr = 0.01
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        # Compute objective and gradient
        obj, grad = _soft_max_coverage(points, grid_points, temperature)
        
        # Gradient descent with projection onto sphere
        new_points = points - lr * grad
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        new_points = new_points / norms
        
        # Check for improvement
        new_obj, _ = _soft_max_coverage(new_points, grid_points, temperature)
        if new_obj < obj:
            points = new_points
        else:
            # Halve learning rate
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return points


def _lloyd_iteration(points, grid_points, n_iters=50):
    """Lloyd-style iteration: move each point toward the centroid of its Voronoi cell (approximated by grid)."""
    points = points.copy()
    start_time = time.time()
    
    for it in range(n_iters):
        if time.time() > DEADLINE:
            break
        
        # Assign each grid point to nearest configuration point
        dots = grid_points @ points.T  # (M, N)
        nearest = np.argmax(dots, axis=1)  # (M,)
        
        # Compute new positions as centroids of assigned grid points
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        for k in range(N):
            mask = nearest == k
            if np.any(mask):
                new_points[k] = grid_points[mask].mean(axis=0)
                counts[k] = np.sum(mask)
            else:
                new_points[k] = points[k]
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        new_points = new_points / norms
        
        # Check for improvement
        new_radius = _covering_radius_grid(new_points, n_theta=90, n_phi=180)
        old_radius = _covering_radius_grid(points, n_theta=90, n_phi=180)
        
        if new_radius < old_radius:
            points = new_points
        else:
            break
    
    return points


def _refine_facets(points, n_iters=100, step_size=0.02):
    """Refine points by moving vertices of worst facets toward facet normals."""
    points = points.copy()
    
    for it in range(n_iters):
        if time.time() > DEADLINE:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Compute circumradius for each facet
        radii = np.zeros(len(hull.simplices))
        normals = []
        for idx, simplex in enumerate(hull.simplices):
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                normals.append(np.zeros(3))
                radii[idx] = 0.0
                continue
            n_unit = n / n_norm
            d = abs(np.dot(n_unit, v1))
            d = min(max(d, 0.0), 1.0)
            radii[idx] = np.arccos(d)
            normals.append(n_unit)
        
        normals = np.array(normals)
        
        # Find the worst facets
        max_radius = np.max(radii)
        if max_radius < 1e-10:
            break
        
        # Move vertices of facets with radius > 0.9 * max_radius
        threshold = 0.9 * max_radius
        worst_mask = radii > threshold
        
        if not np.any(worst_mask):
            break
        
        # Accumulate displacement for each point
        displacement = np.zeros_like(points)
        count = np.zeros(N)
        
        for idx in np.where(worst_mask)[0]:
            simplex = hull.simplices[idx]
            n_unit = normals[idx]
            for vi in simplex:
                # Move vertex toward the normal direction
                tangent = n_unit - np.dot(n_unit, points[vi]) * points[vi]
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-12:
                    displacement[vi] += tangent / tangent_norm
                    count[vi] += 1
        
        # Apply displacement
        for i in range(N):
            if count[i] > 0:
                avg_disp = displacement[i] / count[i]
                new_pt = points[i] + step_size * avg_disp
                new_norm = np.linalg.norm(new_pt)
                if new_norm > 1e-12:
                    points[i] = new_pt / new_norm
        
    return points


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    best_points = _fibonacci_sphere(N, offset=0.5)
    best_radius = _covering_radius_exact(best_points)
    
    # Create a moderate grid for gradient optimization
    grid_points = _make_grid(n_theta=120, n_phi=240)
    
    # Multiple random starts with gradient-based refinement
    n_starts = 5
    for s in range(n_starts):
        if time.time() > DEADLINE - 20:
            break
        
        if s == 0:
            points = best_points.copy()
        else:
            noise = np.random.normal(0, 0.2, size=(N, 3))
            points = best_points + noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            points = points / norms
        
        # Gradient optimization
        points = _gradient_optimize(points, grid_points, max_iters=100, time_budget=20.0)
        
        # Lloyd iteration
        points = _lloyd_iteration(points, grid_points, n_iters=30)
        
        # Facet-based refinement
        points = _refine_facets(points, n_iters=50, step_size=0.01)
        
        # Compute exact radius
        radius = _covering_radius_exact(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Final refinement with gradient
    if time.time() < DEADLINE - 15:
        best_points = _gradient_optimize(best_points, grid_points, max_iters=150, time_budget=25.0)
        best_points = _lloyd_iteration(best_points, grid_points, n_iters=30)
        best_points = _refine_facets(best_points, n_iters=50, step_size=0.005)
        final_radius = _covering_radius_exact(best_points)
        if final_radius < best_radius:
            best_radius = final_radius
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_points = best_points / norms
    
    return best_points
