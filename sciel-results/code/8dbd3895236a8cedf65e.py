import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50

def _covering_radius_and_grads(points):
    """
    Compute the covering radius and gradients for points on the unit sphere.
    Returns (radius, grads) where grads is shape (N, 3).
    Uses convex hull facets. The covering radius is the max circumradius of facets.
    """
    points = points.copy()
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback: use grid-based approximation
        grid = _generate_grid(60)
        dists = np.arccos(np.clip(grid @ points.T, -1, 1))  # (M, N)
        min_dists = np.min(dists, axis=0)  # (N,)
        radius = np.max(min_dists)
        # Approximate gradients
        grads = np.zeros_like(points)
        for i in range(N):
            mask = np.argmin(dists[:, i])
            g = -grid[mask]
            grads[i] = g
        return radius, grads

    # Compute circumradii for all facets
    n_facets = len(hull.simplices)
    radii = np.zeros(n_facets)
    circumcenters = np.zeros((n_facets, 3))
    
    for fi, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[fi] = 0
            circumcenters[fi] = a
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
            
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        radii[fi] = r
        circumcenters[fi] = n

    # Find the max radius facet(s)
    max_radius = np.max(radii)
    
    # For gradient computation, use a soft-max to handle ties
    # Use log-sum-exp for numerical stability
    beta = 100  # Sharpness of soft-max
    exp_radii = np.exp(beta * (radii - max_radius))
    weights = exp_radii / np.sum(exp_radii)
    
    # Gradient is weighted sum of gradients from each facet
    grads = np.zeros_like(points)
    
    for fi in range(n_facets):
        if weights[fi] < 1e-8:
            continue
            
        simplex = hull.simplices[fi]
        a_idx, b_idx, c_idx = simplex
        a = points[a_idx]
        b = points[b_idx]
        c = points[c_idx]
        
        cc = circumcenters[fi]
        r = radii[fi]
        
        # Gradient of arccos(x) is -1/sqrt(1-x^2)
        # x = dot(cc, a), so d(arccos)/da = -cc/sqrt(1-x^2)
        # But we need to account for how cc depends on a, b, c
        
        # Simplified gradient: move each vertex toward the circumcenter
        # This is an approximation but works well in practice
        w = weights[fi]
        
        # The circumcenter depends on a, b, c. 
        # For a spherical triangle, moving a vertex toward the circumcenter
        # decreases the circumradius.
        
        # Gradient direction for each vertex: toward circumcenter
        # Project onto tangent plane
        for idx in [a_idx, b_idx, c_idx]:
            p = points[idx]
            # Direction toward circumcenter, projected to tangent plane
            d = cc - np.dot(cc, p) * p
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-10:
                d = d / d_norm
            grads[idx] += w * d
    
    # Normalize gradients
    grad_norms = np.linalg.norm(grads, axis=1, keepdims=True)
    grad_norms = np.where(grad_norms < 1e-10, 1.0, grad_norms)
    grads = grads / grad_norms
    
    return max_radius, grads

def _generate_grid(n_theta=50, n_phi=100):
    """Generate a grid of points on the sphere for fallback evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _local_search(points, max_steps=500, step_size=0.1, deadline=None):
    """
    Local search to minimize the covering radius using gradient information.
    """
    current_points = points.copy()
    best_radius, _ = _covering_radius_and_grads(current_points)
    best_points = current_points.copy()
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        radius, grads = _covering_radius_and_grads(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        if radius < 1e-8:
            break
            
        # Gradient descent step with line search
        # Move opposite to gradient (toward lower radius)
        lr = step_size
        
        # Try several step sizes
        best_trial_radius = radius
        best_trial_points = current_points.copy()
        
        for trial_lr in [lr, lr * 0.5, lr * 0.1, lr * 2.0]:
            new_points = current_points - trial_lr * grads
            # Normalize to sphere
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            
            trial_radius, _ = _covering_radius_and_grads(new_points)
            
            if trial_radius < best_trial_radius:
                best_trial_radius = trial_radius
                best_trial_points = new_points.copy()
        
        if best_trial_radius < radius:
            current_points = best_trial_points
        else:
            # Stuck, try smaller steps
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_points, best_radius

def _sampled_objective(x_flat, points_init):
    """
    Objective function for scipy optimization: 
    covering radius approximated by max distance to nearest point on a grid.
    """
    points = x_flat.reshape(-1, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Use a moderate grid for speed
    grid = _generate_grid(40)
    dists = np.arccos(np.clip(grid @ points.T, -1, 1))
    min_dists = np.min(dists, axis=0)
    return np.max(min_dists)

def _polish_with_scipy(points, deadline=None):
    """
    Final polish using L-BFGS-B on a sampled objective.
    """
    if deadline is not None and time.time() > deadline - 5:
        return points, _covering_radius_and_grads(points)[0]
    
    x0 = points.flatten()
    
    # Use L-BFGS-B with bounds to keep points on sphere approximately
    # Actually, we'll use a reparameterization or just constrain via normalization
    # For simplicity, use the flat representation and normalize after each step
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return _sampled_objective(x, pts)
    
    try:
        result = scipy.optimize.minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        best_points = result.x.reshape(-1, 3)
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
        best_radius = _covering_radius_and_grads(best_points)[0]
        return best_points, best_radius
    except Exception:
        return points, _covering_radius_and_grads(points)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    best_points = None
    best_radius = np.inf
    
    # Phase 1: Multiple restarts with gradient-based local search
    n_restarts = 30
    
    for i in range(n_restarts):
        if time.time() > deadline - 30:  # Reserve 30s for polishing
            break
            
        # Initialize points with Fibonacci sphere + perturbation
        rng = np.random.default_rng(seed=12345 + i)
        
        # Base Fibonacci
        j = np.arange(N) + 0.5
        z = 1.0 - 2.0 * j / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * j
        r = np.sqrt(1.0 - z * z)
        base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add noise with varying amplitude
        noise_scale = 0.05 + 0.1 * (i % 5) / 4
        noise = rng.normal(0, noise_scale, size=(N, 3))
        points = base_points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Run local search with gradient
        pts, rad = _local_search(points, max_steps=300, step_size=0.08, deadline=deadline - 30)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Phase 2: Polish the best solution with scipy
    if best_points is not None:
        pts, rad = _polish_with_scipy(best_points, deadline=deadline)
        if rad < best_radius:
            best_radius = rad
            best_points = pts
    
    # Phase 3: Additional gradient-based polish with very small steps
    if best_points is not None and time.time() < deadline - 5:
        pts, rad = _local_search(best_points, max_steps=200, step_size=0.01, deadline=deadline - 2)
        if rad < best_radius:
            best_radius = rad
            best_points = pts
    
    if best_points is None:
        # Fallback to seed
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_points
