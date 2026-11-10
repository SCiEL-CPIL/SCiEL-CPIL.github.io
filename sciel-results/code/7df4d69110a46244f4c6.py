"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _spherical_circumcenter(a, b, c):
    """
    Compute the circumcenter of a spherical triangle with vertices a, b, c on the unit sphere.
    The circumcenter is the point p on the sphere such that p·a = p·b = p·c (all equal).
    """
    n = np.cross(a - b, a - c)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        centroid = a + b + c
        norm_c = np.linalg.norm(centroid)
        if norm_c < 1e-12:
            return a
        return centroid / norm_c
    
    p1 = n / norm_n
    p2 = -p1
    
    min_dot1 = min(np.dot(p1, a), np.dot(p1, b), np.dot(p1, c))
    min_dot2 = min(np.dot(p2, a), np.dot(p2, b), np.dot(p2, c))
    
    if min_dot1 <= min_dot2:
        return p1
    else:
        return p2


def _compute_circumcenters(pts, hull):
    """
    Compute circumcenters for all triangular facets of the convex hull.
    Returns array of circumcenters (one per simplex) and the angular radius for each.
    """
    n_facets = len(hull.simplices)
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        cc = _spherical_circumcenter(a, b, c)
        circumcenters[i] = cc
        
        dots = np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)])
        min_dot = np.min(dots)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        radii[i] = np.arccos(min_dot)
    
    return circumcenters, radii


def _cover_radius(pts):
    """
    Compute the covering radius using the convex hull facets.
    The covering radius is the maximum circumradius over all facets.
    """
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    _, radii = _compute_circumcenters(pts, hull)
    return np.max(radii)


def _cover_radius_with_grad(pts):
    """
    Compute the covering radius and the gradient with respect to each point.
    
    For the worst facet with circumcenter p and radius r, the gradient of r
    with respect to vertex v (on the unit sphere) is:
        grad_v(r) = -(p - (p·v)*v) / sin(r)
    
    This is the direction to move v to decrease r.
    """
    if len(pts) < 3:
        return np.pi, np.zeros_like(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros_like(pts)
    
    circumcenters, radii = _compute_circumcenters(pts, hull)
    
    # Find the worst facet
    max_idx = np.argmax(radii)
    max_radius = radii[max_idx]
    
    if max_radius < 1e-10:
        return max_radius, np.zeros_like(pts)
    
    # Get the circumcenter and the three vertices of the worst facet
    cc = circumcenters[max_idx]
    simplex = hull.simplices[max_idx]
    
    # Gradient: for each vertex v of the worst facet,
    # grad_v(r) = -(cc - (cc·v)*v) / sin(r)
    sin_r = np.sin(max_radius)
    if sin_r < 1e-12:
        return max_radius, np.zeros_like(pts)
    
    grad = np.zeros_like(pts)
    for idx in simplex:
        v = pts[idx]
        dot_cv = np.dot(cc, v)
        # Project cc onto the tangent plane at v
        cc_tangent = cc - dot_cv * v
        # Gradient direction (to decrease r, we move in the direction of cc_tangent)
        # Actually, we want to move v to reduce r, so we move v toward the direction
        # that increases the angular distance from cc.
        # dr/dv = -cc_tangent / sin(r), so to decrease r, move in direction +cc_tangent
        grad[idx] = cc_tangent / sin_r
    
    return max_radius, grad


def _gradient_descent(pts, max_iters=500, lr=0.01, time_budget=30.0):
    """
    Gradient descent on the sphere to minimize the covering radius.
    """
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        radius, grad = _cover_radius_with_grad(pts)
        
        if radius >= best_radius - 1e-10:
            # No improvement possible from this gradient
            break
        
        # Update: move each point in the direction of the gradient
        new_pts = pts.copy()
        for i in range(len(pts)):
            if np.linalg.norm(grad[i]) > 1e-12:
                # Move in the direction of the gradient (to decrease radius)
                new_pts[i] = pts[i] + lr * grad[i]
                # Normalize back to the sphere
                norm = np.linalg.norm(new_pts[i])
                if norm > 1e-10:
                    new_pts[i] /= norm
        
        new_radius = _cover_radius(new_pts)
        
        if new_radius < best_radius - 1e-10:
            pts = new_pts
            best_radius = new_radius
        else:
            # Reduce learning rate
            lr *= 0.8
            if lr < 1e-6:
                break
    
    return pts, best_radius


def _force_based_optimize(pts, max_iters=300, time_budget=30.0):
    """
    Force-based optimization: points repel each other, with stronger repulsion
    in regions that are under-covered.
    
    The force on point i is proportional to the sum of contributions from other points,
    weighted by how much "coverage" they provide.
    """
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        radius, grad = _cover_radius_with_grad(pts)
        
        if radius >= best_radius - 1e-10:
            break
        
        # Use gradient-based update with adaptive step size
        step = 0.05
        improved = False
        
        for trial in range(5):
            new_pts = pts.copy()
            for i in range(len(pts)):
                if np.linalg.norm(grad[i]) > 1e-12:
                    new_pts[i] = pts[i] + step * grad[i]
                    norm = np.linalg.norm(new_pts[i])
                    if norm > 1e-10:
                        new_pts[i] /= norm
            
            new_radius = _cover_radius(new_pts)
            
            if new_radius < best_radius - 1e-10:
                pts = new_pts
                best_radius = new_radius
                improved = True
                break
            else:
                step *= 0.5
        
        if not improved:
            break
    
    return pts, best_radius


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(N, rng):
    """Generate N random points on the unit sphere."""
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_fibonacci(N, rng, noise_scale=0.1):
    """Generate a perturbed Fibonacci spiral."""
    base = _fibonacci_spiral(N)
    noise = rng.randn(N, 3) * noise_scale
    pts = base + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius(best_pts)
    
    # Gradient-based optimization from Fibonacci start
    if time.time() < deadline - 30:
        time_budget = min(30.0, deadline - 30 - (time.time() - start_time))
        optimized, opt_radius = _force_based_optimize(pts, max_iters=500, time_budget=time_budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Try multiple random restarts with gradient-based optimization
    rng = np.random.RandomState(42)
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate different initial configurations
        if restart % 3 == 0:
            # Perturbed Fibonacci
            random_pts = _perturbed_fibonacci(N, rng, noise_scale=0.05 + 0.02 * restart)
        elif restart % 3 == 1:
            # Pure random
            random_pts = _random_sphere_points(N, rng)
        else:
            # Fibonacci with different phase
            i = np.arange(N) + 0.5 + restart * 0.1
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + restart * 0.3
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            random_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        remaining = deadline - 20 - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Estimate time budget for this restart
        time_per_restart = remaining / (max_restarts - restart)
        time_budget = min(time_per_restart * 0.8, 20.0)
        
        optimized, opt_radius = _force_based_optimize(random_pts, max_iters=300, time_budget=time_budget)
        
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final refinement: more iterations on the best configuration
    if time.time() < deadline - 5:
        time_budget = deadline - 5 - (time.time() - start_time)
        if time_budget > 5:
            optimized, opt_radius = _force_based_optimize(best_pts, max_iters=500, time_budget=time_budget)
            if opt_radius < best_radius:
                best_pts = optimized
                best_radius = opt_radius
    
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    # Normalize to ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
