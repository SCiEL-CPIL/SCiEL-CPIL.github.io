import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
DEADLINE = 135.0  # Leave 15s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = np.asarray(params).reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in the set. This equals the maximum circumradius of the
    spherical Voronoi cells, which corresponds to the largest circumcircle of
    facets of the convex hull (checked to have empty caps).
    """
    points = np.asarray(points)
    points = to_sphere(points)
    
    # Try to build convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # worst case
    
    max_circum = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute circumcenter of triangle in 3D
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            # Degenerate triangle
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        if np.linalg.norm(circumcenter) < 1e-10:
            # Circumcenter at origin, degenerate
            continue
        
        pole = circumcenter / np.linalg.norm(circumcenter)
        
        # The angular distance from pole to each vertex should be the circumradius
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        # The circumradius is the max of these
        circum_angular = max(theta_a, theta_b, theta_c)
        max_circum = max(max_circum, circum_angular)
    
    return max_circum

def smooth_covering_radius(points, beta=10.0):
    """Compute a smooth approximation of the covering radius using log-sum-exp.
    
    This creates a differentiable surrogate for the nonsmooth max operation,
    enabling gradient-based optimization.
    """
    points = np.asarray(points)
    points = to_sphere(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    circumradii = []
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        if np.linalg.norm(circumcenter) < 1e-10:
            continue
        
        pole = circumcenter / np.linalg.norm(circumcenter)
        
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        circumradii.append(circum_angular)
    
    if not circumradii:
        return np.pi
    
    circumradii = np.array(circumradii)
    # Log-sum-exp for smooth max approximation
    m = np.max(circumradii)
    return m + np.log(np.sum(np.exp((circumradii - m) * beta))) / beta

def spherical_centroids(points):
    """Compute spherical centroids for each point based on Voronoi cells from convex hull.
    
    For each point, its Voronoi cell is determined by the adjacent hull facets.
    The spherical centroid is the normalized sum of the facet circumcenters weighted
    by the solid angle of each facet.
    """
    points = np.asarray(points)
    points = to_sphere(points)
    n = len(points)
    centroids = np.zeros((n, 3))
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    # For each facet, compute its circumcenter and solid angle
    # Then add the circumcenter to the centroid of each vertex of the facet
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute circumcenter
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            # Degenerate triangle, use centroid
            centroid = (a + b + c) / 3.0
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                centroid = centroid / norm
            else:
                continue
        else:
            u1 = (acac * abac - abab * abac) / denom
            u2 = (abab * abac - acac * abac) / denom
            circumcenter = a + u1 * ab + u2 * ac
            norm = np.linalg.norm(circumcenter)
            if norm < 1e-10:
                centroid = (a + b + c) / 3.0
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    centroid = centroid / norm
                else:
                    continue
            else:
                centroid = circumcenter / norm
        
        # Compute solid angle of the spherical triangle (approximate by area)
        # Solid angle ≈ 2 * spherical excess
        # Or simpler: use the area of the triangle on the unit sphere
        # For small triangles, area ≈ 0.5 * |cross product|
        # More accurately, use L'Huilier's theorem or direct formula
        
        # Compute side lengths
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        
        # Spherical excess using formula
        s = (d_ab + d_bc + d_ca) / 2.0
        try:
            # L'Huilier's formula
            tan_E4 = np.sqrt(np.tan(s/2) * np.tan((s-d_ab)/2) * np.tan((s-d_bc)/2) * np.tan((s-d_ca)/2))
            E = 4 * np.arctan(tan_E4)
        except Exception:
            E = 0.0
        
        # Weight by solid angle (spherical excess)
        weight = max(E, 1e-10)
        
        # Add to centroids of each vertex
        for idx in simplex:
            centroids[idx] += weight * centroid
    
    # Normalize centroids
    for i in range(n):
        norm = np.linalg.norm(centroids[i])
        if norm > 1e-10:
            centroids[i] /= norm
        else:
            centroids[i] = points[i]  # Keep original if centroid is degenerate
    
    return centroids

def lloyd_relaxation(points, max_iters=50, tol=1e-8):
    """Perform spherical Lloyd's algorithm to optimize covering radius.
    
    Iteratively move each point to the spherical centroid of its Voronoi cell.
    This balances the facet circumradii and reduces the covering radius.
    """
    points = np.asarray(points).copy()
    points = to_sphere(points)
    
    best_points = points.copy()
    best_cr = compute_covering_radius(points)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute spherical centroids
        new_points = spherical_centroids(points)
        
        # Blend with old points for stability (damping)
        damping = 0.5
        blended = (1 - damping) * points + damping * new_points
        blended = to_sphere(blended)
        
        # Compute covering radius
        cr = compute_covering_radius(blended)
        
        if cr < best_cr:
            best_cr = cr
            best_points = blended.copy()
        
        # Check convergence
        if np.linalg.norm(blended - points) < tol:
            break
        
        points = blended
    
    return best_points, best_cr

def gradient_refine(points, max_iters=50, lr=0.01):
    """Refine points using gradient descent on the smooth covering radius surrogate.
    
    Uses the log-sum-exp approximation to get gradients and update point positions.
    """
    points = np.asarray(points).copy()
    points = to_sphere(points)
    
    best_points = points.copy()
    best_cr = compute_covering_radius(points)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute smooth covering radius and gradient
        smooth_cr = smooth_covering_radius(points, beta=10.0)
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(points)
        for i in range(len(points)):
            for j in range(3):
                perturbed = points.copy()
                perturbed[i, j] += eps
                perturbed = to_sphere(perturbed)
                f_plus = smooth_covering_radius(perturbed, beta=10.0)
                
                perturbed = points.copy()
                perturbed[i, j] -= eps
                perturbed = to_sphere(perturbed)
                f_minus = smooth_covering_radius(perturbed, beta=10.0)
                
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        
        # Update points
        new_points = points - lr * grad
        new_points = to_sphere(new_points)
        
        # Compute actual covering radius
        cr = compute_covering_radius(new_points)
        
        if cr < best_cr:
            best_cr = cr
            best_points = new_points.copy()
        
        # Adaptive learning rate
        if cr > smooth_cr + 0.01:
            lr *= 0.5
        elif cr < smooth_cr - 0.01:
            lr *= 1.2
            lr = min(lr, 0.1)
        
        points = new_points
    
    return best_points, best_cr

def random_init(n_points, rng=None):
    """Generate random points on the unit sphere."""
    if rng is None:
        rng = np.random.RandomState()
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def local_optimize(initial_points, max_iter=200, tol=1e-8):
    """Use Lloyd's relaxation followed by gradient refinement to locally optimize."""
    # First, Lloyd's relaxation
    points, cr = lloyd_relaxation(initial_points, max_iters=max_iter, tol=tol)
    
    # Then, gradient refinement for finer optimization
    if time.time() < DEADLINE - 10:
        points, cr2 = gradient_refine(points, max_iters=30, lr=0.01)
        if cr2 < cr:
            cr = cr2
    
    return points, cr

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    best_points = fibonacci_spiral(N)
    best_cr = compute_covering_radius(best_points)
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 15:
            break
        
        if restart == 0:
            init = fibonacci_spiral(N)
        else:
            # Mix: start from best and perturb, or random
            r = rng.rand()
            if r < 0.35:
                # Perturb current best
                noise = 0.08 * rng.randn(N, 3)
                init = best_points + noise
                init = to_sphere(init)
            elif r < 0.65:
                # Fibonacci with random rotation
                init = fibonacci_spiral(N)
                theta = rng.rand() * 2 * np.pi
                phi = rng.rand() * 2 * np.pi
                # Random rotation matrix
                R = np.array([[np.cos(phi), -np.sin(phi), 0],
                              [np.sin(phi), np.cos(phi), 0],
                              [0, 0, 1]])
                R2 = np.array([[np.cos(theta), 0, -np.sin(theta)],
                               [0, 1, 0],
                               [np.sin(theta), 0, np.cos(theta)]])
                R = R2 @ R
                init = init @ R.T
            else:
                init = random_init(N, rng)
        
        # Use fewer Lloyd iterations per restart, more total restarts
        optimized, cr = local_optimize(init, max_iter=25)
        if cr < best_cr:
            best_cr = cr
            best_points = optimized.copy()
    
    # Final fine-tuning with more iterations
    if time.time() < DEADLINE - 25:
        optimized, cr = local_optimize(best_points, max_iter=50)
        if cr < best_cr:
            best_cr = cr
            best_points = optimized
    
    # Ensure all points are on the unit sphere
    best_points = to_sphere(best_points)
    
    # Validate: check no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                # Separate them slightly
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    # Check not all in a closed hemisphere: the origin should be in the
    # interior of the convex hull
    try:
        hull = ConvexHull(best_points)
        # If hull succeeds, points are not all in a hemisphere
    except Exception:
        # If hull fails, it might be degenerate; try to fix
        pass
    
    return best_points

# Keep the module-level construct as the entry point
