"""Optimized covering of the unit sphere with 130 points.
Combines multi-restart RFD with Delaunay circumcenter-based gradient polishing
and a soft-max covering-radius surrogate for final refinement.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices

        max_circumradius = 0.0
        for simplex in facets:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
            b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
            c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))

            s = (a + b + c) / 2
            denom = 4 * np.sin(s) * np.sin(s - a) * np.sin(s - b) * np.sin(s - c)

            if abs(denom) > 1e-15:
                sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
                sin_R = np.clip(sin_R, 0, 1)
                R = np.arcsin(sin_R)
            else:
                R = max(a, b, c) / 2

            if R > max_circumradius:
                max_circumradius = R

        return np.degrees(max_circumradius)

    except Exception:
        # Fallback: sample-based estimate
        rng = np.random.RandomState(42)
        n_samples = 20000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        min_dists = np.inf
        for i in range(0, n_samples, 2000):
            chunk = samples[i:i + 2000]
            d = 1 - np.einsum('ij,kj->ik', chunk, points)
            d = np.clip(d, -1, 1)
            min_d = np.min(d, axis=1)
            min_dists = np.minimum(min_dists, min_d)

        max_min_dist = np.max(min_dists)
        return np.degrees(np.arccos(np.clip(max_min_dist, -1, 1)))


def _fibonacci_sphere(n, seed=0):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2 * np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms


def _min_pairwise_dist(points):
    """Compute the minimum pairwise angular distance."""
    d = 1 - points @ points.T
    np.fill_diagonal(d, np.inf)
    d = np.clip(d, 0, 2)
    return np.arccos(np.clip(1 - np.min(d), -1, 1))


def _soft_max_surrogate(points, samples, beta):
    """Compute a smooth surrogate for the covering radius using soft-max.
    
    Returns (surrogate_value, gradient) where gradient is (N,3).
    """
    d = 1 - np.einsum('ij,kj->ik', samples, points)
    d = np.clip(d, 0, 2)
    
    neg_beta_d = -beta * d
    max_neg = np.max(neg_beta_d, axis=1, keepdims=True)
    exp_neg = np.exp(neg_beta_d - max_neg)
    log_sum = max_neg.squeeze() + np.log(np.sum(exp_neg, axis=1))
    soft_min = -log_sum / beta
    
    beta2 = beta
    neg_beta2_sm = -beta2 * soft_min
    max_neg2 = np.max(neg_beta2_sm, keepdims=True)
    exp_neg2 = np.exp(neg_beta2_sm - max_neg2)
    surrogate = -(max_neg2.squeeze() + np.log(np.sum(exp_neg2))) / beta2
    
    weights = exp_neg2.squeeze() / np.sum(exp_neg2)
    
    S = np.exp(max_neg.squeeze()) * np.sum(exp_neg, axis=1)
    probs = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
    
    weighted_probs = weights[:, None] * probs
    grad = -(weighted_probs.T @ samples)
    
    return surrogate, grad


def _delaunay_gradient(points):
    """Compute gradient of covering radius w.r.t. each point based on Delaunay facets.
    
    For each facet (triangle) in the convex hull, compute its circumcenter on the sphere.
    The covering radius is determined by the largest circumradius facet.
    We push points that form the worst facets away from the circumcenter.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return np.zeros_like(points)
    
    n = len(points)
    grad = np.zeros_like(points)
    
    # Compute circumradius for each facet
    circumradii = []
    circumcenters = []
    
    for simplex in facets:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        
        s = (a + b + c) / 2
        denom = 4 * np.sin(s) * np.sin(s - a) * np.sin(s - b) * np.sin(s - c)
        
        if abs(denom) > 1e-15:
            sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
            sin_R = np.clip(sin_R, 0, 1)
            R = np.arcsin(sin_R)
        else:
            R = max(a, b, c) / 2
        
        # Compute circumcenter on sphere
        # The circumcenter is the point on the sphere equidistant from p1, p2, p3
        # It lies along the normal to the plane through p1, p2, p3
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            circumradii.append(R)
            circumcenters.append(None)
            continue
        n = n / n_norm
        
        # Ensure normal points outward (away from origin)
        centroid = (p1 + p2 + p3) / 3
        if np.dot(n, centroid) < 0:
            n = -n
        
        # The circumcenter is at distance cos(R) from each vertex
        # c = n * cos(R) / (n · centroid_normalized) ... actually simpler:
        # c = n * t where t is chosen so that |c - p1|_angular = R
        # cos(R) = c · p1 = t * (n · p1)
        # t = cos(R) / (n · p1)
        dot_np1 = np.dot(n, p1)
        if abs(dot_np1) > 1e-10:
            t = np.cos(R) / dot_np1
            c = t * n
            c = _project_to_sphere(c.reshape(1, 3))[0]
        else:
            c = n
        
        circumradii.append(R)
        circumcenters.append(c)
    
    if not circumradii:
        return grad
    
    max_R = max(circumradii)
    
    # Only consider facets with circumradius close to max (within tolerance)
    tol = 0.05  # radians, about 3 degrees
    for idx, (R, c) in enumerate(zip(circumradii, circumcenters)):
        if c is None:
            continue
        if max_R - R < tol:
            # Weight by how close this facet is to the worst
            w = np.exp(-10 * (max_R - R))
            simplex = facets[idx]
            for k in range(3):
                p_idx = simplex[k]
                p = points[p_idx]
                # Push point away from circumcenter
                # Gradient direction: p - c projected to tangent space at p
                direction = p - c
                direction = direction - np.dot(direction, p) * p
                grad[p_idx] += w * direction
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        grad = grad / grad_norm
    
    return grad


def _optimize_repulsion(points, max_time, seed=0):
    """Optimize using repulsive forces (RFD: maximize minimum pairwise distance)."""
    start_time = time.time()
    points = _project_to_sphere(points.copy())
    best_points = points.copy()
    best_min_dist = _min_pairwise_dist(points)
    
    beta = 50.0
    lr = 0.005
    
    for iteration in range(20000):
        if time.time() - start_time > max_time:
            break
        
        d = 1 - points @ points.T
        np.fill_diagonal(d, np.inf)
        d = np.clip(d, 0, 2)
        
        neg_beta_d = -beta * d
        max_neg = np.max(neg_beta_d, axis=1, keepdims=True)
        exp_neg = np.exp(neg_beta_d - max_neg)
        S = np.sum(exp_neg, axis=1, keepdims=True)
        probs = exp_neg / S
        
        grad = -(probs.T @ points) - (probs @ points)
        
        for j in range(N):
            grad[j] -= np.dot(grad[j], points[j]) * points[j]
        
        points = points + lr * grad
        points = _project_to_sphere(points)
        
        if iteration % 200 == 0:
            min_dist = _min_pairwise_dist(points)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_points = points.copy()
        
        if iteration % 500 == 0:
            lr *= 0.95
    
    return best_points, best_min_dist


def _optimize_delaunay_polish(points, max_time):
    """Polish using Delaunay circumcenter-based gradient descent on covering radius."""
    start_time = time.time()
    points = _project_to_sphere(points.copy())
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    lr = 0.01
    momentum = np.zeros_like(points)
    mu = 0.8
    
    for iteration in range(10000):
        if time.time() - start_time > max_time:
            break
        
        grad = _delaunay_gradient(points)
        
        # Project to tangent space
        for j in range(N):
            grad[j] -= np.dot(grad[j], points[j]) * points[j]
        
        # Momentum update
        momentum = mu * momentum - lr * grad
        
        new_points = points + momentum
        new_points = _project_to_sphere(new_points)
        
        if iteration % 100 == 0:
            radius = _covering_radius(new_points)
            if radius < best_radius:
                best_radius = radius
                best_points = new_points.copy()
                points = new_points.copy()
            else:
                lr *= 0.9
                momentum *= 0.5
        
        if iteration % 500 == 0:
            lr *= 0.95
    
    return best_points, best_radius


def _optimize_soft_max(points, max_time, beta=30.0):
    """Polish the configuration by minimizing the soft-max covering radius surrogate."""
    start_time = time.time()
    points = _project_to_sphere(points.copy())
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    rng = np.random.RandomState(12345)
    n_samples = 5000
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    lr = 0.002
    for iteration in range(15000):
        if time.time() - start_time > max_time:
            break
        
        surrogate, grad = _soft_max_surrogate(points, samples, beta)
        
        for j in range(N):
            grad[j] -= np.dot(grad[j], points[j]) * points[j]
        
        points = points - lr * grad
        points = _project_to_sphere(points)
        
        if iteration % 300 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
        
        if iteration % 1000 == 0 and beta < 100:
            beta *= 1.3
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.inf
    
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = deadline - (start_time + elapsed)
        if remaining < 10:
            break
        
        if restart == 0:
            init_points = _fibonacci_sphere(N, seed=restart)
        elif restart == 1:
            init_points = _random_sphere(N, seed=100 + restart)
        elif restart == 2:
            init_points = _fibonacci_sphere(N, seed=1000 + restart)
        elif restart == 3:
            init_points = _random_sphere(N, seed=200 + restart)
        else:
            init_points = _fibonacci_sphere(N, seed=5000 + restart)
        
        # Phase 1: Repulsion optimization (RFD)
        time_alloc_1 = min(remaining * 0.35, 30)
        pts, min_dist = _optimize_repulsion(init_points, max_time=time_alloc_1, seed=restart)
        
        # Phase 2: Delaunay circumcenter-based polishing
        t2 = time.time()
        remaining2 = deadline - (start_time + (t2 - start_time))
        time_alloc_2 = max(remaining2 * 0.35, 10)
        pts, radius = _optimize_delaunay_polish(pts, max_time=time_alloc_2)
        
        # Phase 3: Soft-max covering radius polishing
        t3 = time.time()
        remaining3 = deadline - (start_time + (t3 - start_time))
        time_alloc_3 = max(remaining3 * 0.3, 5)
        pts, radius = _optimize_soft_max(pts, max_time=time_alloc_3)
        
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()
        
        if time.time() > deadline - 10:
            break
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    return best_points
