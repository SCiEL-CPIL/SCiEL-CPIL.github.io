"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def _compute_covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum, over all points on the sphere, of the
    angular distance to the nearest point in the configuration. This equals
    the maximum circumradius of the facets of the convex hull (for points
    not in a closed hemisphere).
    
    We compute it via the convex hull facets: for each facet (face of the
    convex hull), the farthest point on that facet's spherical cap from the
    nearest configuration point gives the local covering radius. The global
    covering radius is the max over all facets.
    
    A more direct approach: the covering radius is half the maximum angular
    distance between any two points that are both at distance >= covering_radius
    from all configuration points. But the standard way is:
    
    For each face of the convex hull (which corresponds to a region of the sphere
    not covered more tightly), the circumradius of that face's spherical triangle
    (or polygon) is the angular distance from the face center (in the spherical
    sense) to the farthest vertex. Actually, the covering radius for a facet is
    the angular distance from the circumcenter of the facet (on the sphere) to
    the farthest vertex of that facet.
    
    The global covering radius = max over all hull facets of (circumradius of that facet).
    """
    from scipy.spatial import ConvexHull
    
    points = np.asarray(points, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: compute max nearest-neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Sample many points on the sphere
        rng = np.random.RandomState(42)
        samples = rng.randn(50000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.degrees(np.arccos(np.clip(np.max(dists), -1.0, 1.0)))
    
    # For each facet, compute the circumradius
    # A facet is defined by 3 points (for triangulated hull in 3D)
    # The circumradius of a spherical triangle with vertices a, b, c
    # is the angular distance from the circumcenter to any vertex.
    
    max_circumradius = 0.0
    for face in hull.simplices:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point equidistant from a, b, c on the sphere.
        # It lies in the plane of the triangle, at the intersection of the
        # perpendicular bisectors.
        
        # Method: find point x on the sphere such that x·a = x·b = x·c
        # This means x is in the null space of [a-b; a-c] (in the sense that
        # x·(a-b) = 0 and x·(a-c) = 0), so x is parallel to (a-b) × (a-c).
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        
        # The circumcenter direction is along cross (or -cross)
        # We need to pick the sign that gives the correct hemisphere
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-10:
            # Degenerate triangle, skip
            continue
        circum_dir = cross / norm_cross
        
        # Check which sign is correct: the circumcenter should be on the same
        # side of the triangle as the origin (for the outward-facing normal)
        # Actually, for a convex hull, the facet normal points outward.
        # The circumcenter of the spherical cap should be in the direction
        # of the facet normal (outward from the hull).
        
        # The facet normal for face (a, b, c) in CCW order is cross(b-a, c-a)
        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm < 1e-10:
            continue
        facet_normal = facet_normal / fnorm
        
        # The circumcenter should be in the direction of the facet normal
        # (outward), so check the dot product
        if np.dot(circum_dir, facet_normal) < 0:
            circum_dir = -circum_dir
        
        # The circumradius is the angular distance from circum_dir to a
        cos_angle = np.clip(np.dot(circum_dir, a), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle > max_circumradius:
            max_circumradius = angle
    
    return np.degrees(max_circumradius)


def _compute_covering_radius_fast(points):
    """Faster covering radius computation using only the hull facets."""
    from scipy.spatial import ConvexHull
    
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        samples = rng.randn(20000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.degrees(np.arccos(np.clip(np.max(dists), -1.0, 1.0)))
    
    max_circumradius = 0.0
    for face in hull.simplices:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-10:
            continue
        circum_dir = cross / norm_cross
        
        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm < 1e-10:
            continue
        facet_normal = facet_normal / fnorm
        
        if np.dot(circum_dir, facet_normal) < 0:
            circum_dir = -circum_dir
        
        cos_angle = np.clip(np.dot(circum_dir, a), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle > max_circumradius:
            max_circumradius = angle
    
    return np.degrees(max_circumradius)


def _gradient_covering_radius(points):
    """Compute approximate gradient of the covering radius w.r.t. points.
    
    The covering radius is determined by the largest facet. We compute the
    gradient by identifying which facets are active (near the max) and
    computing the gradient contribution from each.
    """
    from scipy.spatial import ConvexHull
    
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    grad = np.zeros_like(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return grad
    
    # Compute circumradius for each facet
    circumradii = []
    circumcenters = []
    for face in hull.simplices:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-10:
            circumradii.append(0.0)
            circumcenters.append(None)
            continue
        circum_dir = cross / norm_cross
        
        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm < 1e-10:
            circumradii.append(0.0)
            circumcenters.append(None)
            continue
        facet_normal = facet_normal / fnorm
        
        if np.dot(circum_dir, facet_normal) < 0:
            circum_dir = -circum_dir
        
        cos_angle = np.clip(np.dot(circum_dir, a), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        circumradii.append(angle)
        circumcenters.append((circum_dir, face))
    
    circumradii = np.array(circumradii)
    if len(circumradii) == 0:
        return grad
    
    max_cr = np.max(circumradii)
    # Consider facets within a small tolerance of the max
    threshold = max_cr - 1e-4
    active_mask = circumradii >= threshold
    
    if not np.any(active_mask):
        return grad
    
    for idx in np.where(active_mask)[0]:
        if circumcenters[idx] is None:
            continue
        circum_dir, face = circumcenters[idx]
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        
        # The circumradius r = arccos(circum_dir · a)
        # dr/da = -circum_dir / sqrt(1 - (circum_dir·a)^2)
        # But circum_dir also depends on a, b, c.
        
        # Simplified: treat circum_dir as fixed (it changes slowly)
        # and compute gradient w.r.t. a only, then project.
        
        dot_a = np.dot(circum_dir, a)
        if abs(dot_a) > 0.999999:
            continue
        denom = np.sqrt(1.0 - dot_a * dot_a)
        
        # Gradient of r w.r.t. a: -circum_dir / denom
        # But we need to project to the tangent space (perpendicular to a)
        grad_a = -circum_dir / denom
        grad_a = grad_a - np.dot(grad_a, a) * a  # project to tangent
        
        # Similarly for b and c (by symmetry)
        dot_b = np.dot(circum_dir, b)
        if abs(dot_b) > 0.999999:
            grad_b = np.zeros(3)
        else:
            denom_b = np.sqrt(1.0 - dot_b * dot_b)
            grad_b = -circum_dir / denom_b
            grad_b = grad_b - np.dot(grad_b, b) * b
        
        dot_c = np.dot(circum_dir, c)
        if abs(dot_c) > 0.999999:
            grad_c = np.zeros(3)
        else:
            denom_c = np.sqrt(1.0 - dot_c * dot_c)
            grad_c = -circum_dir / denom_c
            grad_c = grad_c - np.dot(grad_c, c) * c
        
        weight = 1.0 / np.sum(active_mask)
        grad[face[0]] += weight * grad_a
        grad[face[1]] += weight * grad_b
        grad[face[2]] += weight * grad_c
    
    return grad


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_covering_radius(points, max_time=110.0):
    """Optimize points to minimize covering radius using gradient descent
    with adaptive step size and periodic re-identification of active facets."""
    
    start_time = time.time()
    points = np.array(points, dtype=np.float64)
    n = len(points)
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_points = points.copy()
    best_radius = _compute_covering_radius_fast(points)
    
    # Learning rate schedule
    lr = 0.01
    lr_decay = 0.999
    min_lr = 1e-6
    
    step = 0
    while time.time() - start_time < max_time:
        # Compute gradient
        grad = _gradient_covering_radius(points)
        
        # Check if gradient is meaningful
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step size
        step_size = lr
        new_points = points - step_size * grad
        
        # Project back to sphere
        new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / new_norms
        
        # Evaluate new radius
        new_radius = _compute_covering_radius_fast(new_points)
        
        if new_radius < best_radius:
            best_points = new_points.copy()
            best_radius = new_radius
            points = new_points
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.05)
        else:
            # Decrease learning rate
            lr *= 0.8
            if lr < min_lr:
                break
        
        step += 1
    
    return best_points, best_radius


def _local_search(points, max_time=60.0):
    """Local search: try small perturbations of individual points."""
    start_time = time.time()
    points = np.array(points, dtype=np.float64)
    n = len(points)
    best_radius = _compute_covering_radius_fast(points)
    best_points = points.copy()
    
    rng = np.random.RandomState(12345)
    
    while time.time() - start_time < max_time:
        # Pick a random point to perturb
        idx = rng.randint(n)
        
        # Generate a random perturbation direction
        perturbation = rng.randn(3)
        # Project to tangent space
        tangent = perturbation - np.dot(perturbation, points[idx]) * points[idx]
        tnorm = np.linalg.norm(tangent)
        if tnorm < 1e-10:
            continue
        tangent /= tnorm
        
        # Try several step sizes
        improved = False
        for step_size in [0.05, 0.02, 0.01, 0.005, 0.002]:
            new_points = points.copy()
            new_points[idx] = points[idx] + step_size * tangent
            new_norm = np.linalg.norm(new_points[idx])
            new_points[idx] /= new_norm
            
            new_radius = _compute_covering_radius_fast(new_points)
            if new_radius < best_radius - 1e-6:
                points = new_points
                best_points = points.copy()
                best_radius = new_radius
                improved = True
                break
        
        if not improved:
            # Try the opposite direction
            for step_size in [0.05, 0.02, 0.01, 0.005, 0.002]:
                new_points = points.copy()
                new_points[idx] = points[idx] - step_size * tangent
                new_norm = np.linalg.norm(new_points[idx])
                new_points[idx] /= new_norm
                
                new_radius = _compute_covering_radius_fast(new_points)
                if new_radius < best_radius - 1e-6:
                    points = new_points
                    best_points = points.copy()
                    best_radius = new_radius
                    improved = True
                    break
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Phase 1: Gradient-based optimization
    points, radius = _optimize_covering_radius(points, max_time=60.0)
    
    # Check time
    if time.time() > deadline - 40.0:
        # Not enough time for local search
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    # Phase 2: Local search
    points, radius = _local_search(points, max_time=35.0)
    
    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                # Nudge apart
                points[j] += 1e-6 * np.random.randn(3)
                points[j] /= np.linalg.norm(points[j])
    
    return points

# EVOLVE-BLOCK-END
