"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This equals the maximum circumradius of the
    facets of the convex hull (each facet's circumscribed circle on the sphere).
    
    We compute this by finding, for each point, its Voronoi cell on the sphere,
    and the farthest point in that cell. Practically, we use the convex hull approach:
    the covering radius is determined by the facets of the convex hull of the points
    in R^3. For each facet, the center of the circumscribed circle (projected to sphere)
    gives the farthest point.
    
    Alternative: sample many points and find max min-distance. But for precision,
    we use the hull-based approach.
    """
    from scipy.spatial import ConvexHull
    
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
        # For each facet, find the circumcenter in the plane, project to sphere
        # The angular distance from the facet's circumcenter (on sphere) to the facet vertices
        # gives the local covering radius contribution.
        
        # More direct: for each facet, the farthest point on the sphere in that facet's
        # Voronoi region is the circumcenter of the facet (if it projects inside the facet).
        # The angular radius is the angle between the circumcenter and any vertex.
        
        max_rad = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]  # 3 points (or 4 for tetrahedra, but on sphere we have triangles mostly)
            # For a triangle on the sphere, the circumcenter is the point on the sphere
            # equidistant from all three vertices.
            # In 3D, the circumcenter of a triangle lies in the plane of the triangle.
            # We find it, normalize, and compute the angle to each vertex.
            
            a, b, c = tri[0], tri[1], tri[2]
            
            # Circumcenter of triangle abc in 3D
            # Solve for point x = a + s*(b-a) + t*(c-a) such that |x-a|^2 = |x-b|^2 = |x-c|^2
            # This gives: 2*x dot (b-a) = |b-a|^2, 2*x dot (c-a) = |c-a|^2
            
            ab = b - a
            ac = c - a
            A_mat = np.array([ab, ac])  # 2x3
            rhs = np.array([np.dot(ab, ab) / 2.0, np.dot(ac, ac) / 2.0])
            
            # x = a + A_mat^T @ [s, t]
            # A_mat @ [s,t] = rhs  (in the plane)
            # Actually: x = a + s*ab + t*ac
            # |x-a|^2 = |x-b|^2 => 2*s*|ab|^2 + 2*t*(ab.ac) - |ab|^2 = 0
            # |x-a|^2 = |x-c|^2 => 2*s*(ab.ac) + 2*t*|ac|^2 - |ac|^2 = 0
            
            ab_ac = np.dot(ab, ac)
            ab_ab = np.dot(ab, ab)
            ac_ac = np.dot(ac, ac)
            
            M = np.array([[ab_ab, ab_ac],
                          [ab_ac, ac_ac]])
            r = np.array([ab_ab / 2.0, ac_ac / 2.0])
            
            try:
                st = np.linalg.solve(M, r)
            except np.linalg.LinAlgError:
                continue
            
            x = a + st[0] * ab + st[1] * ac
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-10:
                continue
            x = x / norm_x
            
            # Angular distance from x to each vertex
            d1 = np.arccos(np.clip(np.dot(x, a), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(x, b), -1.0, 1.0))
            d3 = np.arccos(np.clip(np.dot(x, c), -1.0, 1.0))
            
            # Check if x is actually the farthest point in the Voronoi cell for this facet.
            # For now, take the max of these three distances as the local radius.
            local_rad = max(d1, d2, d3)
            if local_rad > max_rad:
                max_rad = local_rad
        
        return max_rad
    
    except Exception:
        # Fallback: dense sampling
        return _covering_radius_sample(pts)


def _covering_radius_sample(pts, n_samples=50000):
    """Fallback: sample the sphere and compute max min-angular distance."""
    pts = np.asarray(pts, dtype=float)
    # Generate random points on sphere
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    phi = np.arccos(np.random.uniform(-1, 1, n_samples))
    samples = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # Compute min distance from each sample to nearest point
    # Use batched dot products
    max_min_dist = 0.0
    batch_size = 10000
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        s = samples[start:end]
        # Dot product matrix: (batch, N)
        dots = s @ pts.T
        min_dist = np.arccos(np.clip(np.max(dots, axis=1), -1.0, 1.0))
        batch_max = np.max(min_dist)
        if batch_max > max_min_dist:
            max_min_dist = batch_max
    return max_min_dist


def _smooth_covering_radius(pts, temperature=0.01):
    """
    Smooth surrogate for the covering radius using log-sum-exp on angular distances.
    For optimization, we minimize a smooth approximation of the max.
    """
    pts = np.asarray(pts, dtype=float)
    # Sample points on sphere
    n_samples = 20000
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    phi = np.arccos(np.random.uniform(-1, 1, n_samples))
    samples = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # Compute angular distances from each sample to each point
    # dots[i, j] = dot(sample_i, point_j)
    dots = samples @ pts.T
    # Angular distance = arccos(dot)
    # For smooth max, use log-sum-exp on exp(-dist/temperature)
    # Actually, we want to minimize the max distance, so we use:
    # smooth_max(dist) ≈ -temperature * log(sum(exp(-dist/temperature)))
    
    dists = np.arccos(np.clip(dots, -1.0, 1.0))  # (n_samples, N)
    # For each sample, the min distance to any point is what matters for covering
    min_dists = np.min(dists, axis=1)  # (n_samples,)
    
    # Smooth max of min_dists
    scaled = -min_dists / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = -temperature * log_sum
    
    return smooth_max


def _lloyd_relaxation(pts, n_iter=100, time_limit=None, start_time=None):
    """
    Lloyd's algorithm for spherical k-means / covering.
    Each point moves to the centroid of its Voronoi cell.
    """
    pts = np.array(pts, dtype=float)
    n = len(pts)
    
    if start_time is None:
        start_time = time.time()
    
    for it in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Assign each point to nearest site (using a grid of sphere points)
        # For efficiency, use a set of representative points
        n_samples = 20000
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        phi = np.arccos(np.random.uniform(-1, 1, n_samples))
        samples = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Find nearest site for each sample
        dots = samples @ pts.T  # (n_samples, n)
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroid for each site
        new_pts = np.zeros_like(pts)
        for k in range(n):
            mask = nearest == k
            if np.sum(mask) > 0:
                centroid = np.sum(samples[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_pts[k] = centroid / norm
                else:
                    new_pts[k] = pts[k]
            else:
                new_pts[k] = pts[k]
        
        pts = _normalize(new_pts)
    
    return pts


def _gradient_descent(pts, n_iter=200, lr=0.01, time_limit=None, start_time=None):
    """
    Gradient descent on the smooth covering radius surrogate.
    """
    pts = np.array(pts, dtype=float)
    n = len(pts)
    
    if start_time is None:
        start_time = time.time()
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for it in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute gradient of smooth covering radius
        # Use finite differences or analytical gradient
        # For simplicity, use a few random perturbations to estimate gradient
        
        # Sample-based gradient estimation
        n_samples = 5000
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        phi = np.arccos(np.random.uniform(-1, 1, n_samples))
        samples = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        dots = samples @ pts.T  # (n_samples, n)
        nearest = np.argmax(dots, axis=1)
        
        # For each sample, the gradient w.r.t. the nearest point
        # The objective is max over samples of arccos(dot(sample, nearest_point))
        # Gradient of arccos(x) = -1/sqrt(1-x^2)
        # Gradient of dot(sample, p) w.r.t. p = sample
        
        grad = np.zeros_like(pts)
        for k in range(n):
            mask = nearest == k
            if np.sum(mask) == 0:
                continue
            s_masked = samples[mask]
            d_masked = dots[mask, k]
            # Gradient of -arccos(d) w.r.t. p_k = -(-1/sqrt(1-d^2)) * s = s/sqrt(1-d^2)
            # But we want to minimize the max, so we push the nearest point away from samples
            # where it's the nearest and the distance is large.
            # Actually, to reduce the covering radius, we want to move points toward
            # the farthest points in their Voronoi cells.
            
            # Simplified: move each point toward the centroid of samples it's nearest to
            centroid = np.mean(s_masked, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                grad[k] = centroid / norm
            else:
                grad[k] = pts[k]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.where(grad_norm < 1e-10, 1.0, grad_norm)
        grad = grad / grad_norm
        
        # Update
        pts = pts + lr * grad
        pts = _normalize(pts)
        
        # Check improvement
        current_radius = _covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    return best_pts


def _refine(pts, time_budget=60.0):
    """
    Refine a configuration using multiple methods.
    """
    start_time = time.time()
    
    # Step 1: Lloyd relaxation
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        pts = _lloyd_relaxation(pts, n_iter=50, time_limit=remaining * 0.3, start_time=start_time)
    
    # Step 2: Gradient descent on smooth surrogate
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        pts = _gradient_descent(pts, n_iter=100, lr=0.005, time_limit=remaining * 0.5, start_time=start_time)
    
    # Step 3: Final Lloyd polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 3:
        pts = _lloyd_relaxation(pts, n_iter=30, time_limit=remaining * 0.7, start_time=start_time)
    
    return pts


def _generate_fibonacci(n):
    """Golden-angle Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _generate_random(n, seed=0):
    """Random points on sphere."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = np.arccos(rng.uniform(-1, 1, n))
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])


def _generate_lattice(n):
    """Approximate icosahedral lattice-based points."""
    # Start with icosahedron vertices and subdivide
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Subdivide to get more points
    # Use recursive subdivision of triangles
    points = ico.copy()
    # For now, just use Fibonacci as a good approximation
    return _generate_fibonacci(n)


def construct():
    """
    Main entry point. Returns an array of shape (50, 3) of unit vectors.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_generate_fibonacci(N))
    
    # 2. Random starts with different seeds
    for seed in range(5):
        inits.append(_generate_random(N, seed=seed))
    
    # 3. Slightly perturbed Fibonacci
    for seed in range(3):
        rng = np.random.RandomState(seed + 100)
        fib = _generate_fibonacci(N)
        noise = rng.normal(0, 0.1, size=fib.shape)
        perturbed = fib + noise
        inits.append(_normalize(perturbed))
    
    # Refine each initial configuration
    best_pts = None
    best_radius = np.inf
    
    for idx, init in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate time per configuration
        time_per_config = min(remaining / max(1, len(inits) - idx), 25.0)
        
        refined = _refine(init, time_budget=time_per_config)
        
        # Evaluate
        radius = _covering_radius(refined)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = refined.copy()
        
        # Check if we're close to the known best
        if best_radius < np.radians(18.31):
            break
    
    # Final check and return
    if best_pts is None:
        best_pts = _generate_fibonacci(N)
    
    # Ensure points are on unit sphere
    best_pts = _normalize(best_pts)
    
    # Final validation: ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                # Perturb
                rng = np.random.RandomState(i * 1000 + j)
                best_pts[j] = best_pts[j] + rng.normal(0, 0.01, size=3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts

# EVOLVE-BLOCK-END
