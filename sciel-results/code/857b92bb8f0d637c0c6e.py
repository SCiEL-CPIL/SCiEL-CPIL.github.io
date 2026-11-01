import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(p):
    """
    Compute convex hull facets and their circumcenters/radii (vectorized).
    Returns facets, circumcenters, radii.
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    # Vectorized circumcenter computation
    a = p[facets[:, 0]]  # (F, 3)
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b  # (F, 3)
    v2 = b - c
    n = np.cross(v1, v2)  # (F, 3)
    norm_n = np.linalg.norm(n, axis=1)  # (F,)
    
    # Handle degenerate facets
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    return facets, circumcenters, radii

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    facets, circumcenters, radii = _compute_hull_data(config)
    if radii is None:
        # Fallback to sampling if hull fails
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    return np.degrees(np.max(radii))

def _soft_max_objective(params, beta=200.0):
    """
    Soft-max surrogate: approximates max(radii) via log-sum-exp.
    As beta -> inf, this approaches the true max.
    This is smoother than sum(radii**power) for the minimax problem.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    # Soft-max: (1/beta) * log(sum(exp(beta * radii)))
    # For numerical stability, subtract max
    r_max = np.max(radii)
    if r_max < 1e-15:
        return r_max
    shifted = beta * (radii - r_max)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
    return log_sum_exp

def _power_objective(params, power=2.0):
    """Sum of radii^power as objective."""
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
    return np.sum(radii**power)

def _repulsion_relax(points, alpha_init=0.1, decay=0.998, max_iter=500, deadline=None):
    """
    Repulsion Relaxation (Thomson Problem solver).
    Minimize sum 1/||p_i - p_j||^2 to spread points evenly.
    """
    points = points.copy()
    alpha = alpha_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Calculate forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane: F_tan = F - (F . p) p
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update
        points = points + alpha * forces_tan
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Decay alpha
        alpha *= decay
        
    return points

def _lloyd_step(config, hull):
    """
    Perform one Lloyd iteration.
    Moves each point towards the spherical mean of the circumcenters of the faces for which it is the nearest site.
    """
    new_config = config.copy()
    
    # Compute circumcenters for all faces
    circumcenters = []
    face_indices = []
    for simplex in hull.simplices:
        pts = config[simplex]
        c = _circumcenter(pts[0], pts[1], pts[2])
        circumcenters.append(c)
        face_indices.append(simplex)
        
    circumcenters = np.array(circumcenters)
    
    # For each circumcenter, find the nearest site in the config
    # Dots: (num_faces, N)
    dots = circumcenters @ config.T
    nearest_idx = np.argmax(dots, axis=1)
    
    # Accumulate targets for each site
    # We want to move site i to the mean of all circumcenters where nearest_idx == i
    sums = np.zeros((N, 3))
    counts = np.zeros(N)
    
    for i, idx in enumerate(nearest_idx):
        sums[idx] += circumcenters[i]
        counts[idx] += 1
        
    for i in range(N):
        if counts[i] > 0:
            target = sums[i]
            n = np.linalg.norm(target)
            if n > 1e-10:
                target = target / n
                # Blend with current position to ensure stability
                new_config[i] = target
            else:
                new_config[i] = config[i]
        else:
            new_config[i] = config[i]
            
    return new_config

def _circumcenter(p1, p2, p3):
    """
    Compute the circumcenter of a spherical triangle defined by unit vectors p1, p2, p3.
    Returns the unit vector pointing to the circumcenter.
    """
    v1 = p1 - p2
    v2 = p2 - p3
    c = np.cross(v1, v2)
    n = np.linalg.norm(c)
    if n < 1e-10:
        return _normalize(p1 + p2 + p3)
    c = c / n
    
    avg = _normalize(p1 + p2 + p3)
    if np.dot(c, avg) < 0:
        c = -c
    return c

def _local_optimize_lloyd(config, deadline):
    """
    Optimize using Lloyd's algorithm.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    
    it = 0
    while time.time() < deadline - 5.0:
        try:
            hull = ConvexHull(config)
        except Exception:
            config += 1e-5 * np.random.randn(N, 3)
            config = np.array([_normalize(p) for p in config])
            continue
            
        new_config = _lloyd_step(config, hull)
        
        # Check for improvement
        new_score = _covering_radius_deg_hull(new_config)
        
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_config = new_config.copy()
            config = new_config
        else:
            # If no improvement, we might be in a local minimum or oscillating.
            # Add a small perturbation to escape.
            idx = np.random.randint(N)
            rand_dir = np.random.randn(3)
            rand_dir = _normalize(rand_dir)
            
            new_pt = config[idx] + 0.1 * rand_dir
            new_pt = _normalize(new_pt)
            
            old_pt = config[idx].copy()
            config[idx] = new_pt
            
            try:
                hull2 = ConvexHull(config)
                perturbed_score = _covering_radius_deg_hull(config)
                if perturbed_score < best_score:
                    best_score = perturbed_score
                    best_config = config.copy()
                else:
                    config[idx] = old_pt
            except Exception:
                config[idx] = old_pt
        
        it += 1
        if it > 2000:
            break
            
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    fib_pts = _fib_sample(N)
    initial_configs.append(fib_pts)
    
    # 2. Fibonacci with different phase offsets
    for phase in [0.37, 0.618, 0.25]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)
    
    # 3. Random points (seeded for reproducibility)
    np.random.seed(42)
    for seed_offset in range(3):
        np.random.seed(42 + seed_offset)
        rand_pts = np.random.randn(N, 3)
        rand_pts = np.array([_normalize(p) for p in rand_pts])
        initial_configs.append(rand_pts)
    
    best_config = initial_configs[0].copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Phase 1: Repulsion Relaxation to spread points
    for config in initial_configs:
        if time.time() > deadline - 30.0:
            break
        relaxed = _repulsion_relax(config, alpha_init=0.05, decay=0.995, max_iter=200, deadline=deadline - 30.0)
        score = _covering_radius_deg_hull(relaxed)
        if score < best_score:
            best_score = score
            best_config = relaxed.copy()
    
    # Phase 2: L-BFGS optimization with soft-max objective
    # Use the best configuration from Phase 1 as starting point
    # Also try starting from the original fib spiral
    
    starting_points = [best_config.copy(), _fib_sample(N)]
    
    for start_pts in starting_points:
        if time.time() > deadline - 15.0:
            break
        
        # Try different beta values for soft-max
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                res = minimize(_soft_max_objective, start_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200},
                               args=(beta,))
                
                if res.fun < _soft_max_objective(start_pts.flatten(), beta):
                    new_pts = res.x.reshape(N, 3)
                    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    new_pts = new_pts / norms
                    
                    current_radius = _covering_radius_deg_hull(new_pts)
                    if current_radius < best_score:
                        best_score = current_radius
                        best_config = new_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Lloyd's algorithm for local optimization
    if time.time() < deadline - 5.0:
        best_config = _local_optimize_lloyd(best_config, deadline)
        final_score = _covering_radius_deg_hull(best_config)
        if final_score < best_score:
            best_score = final_score
    
    # Final normalization
    best_config = np.array([_normalize(p) for p in best_config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_config[i] - best_config[j]) < 1e-6:
                best_config[j] = best_config[j] + 1e-4 * np.random.randn(3)
                best_config[j] = _normalize(best_config[j])
                
    return best_config
