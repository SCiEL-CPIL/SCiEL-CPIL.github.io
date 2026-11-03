"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r

def _fast_facets(points):
    """Compute all facet circumradii and centers efficiently.
    Returns: (radii array, centers array, simplex_indices array)
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    
    n_facets = len(hull.simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    valid = np.zeros(n_facets, dtype=bool)
    
    for fi, simplex in enumerate(hull.simplices):
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        radii[fi] = r
        centers[fi] = v
        valid[fi] = True
    
    valid_idx = np.where(valid)[0]
    return radii[valid_idx], centers[valid_idx], hull.simplices[valid_idx]

def _smooth_covering_radius(points, beta=50.0):
    """
    Smooth approximation of max facet circumradius using log-sum-exp.
    Lower beta = smoother (more like mean), higher beta = closer to max.
    """
    result = _fast_facets(points)
    if result is None:
        return _covering_radius(points)
    radii, centers, simplices = result
    if len(radii) == 0:
        return 0.0
    
    # Log-sum-exp trick for numerical stability
    r_max = np.max(radii)
    if r_max < 1e-15:
        return 0.0
    shifted = radii - r_max
    log_sum_exp = r_max + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Gradient of the smooth covering radius with respect to point positions.
    Uses the log-sum-exp formulation: f = (1/beta) * log(sum_j exp(beta * r_j))
    df/dp_i = (1/beta) * sum_j (beta * exp(beta*r_j) / sum_k exp(beta*r_k)) * dr_j/dp_i
           = sum_j w_j * dr_j/dp_i
    where w_j = exp(beta*r_j) / sum_k exp(beta*r_k)
    
    For each facet j with vertices (a,b,c) and circumcenter v_j:
    The circumradius r_j = arccos(v_j . a)
    dr_j/dp_a = -v_j / sqrt(1 - (v_j.a)^2)  (tangent component)
    
    We project gradients onto the tangent space of the sphere.
    """
    points = _normalize(points)
    result = _fast_facets(points)
    if result is None:
        return np.zeros_like(points)
    radii, centers, simplices = result
    n_facets = len(radii)
    if n_facets == 0:
        return np.zeros_like(points)
    
    # Compute softmax weights
    r_max = np.max(radii)
    if r_max < 1e-15:
        return np.zeros_like(points)
    shifted = beta * (radii - r_max)
    exp_shifted = np.exp(shifted)
    total = np.sum(exp_shifted)
    if total < 1e-30:
        return np.zeros_like(points)
    weights = exp_shifted / total  # w_j
    
    grad = np.zeros_like(points)
    
    for j in range(n_facets):
        w = weights[j]
        if w < 1e-15:
            continue
        v = centers[j]  # circumcenter (unit vector)
        simplex = simplices[j]
        r = radii[j]
        
        # cos_r = v . a = cos(r)
        # dr/d(v.a) = -1/sqrt(1 - (v.a)^2) = -1/sin(r)
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        dr_dcos = -1.0 / sin_r
        
        for idx in simplex:
            p = points[idx]
            # d(v.p)/dp = v (but v depends on p too; for simplicity, treat v as fixed)
            # Actually, we need the full derivative. For a first-order approximation,
            # we use: dr = dr_dcos * d(v.p) = dr_dcos * v . dp
            # So grad contribution: w * dr_dcos * v
            grad[idx] += w * dr_dcos * v
    
    # Project to tangent space: for each point i, remove radial component
    for i in range(len(points)):
        p = points[i]
        grad[i] = grad[i] - np.dot(grad[i], p) * p
    
    return grad

def _repulsive_energy(points, power=2.0):
    """
    Vectorized repulsive energy computation.
    """
    points = _normalize(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    # Upper triangle only
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist_sq[mask]
    d = np.clip(d, 1e-10, None)
    energy = np.sum(1.0 / (d ** (power / 2.0)))
    return energy

def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    
    return pts[:n]

def _lloyd_iteration(points, max_iters=500, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = _normalize(points).copy()
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
                continue
            
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr

def _sa_optimize(points, T_start=0.5, T_end=1e-5, n_iter=10000, time_limit=None, start_time=None):
    """
    Simulated Annealing to minimize repulsive energy.
    """
    points = _normalize(points).copy()
    current_energy = _repulsive_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Cooling schedule
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
                
        T = T_start * (cooling ** i)
        
        # Select a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Generate a new position by adding random noise and normalizing
        step_size = 0.1 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        # Update points
        points[idx] = new_point
        
        new_energy = _repulsive_energy(points)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            points[idx] = original_point
            
    return best_points

def _lbfgs_refine(points, max_time=20.0, start_time=None, beta_init=30.0, beta_final=200.0):
    """
    L-BFGS-B refinement using smooth covering radius surrogate.
    Annotates points as flat array; after each iteration, re-normalize.
    Uses adaptive beta: start smooth, end sharp.
    """
    if start_time is None:
        start_time = time.time()
    
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    n_betas = 5  # number of beta stages
    beta_schedule = np.linspace(beta_init, beta_final, n_betas)
    
    for beta_idx, beta in enumerate(beta_schedule):
        remaining = max_time - (time.time() - start_time)
        if remaining < 2:
            break
        
        # Objective function for L-BFGS-B
        def obj_flat(x_flat):
            x = x_flat.reshape(N, 3)
            x = _normalize(x)
            val = _smooth_covering_radius(x, beta=beta)
            return val
        
        def grad_flat(x_flat):
            x = x_flat.reshape(N, 3)
            x = _normalize(x)
            g = _smooth_covering_radius_grad(x, beta=beta)
            return g.flatten()
        
        x0 = points.flatten()
        
        try:
            result = minimize(
                obj_flat, x0,
                jac=grad_flat,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            x_opt = _normalize(result.x.reshape(N, 3))
            cr = _covering_radius(x_opt)
            if cr < best_cr:
                best_cr = cr
                best_points = x_opt.copy()
                points = x_opt.copy()
        except Exception:
            pass
    
    return best_points, best_cr

def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: brief SA + Lloyd + L-BFGS with smooth surrogate."""
    if start_time is None:
        start_time = time.time()
    
    current_cr = _covering_radius(points)
    
    # Stage 1: Brief SA for exploration (reduced time)
    if time_limit is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, current_cr
        sa_time = min(remaining * 0.15, 6.0)
    else:
        sa_time = 6.0
    
    if sa_time > 1.0:
        points_sa = _sa_optimize(points, T_start=0.3, T_end=1e-5, n_iter=3000, time_limit=sa_time, start_time=start_time)
        cr_sa = _covering_radius(points_sa)
        if cr_sa < current_cr:
            points = points_sa
            current_cr = cr_sa
    
    # Stage 2: Lloyd's algorithm
    if time_limit is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, current_cr
        lloyd_time = min(remaining * 0.25, 10.0)
    else:
        lloyd_time = 10.0
    
    if lloyd_time > 1.0:
        points, cr1 = _lloyd_iteration(points, max_iters=200, time_limit=lloyd_time, start_time=start_time)
        if cr1 < current_cr:
            current_cr = cr1
    
    # Stage 3: L-BFGS with smooth surrogate (most important stage)
    if time_limit is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, current_cr
        lbfgs_time = min(remaining * 0.6, 25.0)
    else:
        lbfgs_time = 25.0
    
    if lbfgs_time > 2.0:
        points, cr2 = _lbfgs_refine(points, max_time=lbfgs_time, start_time=start_time)
        if cr2 < current_cr:
            current_cr = cr2
    
    return points, current_cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    # Phase 1: Refine each initial configuration
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        time_per_config = min(remaining / (len(initial_configs) - i), 30)
        refined, cr = _refine(init_pts, time_limit=time_per_config, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()
    
    # Phase 2: Final high-precision polish with L-BFGS
    remaining = deadline - time.time()
    if remaining > 10:
        # Use a tighter beta schedule for final polish
        final_time = min(remaining * 0.7, 20.0)
        polished, final_cr = _lbfgs_refine(
            best_points, 
            max_time=final_time, 
            start_time=start_time,
            beta_init=50.0,
            beta_final=300.0
        )
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = polished.copy()
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
