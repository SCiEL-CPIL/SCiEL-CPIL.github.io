"""Sphere covering with 50 points – optimized algorithm."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _normalize(points):
    """Normalize points to unit sphere."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fast_facets(points):
    """
    Compute facet data: circumcenters, radii, and vertex indices for each facet.
    Returns (centers, radii, facets) or (None, None, None) on failure.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    centers = []
    radii = []
    facets = []
    
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
        centers.append(v)
        radii.append(r)
        facets.append(simplex)
    
    if not centers:
        return None, None, None
    
    return np.array(centers), np.array(radii), facets

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = _normalize(points)
    centers, radii, facets = _fast_facets(points)
    if centers is None:
        # Fallback: random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    return np.max(radii)

def _smooth_covering_radius_grad(x_flat, beta=50.0):
    """
    Soft-max (log-sum-exp) approximation of the covering radius and its gradient.
    x_flat: flat array of shape (N*3,)
    Returns (objective, gradient_flat).
    """
    x = x_flat.reshape(N, 3)
    x = _normalize(x)
    
    centers, radii, facets = _fast_facets(x)
    if centers is None:
        return 100.0, np.zeros_like(x_flat)
    
    scaled = radii * beta
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / beta
    
    # Softmax weights
    weights = np.exp(scaled - max_scaled) / np.sum(np.exp(scaled - max_scaled))
    
    grad_flat = np.zeros_like(x_flat)
    for i in range(len(centers)):
        center = centers[i]
        weight = weights[i]
        if weight < 1e-10:
            continue
        simplex = facets[i]
        for idx in simplex:
            p = x[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            # Gradient of arccos(dot(c,p)) projected to tangent plane
            grad_p = -center / sin_r
            grad_p_tangent = grad_p - np.dot(grad_p, p) * p
            grad_flat[idx * 3:(idx + 1) * 3] += weight * grad_p_tangent
    
    return soft_max, grad_flat

def _riesz_energy_grad(x_flat, s=2.0):
    """
    Riesz s-energy: sum_{i != j} 1/||x_i - x_j||^s
    and its gradient. This is a smooth auxiliary objective that encourages
    uniform distribution.
    """
    x = x_flat.reshape(N, 3)
    x = _normalize(x)
    
    energy = 0.0
    grad_flat = np.zeros_like(x_flat)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = x[i] - x[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            # d/dx_i [1/dist^s] = -s * d / dist^(s+2)
            energy += 1.0 / (dist ** s)
            grad_flat[i*3:(i+1)*3] += -s * d / (dist ** (s+2))
    
    # Project gradient to tangent plane
    for i in range(N):
        g = grad_flat[i*3:(i+1)*3]
        g_tangent = g - np.dot(g, x[i]) * x[i]
        grad_flat[i*3:(i+1)*3] = g_tangent
    
    return energy, grad_flat

def _lbfgs_smooth(points, time_limit=None, start_time=None, beta=50.0, maxiter=200):
    """L-BFGS-B optimization using smooth log-sum-exp surrogate."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def _obj(x_flat):
        return _smooth_covering_radius_grad(x_flat, beta)[0]
    
    def _grad(x_flat):
        return _smooth_covering_radius_grad(x_flat, beta)[1]
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            _obj, x0,
            jac=_grad,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        if result.success:
            new_points = _normalize(result.x.reshape(N, 3))
            new_cr = _covering_radius(new_points)
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = new_points.copy()
    except Exception:
        pass
    
    return best_points, best_cr

def _lbfgs_riesz(points, time_limit=None, start_time=None, s=2.0, maxiter=100):
    """L-BFGS-B optimization using Riesz energy as auxiliary objective."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def _obj(x_flat):
        return _riesz_energy_grad(x_flat, s)[0]
    
    def _grad(x_flat):
        return _riesz_energy_grad(x_flat, s)[1]
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            _obj, x0,
            jac=_grad,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        if result.success:
            new_points = _normalize(result.x.reshape(N, 3))
            new_cr = _covering_radius(new_points)
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = new_points.copy()
    except Exception:
        pass
    
    return best_points, best_cr

def _spherical_gd(points, max_iters=500, lr_init=0.003, lr_min=1e-8, time_limit=None, start_time=None):
    """Gradient descent to minimize covering radius."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        centers, radii, facets = _fast_facets(points)
        if centers is None:
            break
        
        # Sort facets by radius descending
        order = np.argsort(-radii)
        top_k = min(5, len(order))
        
        step = np.zeros_like(points)
        for i in order[:top_k]:
            weight = max(0.0, radii[i] - best_cr + 0.01)
            if weight < 1e-10:
                continue
            center = centers[i]
            simplex = facets[i]
            for idx in simplex:
                p = points[idx]
                d = center - p
                d_tangent = d - np.dot(d, p) * p
                step[idx] += weight * d_tangent
        
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        step /= step_norm
        
        new_points = _normalize(points + lr * step)
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr - 1e-13:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            lr = max(lr * 0.999, lr_min)
        else:
            lr = max(lr * 0.92, lr_min)
            if lr < lr_min * 1.01:
                break
    
    return best_points, best_cr

def _spherical_lloyd(points, max_iters=300, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        centers, radii, facets = _fast_facets(points)
        if centers is None:
            break
        
        new_points = points.copy()
        point_to_facets = [[] for _ in range(N)]
        for fi, simplex in enumerate(facets):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(N):
            if not point_to_facets[i]:
                continue
            circumcenters = []
            for fi in point_to_facets[i]:
                circumcenters.append(centers[fi])
            if not circumcenters:
                continue
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
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

def _simulated_annealing(points, time_limit=None, start_time=None, seed=42):
    """Simulated annealing to minimize covering radius."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    rng = np.random.RandomState(seed)
    T = 0.05
    cooling = 0.995
    T_min = 1e-6
    
    current_cr = best_cr
    current_points = points.copy()
    
    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = current_points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = current_points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        new_cr = _covering_radius(perturbed)
        delta = new_cr - current_cr
        
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_points = perturbed
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = perturbed.copy()
        
        T *= cooling
    
    return best_points, best_cr

def _multi_beta_lse(points, time_limit=None, start_time=None):
    """
    Multi-stage L-BFGS with increasing beta for progressively sharper
    approximation of the true max.
    """
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    betas = [20.0, 50.0, 100.0, 200.0]
    for beta in betas:
        if time_limit is not None and start_time is not None:
            remaining = time_limit - (time.time() - start_time)
            if remaining < 3:
                break
        
        points, cr = _lbfgs_smooth(points, time_limit=min(remaining*0.8, 10), 
                                     start_time=start_time, beta=beta, maxiter=150)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr

def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(n, seed=42, jitter_scale=0.05):
    """Fibonacci points with random jitter for diversity."""
    pts = _fibonacci_points(n)
    rng = np.random.RandomState(seed)
    for i in range(n):
        noise = rng.randn(3) * jitter_scale
        pts[i] = pts[i] + noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts

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

def _refine(points, time_limit=None, start_time=None):
    """
    Multi-stage refinement pipeline:
    1. Lloyd's algorithm
    2. Riesz energy polish
    3. Multi-beta LSE (smooth covering radius)
    4. Gradient descent polish
    """
    points = _normalize(points)
    
    # Stage 1: Lloyd's algorithm
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, _covering_radius(points)
        lloyd_time = min(remaining * 0.2, 8)
    else:
        lloyd_time = 8
    
    points, cr1 = _spherical_lloyd(points, max_iters=300, time_limit=lloyd_time, start_time=start_time)
    
    # Stage 2: Riesz energy polish
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr1
        riesz_time = min(remaining * 0.15, 6)
    else:
        riesz_time = 6
    
    points, cr2 = _lbfgs_riesz(points, time_limit=riesz_time, start_time=start_time, s=2.0, maxiter=80)
    
    # Stage 3: Multi-beta LSE
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            return points, cr2
        lse_time = min(remaining * 0.35, 15)
    else:
        lse_time = 15
    
    points, cr3 = _multi_beta_lse(points, time_limit=lse_time, start_time=start_time)
    
    # Stage 4: Gradient descent polish
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, cr3
        gd_time = min(remaining * 0.25, 8)
    else:
        gd_time = 8
    
    points, cr4 = _spherical_gd(points, max_iters=400, lr_init=0.002, time_limit=gd_time, start_time=start_time)
    
    return points, cr4

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _jittered_fibonacci_points(N, seed=1, jitter_scale=0.05),
        _jittered_fibonacci_points(N, seed=2, jitter_scale=0.1),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    # Phase 1: Refine each initial configuration
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 12:
            break
        
        time_per_config = min(remaining / (len(initial_configs) - i), 22)
        refined, cr = _refine(init_pts, time_limit=time_per_config, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()
    
    # Phase 2: Simulated annealing from best configuration
    remaining = deadline - time.time()
    if remaining > 15:
        sa_time = min(remaining * 0.25, 20)
        sa_points, sa_cr = _simulated_annealing(best_points, time_limit=sa_time, start_time=start_time, seed=123)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points.copy()
    
    # Phase 3: Final refinement
    remaining = deadline - time.time()
    if remaining > 10:
        final_time = min(remaining * 0.7, 18)
        final_points, final_cr = _refine(best_points, time_limit=final_time, start_time=start_time)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
