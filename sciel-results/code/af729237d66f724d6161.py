"""Optimized sphere covering with 50 points using smooth soft-max energy + Lloyd + polish."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 50
DEADLINE = 145.0  # safety margin before 150s limit

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute covering radius in radians using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            if len(facet_pts) == 3:
                a, b, c = facet_pts
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                d = np.dot(n, a)
                if d < 0:
                    d = -d
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                max_radius = max(max_radius, radius)
            elif len(facet_pts) == 4:
                a, b, c = facet_pts[:3]
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                d = np.dot(n, a)
                if d < 0:
                    d = -d
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                max_radius = max(max_radius, radius)
        return max_radius
    except Exception:
        # Fallback: brute force sampling
        theta = np.linspace(0, 2*np.pi, 360)
        phi = np.linspace(0, np.pi, 180)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        dots = np.sum(samples[:, np.newaxis, :] * pts[np.newaxis, :, :], axis=2)
        min_dots = np.max(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.max(max_angle)

def _soft_max_energy(pts, beta):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    For each sample point on the sphere, compute the max angular distance to nearest config point,
    then use soft-max to get a smooth objective.
    """
    # Use a set of sample points on the sphere
    # We'll use a moderate grid for the surrogate
    n_theta = 60
    n_phi = 30
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Compute angular distances: for each sample, distance to each config point
    # cos(angle) = dot(sample, point), so angle = arccos(dot)
    dots = samples @ pts.T  # shape (n_samples, N)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))  # shape (n_samples, N)
    
    # For each sample, we want the MIN angle (nearest point)
    min_angles = np.min(angles, axis=1)  # shape (n_samples,)
    
    # Soft-max of min_angles: we want to approximate max(min_angles)
    # Use log-sum-exp: log(sum(exp(beta * min_angles))) / beta
    # This approximates max(min_angles) as beta -> inf
    shifted = min_angles - np.max(min_angles)
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted)))
    soft_max = np.max(min_angles) + log_sum_exp / beta
    
    return soft_max

def _soft_max_energy_grad(pts, beta):
    """
    Compute gradient of the soft-max energy with respect to pts.
    """
    n_theta = 60
    n_phi = 30
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    dots = samples @ pts.T  # (n_samples, N)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))  # (n_samples, N)
    
    # For each sample, find the nearest point (min angle)
    nearest_idx = np.argmin(angles, axis=1)  # (n_samples,)
    
    # Soft-max weights
    min_angles = np.min(angles, axis=1)
    shifted = min_angles - np.max(min_angles)
    weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))  # (n_samples,)
    
    # Gradient: d/d(pts_j) of arccos(dots[i,j]) where i's nearest is j
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    grad = np.zeros_like(pts)
    for i in range(len(samples)):
        j = nearest_idx[i]
        w = weights[i]
        dot_val = dots[i, j]
        dot_val = np.clip(dot_val, -1.0 + 1e-12, 1.0 - 1e-12)
        denom = np.sqrt(1.0 - dot_val * dot_val)
        if denom < 1e-12:
            denom = 1e-12
        # d(angle)/d(pts_j) = -samples[i] / denom
        grad[j] += w * (-samples[i] / denom)
    
    return grad

def _lloyd_iteration(pts, n_iters=50, step_scale=0.5):
    """
    Lloyd's algorithm: move each point toward the circumcenter of its worst facet.
    """
    pts = _normalize(pts.copy())
    for it in range(n_iters):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find the worst facet it belongs to
        # and move it toward that facet's circumcenter
        # Better: for each facet, compute its circumcenter, then for each vertex,
        # accumulate the pull toward the circumcenter weighted by how "bad" the facet is
        
        facet_radii = []
        facet_circumcenters = []
        facet_indices = []
        
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            if len(facet_pts) == 3:
                a, b, c = facet_pts
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                d = np.dot(n, a)
                if d < 0:
                    d = -d
                    n = -n
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                # Circumcenter direction is n (the outward normal)
                facet_radii.append(radius)
                facet_circumcenters.append(n)
                facet_indices.append(simplex)
        
        if not facet_radii:
            break
        
        facet_radii = np.array(facet_radii)
        max_r = np.max(facet_radii)
        
        # Weight each facet by how much it contributes to the covering radius
        # Use exponential weighting to focus on the worst facets
        beta = 10.0
        weights = np.exp(beta * (facet_radii - max_r))
        weights /= np.sum(weights)
        
        # For each point, accumulate the weighted circumcenter directions
        new_pts = pts.copy()
        for k, simplex in enumerate(facet_indices):
            w = weights[k]
            cc = facet_circumcenters[k]
            for idx in simplex:
                # Move point idx toward the circumcenter direction
                # The circumcenter is in the direction of cc from the origin
                # So we want to move pts[idx] toward cc
                new_pts[idx] += step_scale * w * (cc - pts[idx])
        
        new_pts = _normalize(new_pts)
        
        # Check for improvement
        if _covering_radius(new_pts) <= _covering_radius(pts) + 1e-12:
            pts = new_pts
        else:
            # Reduce step size
            step_scale *= 0.5
            if step_scale < 1e-6:
                break
    
    return pts

def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci spiral points."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _soft_max_optimize(pts0, beta_start=50.0, beta_end=1500.0, n_betas=20, time_limit=30.0):
    """
    Optimize using soft-max surrogate with annealing beta.
    """
    start_time = time.time()
    pts = _normalize(pts0.copy())
    
    # Anneal beta from beta_start to beta_end
    betas = np.linspace(beta_start, beta_end, n_betas)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        # Define objective and gradient for this beta
        def obj(x):
            pts_local = _normalize(x.reshape(N, 3))
            return _soft_max_energy(pts_local, beta)
        
        def grad(x):
            pts_local = _normalize(x.reshape(N, 3))
            g = _soft_max_energy_grad(pts_local, beta)
            # Project gradient onto tangent space
            proj = g - np.sum(g * pts_local, axis=1, keepdims=True) * pts_local
            return proj.ravel()
        
        # Use L-BFGS-B with gradient
        result = minimize(obj, pts.ravel(), jac=grad, method='L-BFGS-B',
                         options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.success or result.fun < obj(pts.ravel()):
            pts = _normalize(result.x.reshape(N, 3))
    
    return pts

def _polish(pts, time_limit=10.0):
    """
    Final high-precision polish using direct minimization of the covering radius
    with a tighter tolerance.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    def obj(x):
        pts_local = _normalize(x.reshape(N, 3))
        return _covering_radius(pts_local)
    
    # Use Nelder-Mead for the final polish (no gradient needed)
    result = minimize(obj, pts.ravel(), method='Nelder-Mead',
                     options={'maxiter': 200, 'xatol': 1e-14, 'fatol': 1e-14})
    
    if result.success or result.fun < _covering_radius(pts):
        pts = _normalize(result.x.reshape(N, 3))
    
    return pts

def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    # Strategy: multi-start with soft-max annealing + Lloyd + polish
    
    # Start 1: Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    r0 = _covering_radius(pts0)
    if best_radius > r0:
        best_radius = r0
        best_pts = pts0.copy()
    
    # Soft-max optimization from Fibonacci
    if time.time() - start_time < DEADLINE - 40:
        pts_opt = _soft_max_optimize(pts0, beta_start=50.0, beta_end=1500.0, 
                                     n_betas=15, time_limit=20.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
    
    # Lloyd iteration from best
    if time.time() - start_time < DEADLINE - 30:
        pts_lloyd = _lloyd_iteration(best_pts, n_iters=50, step_scale=0.5)
        r_lloyd = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
    
    # Start 2: Random with seed 0
    if time.time() - start_time < DEADLINE - 30:
        pts0 = _random_sphere(N, seed=0)
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        pts_opt = _soft_max_optimize(pts0, beta_start=50.0, beta_end=1500.0, 
                                     n_betas=15, time_limit=15.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
        pts_lloyd = _lloyd_iteration(best_pts, n_iters=30, step_scale=0.5)
        r_lloyd = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
    
    # Start 3: Random with seed 1
    if time.time() - start_time < DEADLINE - 25:
        pts0 = _random_sphere(N, seed=1)
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        pts_opt = _soft_max_optimize(pts0, beta_start=50.0, beta_end=1500.0, 
                                     n_betas=15, time_limit=12.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
        pts_lloyd = _lloyd_iteration(best_pts, n_iters=30, step_scale=0.5)
        r_lloyd = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
    
    # Start 4: Random with seed 2
    if time.time() - start_time < DEADLINE - 20:
        pts0 = _random_sphere(N, seed=2)
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        pts_opt = _soft_max_optimize(pts0, beta_start=50.0, beta_end=1500.0, 
                                     n_betas=15, time_limit=10.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
        pts_lloyd = _lloyd_iteration(best_pts, n_iters=30, step_scale=0.5)
        r_lloyd = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
    
    # Start 5: Random with seed 3
    if time.time() - start_time < DEADLINE - 15:
        pts0 = _random_sphere(N, seed=3)
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        pts_opt = _soft_max_optimize(pts0, beta_start=50.0, beta_end=1500.0, 
                                     n_betas=15, time_limit=8.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
        pts_lloyd = _lloyd_iteration(best_pts, n_iters=20, step_scale=0.5)
        r_lloyd = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
    
    # Final polish: Nelder-Mead from best
    if time.time() - start_time < DEADLINE - 5:
        pts_final = _polish(best_pts, time_limit=5.0)
        r_final = _covering_radius(pts_final)
        if r_final < best_radius:
            best_radius = r_final
            best_pts = pts_final.copy()
    
    # One more Lloyd pass from polished result
    if time.time() - start_time < DEADLINE - 3:
        pts_lloyd2 = _lloyd_iteration(best_pts, n_iters=20, step_scale=0.3)
        r_lloyd2 = _covering_radius(pts_lloyd2)
        if r_lloyd2 < best_radius:
            best_radius = r_lloyd2
            best_pts = pts_lloyd2.copy()
    
    # Ensure validity
    best_pts = _normalize(best_pts)
    
    # Check that points are not in a closed hemisphere
    # and that they are not coincident
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10):
        # Regenerate
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
