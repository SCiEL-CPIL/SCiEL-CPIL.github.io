"""Optimized sphere covering with 50 points using smooth spectral energy + direct polish."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s limit

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

def _spectral_energy_and_grad(pts_flat):
    """
    Smooth surrogate: minimize negative trace of Gram matrix (i.e., maximize sum of pairwise dots).
    This promotes uniform distribution on the sphere.
    Returns (energy, gradient).
    """
    pts = _normalize(pts_flat.reshape(N, 3))
    # Gram matrix G = pts @ pts.T
    G = pts @ pts.T
    # Trace of G = sum of all pairwise dots (including self=1)
    # We want to maximize trace(G) - N (remove self terms) = sum_{i!=j} dot(pi, pj)
    # So minimize -trace(G)
    energy = -np.trace(G)
    
    # Gradient of -trace(G) w.r.t. pts
    # d/dpi trace(G) = d/dpi sum_j (pi·pj) = sum_j pj = pts.sum(axis=0)
    # But we need to account for normalization constraint.
    # Using the projection: for constrained optimization on sphere,
    # the effective gradient is (sum_j pj - (sum_j pj · pi) * pi)
    # Actually, for the Lagrangian with constraint |pi|^2 = 1:
    # grad_i = sum_j pj - lambda_i * pi, where lambda_i = sum_j (pj·pi)
    # For our purpose, we can use: grad_i = sum_j pj (the unconstrained part)
    # and let the optimizer handle the constraint via projection.
    
    # Simpler: use the fact that for points on sphere, the gradient of -sum_{i!=j} pi·pj
    # w.r.t. pi (projected onto tangent space) is:
    # -sum_{j!=i} pj projected onto tangent = -(sum_j pj - pi * (sum_j pj · pi))
    # But since we're using L-BFGS-B with the flat parameterization and normalizing,
    # we can just return the raw gradient and let normalization handle it.
    
    # Gradient of energy = -trace(G) w.r.t. pts_flat
    # d(-trace(G))/dpi = -sum_j pj for each i
    # But this is for the unconstrained case. With normalization, we need to be careful.
    # Let's use: grad = -pts.sum(axis=0, keepdims=True) broadcast to all rows
    # Actually: d(-sum_{i,j} pi·pj)/dpi_i = -sum_j pj
    # This is the same for all i, which is not quite right for the constrained problem.
    
    # Better approach: the gradient of -sum_{i≠j} pi·pj w.r.t. pi on the sphere
    # is: -(sum_{j≠i} pj) projected onto the tangent plane at pi
    # = -(sum_j pj - pi) - (-(sum_j pj - pi)·pi)*pi  [project onto tangent]
    # = -(sum_j pj - pi) + ((sum_j pj - pi)·pi)*pi
    # = -sum_j pj + pi + (sum_j pj·pi - 1)*pi
    # = -sum_j pj + sum_j pj·pi * pi
    # = pi * (sum_j pj·pi) - sum_j pj
    
    # Let's compute this properly
    # sum_j pj is a (3,) vector, same for all i
    sum_pj = pts.sum(axis=0)  # (3,)
    
    # For each i: grad_i = pi * (sum_j pj · pi) - sum_j pj
    # = pi * (pts[i] @ sum_pj) - sum_pj
    dots_with_sum = pts @ sum_pj  # (N,)
    grad = pts * dots_with_sum[:, np.newaxis] - sum_pj[np.newaxis, :]
    
    # This is the gradient on the sphere (tangent space). 
    # For the flat parameterization with normalization, we can return this directly
    # and the optimizer will handle it.
    return energy, grad

def _soft_covering_radius(pts, beta=100.0):
    """
    Soft approximation of max covering radius using log-sum-exp.
    This is smooth and differentiable.
    """
    # For each facet, compute radius
    try:
        hull = ConvexHull(pts)
        radii = []
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
                radii.append(radius)
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
                radii.append(radius)
        
        if not radii:
            return 0.0, None
        
        radii = np.array(radii)
        # Log-sum-exp for soft max
        max_r = np.max(radii)
        if max_r < 1e-10:
            return 0.0, None
        soft_max = max_r + np.log(np.sum(np.exp(beta * (radii - max_r)))) / beta
        return soft_max, radii
    except Exception:
        return 0.0, None

def _spectral_optimize(pts0, time_budget=20.0):
    """Optimize using spectral energy (smooth surrogate)."""
    start_time = time.time()
    pts = _normalize(pts0.copy())
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        return _spectral_energy_and_grad(x)
    
    result = minimize(
        lambda x: _spectral_energy_and_grad(x)[0],
        pts.ravel(),
        jac=lambda x: _spectral_energy_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
    )
    
    if result.fun < 0 or True:  # Always accept if we got a result
        pts = _normalize(result.x.reshape(N, 3))
    
    return pts

def _direct_polish(pts0, time_budget=15.0):
    """
    Direct polish: minimize covering radius using projected gradient descent.
    At each step, find the worst facet and move its vertices to reduce the radius.
    """
    start_time = time.time()
    pts = _normalize(pts0.copy())
    current_radius = _covering_radius(pts)
    
    lr = 0.01
    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break
        if current_radius < 1e-6:
            break
        
        # Find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_facet = None
        worst_radius = 0.0
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
                if radius > worst_radius:
                    worst_radius = radius
                    worst_facet = (simplex, n, d)
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
                    n = -n
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                if radius > worst_radius:
                    worst_radius = radius
                    worst_facet = (simplex, n, d)
        
        if worst_facet is None:
            break
        
        simplex, n, d = worst_facet
        # To reduce the radius arccos(d), we need to increase d
        # d = n·a where n is the outward normal
        # Moving vertices in the direction of n (projected onto sphere) increases d
        
        # For each vertex in the facet, move it slightly in the direction of n
        # projected onto the tangent plane at that vertex
        new_pts = pts.copy()
        for idx in simplex:
            p = pts[idx]
            # Tangent direction: n - (n·p)*p, normalized
            n_tangent = n - np.dot(n, p) * p
            n_tangent_norm = np.linalg.norm(n_tangent)
            if n_tangent_norm > 1e-12:
                n_tangent = n_tangent / n_tangent_norm
                # Move in this direction on the sphere
                angle = lr
                # New point: rotate p by angle in direction n_tangent
                # Using Rodrigues' rotation or simple spherical interpolation
                new_p = p * np.cos(angle) + n_tangent * np.sin(angle)
                new_pts[idx] = new_p / np.linalg.norm(new_p)
        
        new_pts = _normalize(new_pts)
        new_radius = _covering_radius(new_pts)
        
        if new_radius < current_radius:
            pts = new_pts
            current_radius = new_radius
            # Decay learning rate slowly
            lr *= 0.995
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return pts, current_radius

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

def _two_phase_optimize(pts0, time_budget=35.0):
    """
    Two-phase optimization:
    1. Spectral energy minimization (smooth, fast)
    2. Direct covering radius polish (targeted)
    """
    start_time = time.time()
    pts = _normalize(pts0.copy())
    
    # Phase 1: Spectral energy
    phase1_time = min(20.0, time_budget * 0.6)
    if time.time() - start_time < time_budget - 5:
        pts = _spectral_optimize(pts, time_budget=phase1_time)
    
    # Phase 2: Direct polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        pts, _ = _direct_polish(pts, time_budget=min(15.0, remaining))
    
    return pts

def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    # Multiple starting points
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=0),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=42),
    ]
    
    for i, pts0 in enumerate(starts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Evaluate starting point
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        
        # Two-phase optimization
        time_budget = min(30.0, DEADLINE - (time.time() - start_time) - 10)
        if time_budget > 15:
            pts_opt = _two_phase_optimize(pts0, time_budget=time_budget)
            r_opt = _covering_radius(pts_opt)
            if r_opt < best_radius:
                best_radius = r_opt
                best_pts = pts_opt.copy()
    
    # Final polish from best
    remaining = DEADLINE - (time.time() - start_time) - 3
    if remaining > 5:
        pts_final, _ = _direct_polish(best_pts, time_budget=min(10.0, remaining))
        r_final = _covering_radius(pts_final)
        if r_final < best_radius:
            best_radius = r_final
            best_pts = pts_final.copy()
    
    # Ensure validity
    best_pts = _normalize(best_pts)
    return best_pts
