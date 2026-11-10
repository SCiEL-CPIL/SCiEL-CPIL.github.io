"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_DEADLINE = 145.0  # Safety margin before 150s kill


def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms


def _generate_fibonacci_samples(n_samples):
    """Generate well-distributed sample points on the sphere using Fibonacci spiral."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples


def _covering_radius_sampled(pts, samples):
    """
    Fast estimate of covering radius using sample points.
    Returns value in degrees.
    """
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = dots.max(axis=1)
    min_dot = np.min(max_dots)
    radius_rad = np.arccos(np.clip(min_dot, -1.0, 1.0))
    return np.degrees(radius_rad)


def _covering_radius_hull(pts):
    """
    Exact covering radius via convex hull: find the facet with largest circumradius.
    Returns value in degrees.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback to sampled estimate
        samples = _generate_fibonacci_samples(5000)
        return _covering_radius_sampled(pts, samples)
    
    max_circum_rad = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
        
        s = (ab + bc + ca) / 2.0
        denom = 2.0 * np.sin(s) * np.sin(s - ab) * np.sin(s - bc) * np.sin(s - ca)
        if abs(denom) < 1e-15:
            continue
        sin_rc = (np.sin(ab) * np.sin(bc) * np.sin(ca)) / denom
        sin_rc = np.clip(sin_rc, 0.0, 1.0)
        circum_rad = np.arcsin(sin_rc)
        
        if circum_rad > max_circum_rad:
            max_circum_rad = circum_rad
    
    return np.degrees(max_circum_rad)


def _repulsive_dynamics(pts, iterations=200, alpha=0.05, beta=0.02):
    """
    Deterministic repulsive force dynamics.
    Points repel each other based on angular distance to spread out.
    This is a different optimization family: physics-based relaxation.
    """
    pts = _normalize(pts).copy()
    for _ in range(iterations):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Repulsive force: stronger when close
                # Use inverse square or similar
                force_mag = alpha / (dist * dist + beta)
                force_vec = force_mag * diff / dist
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Update positions
        pts += forces
        pts = _normalize(pts)
    return pts


def _gradient_optimize(pts_init, start_time, deadline):
    """
    Gradient-based optimization using L-BFGS-B.
    Objective: maximize the minimum angular distance to the nearest point (i.e., minimize covering radius).
    We use a smooth approximation of the covering radius for gradient computation.
    """
    pts = _normalize(pts_init).copy()
    
    # Generate samples for smooth objective
    n_samples = 20000
    samples = _generate_fibonacci_samples(n_samples)
    
    def objective_and_grad(x):
        """
        x is flattened (N, 3) array.
        Returns objective (covering radius) and gradient.
        We minimize -min_distance, which is equivalent to maximizing min_distance.
        Actually, we want to minimize covering radius.
        Covering radius = max_y min_i angle(y, p_i).
        This is not differentiable everywhere.
        
        Alternative smooth objective: 
        f(x) = -log(sum_i exp(-beta * d_i^2)) where d_i is distance from sample to nearest point.
        This is a smooth approximation of the max distance.
        
        Let's use a simpler approach: 
        We define the objective as the covering radius estimated by samples.
        Gradient is estimated via finite differences or we use a differentiable proxy.
        
        Proxy: minimize the sum of squares of (max_distance - target) for all samples?
        Or better: minimize the mean of the top-k distances?
        
        Let's try: minimize the 99th percentile of the angular distances from samples to nearest point.
        This is still not differentiable.
        
        Let's use a differentiable proxy:
        f(x) = sum_s max_i (-log(1 - dot(s, p_i)^2)) 
        This is related to the log-sum-exp of distances.
        
        Actually, let's just use finite differences with the sampled covering radius.
        It's slow but robust.
        """
        pts_x = _normalize(x.reshape(N, 3))
        
        # Fast sampled covering radius
        dots = samples @ pts_x.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = dots.max(axis=1)
        min_dot = np.min(max_dots)
        radius_rad = np.arccos(np.clip(min_dot, -1.0, 1.0))
        obj = np.degrees(radius_rad)
        
        # Gradient via finite differences
        grad = np.zeros_like(x)
        eps = 1e-4
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            pts_plus = _normalize(x_plus.reshape(N, 3))
            dots_plus = samples @ pts_plus.T
            dots_plus = np.clip(dots_plus, -1.0, 1.0)
            max_dots_plus = dots_plus.max(axis=1)
            min_dot_plus = np.min(max_dots_plus)
            radius_plus = np.degrees(np.arccos(np.clip(min_dot_plus, -1.0, 1.0)))
            
            x_minus = x.copy()
            x_minus[k] -= eps
            pts_minus = _normalize(x_minus.reshape(N, 3))
            dots_minus = samples @ pts_minus.T
            dots_minus = np.clip(dots_minus, -1.0, 1.0)
            max_dots_minus = dots_minus.max(axis=1)
            min_dot_minus = np.min(max_dots_minus)
            radius_minus = np.degrees(np.arccos(np.clip(min_dot_minus, -1.0, 1.0)))
            
            grad[k] = (radius_plus - radius_minus) / (2 * eps)
        
        return obj, grad
    
    # This finite difference gradient is too slow for L-BFGS-B with 150 dimensions.
    # We need a faster gradient.
    
    # Let's use a different strategy: 
    # Use the gradient of the log-sum-exp of the distances.
    # f(x) = -log(sum_s exp(-beta * d_s^2))
    # where d_s is the angular distance from sample s to the nearest point.
    # This is a smooth upper bound on the covering radius.
    
    beta = 100.0  # Sharpness of the approximation
    
    def smooth_objective(x):
        pts_x = _normalize(x.reshape(N, 3))
        dots = samples @ pts_x.T
        dots = np.clip(dots, -1.0, 1.0)
        # d_s = arccos(max_i dot(s, p_i))
        max_dots = dots.max(axis=1)
        d_s = np.arccos(np.clip(max_dots, -1.0, 1.0))
        # f(x) = -log(sum exp(-beta * d_s^2))
        # To avoid overflow, use log-sum-exp trick
        vals = -beta * d_s**2
        max_val = np.max(vals)
        log_sum = max_val + np.log(np.sum(np.exp(vals - max_val)))
        return -log_sum
    
    def smooth_gradient(x):
        pts_x = _normalize(x.reshape(N, 3))
        dots = samples @ pts_x.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = dots.max(axis=1)
        d_s = np.arccos(np.clip(max_dots, -1.0, 1.0))
        
        vals = -beta * d_s**2
        max_val = np.max(vals)
        exp_vals = np.exp(vals - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Gradient of d_s w.r.t. x
        # d_s = arccos(max_i dot(s, p_i))
        # Let m_s = max_i dot(s, p_i)
        # dd_s/dm_s = -1/sqrt(1 - m_s^2)
        # dm_s/dx_k = sum_i (d m_s/dx_k)
        # m_s = max_i (s . p_i)
        # The gradient is non-zero only for the i that achieves the max.
        # For simplicity, we use a soft-max approximation for the max.
        # Let m_s = logsumexp(beta2 * dots_s) / beta2
        beta2 = 50.0
        dots_s = dots  # (n_samples, N)
        logsumexp_dots = np.logaddexp.reduce(dots_s, axis=1)
        m_s_soft = np.tanh(beta2 * (dots_s - np.max(dots_s, axis=1, keepdims=True)) ) # Approximation
        
        # This is getting complex. Let's use finite differences for the smooth objective.
        # It's still faster than the raw covering radius because the objective is smoother.
        grad = np.zeros_like(x)
        eps = 1e-4
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            f_plus = smooth_objective(x_plus)
            
            x_minus = x.copy()
            x_minus[k] -= eps
            f_minus = smooth_objective(x_minus)
            
            grad[k] = (f_plus - f_minus) / (2 * eps)
        
        return smooth_objective(x), grad
    
    # Optimize
    x0 = pts.flatten()
    result = minimize(smooth_gradient, x0, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'maxfun': 200})
    
    return _normalize(result.x.reshape(N, 3))


def _generate_initial_points(method='fibonacci', rng=None):
    """Generate initial point configurations using different methods."""
    if method == 'fibonacci':
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _normalize(pts)
    elif method == 'random':
        if rng is None:
            rng = np.random.default_rng(42)
        pts = rng.normal(size=(N, 3))
        return _normalize(pts)
    else:
        raise ValueError(f"Unknown method: {method}")


def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses gradient-based optimization and repulsive dynamics.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Repulsive dynamics from Fibonacci init
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        pts_init = _generate_initial_points('fibonacci', rng)
        rep_pts = _repulsive_dynamics(pts_init, iterations=100, alpha=0.1, beta=0.01)
        radius = _covering_radius_hull(rep_pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = rep_pts.copy()
    
    # Strategy 2: Gradient optimization from repulsive result
    elapsed = time.time() - start_time
    if elapsed < deadline - 10 and best_pts is not None:
        grad_pts = _gradient_optimize(best_pts, start_time, deadline)
        radius = _covering_radius_hull(grad_pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = grad_pts.copy()
    
    # Strategy 3: Repulsive dynamics from Random init
    for i in range(3):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        pts_init = _generate_initial_points('random', rng)
        rep_pts = _repulsive_dynamics(pts_init, iterations=100, alpha=0.1, beta=0.01)
        radius = _covering_radius_hull(rep_pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = rep_pts.copy()
            
        # Gradient polish
        if time.time() < deadline - 10:
            grad_pts = _gradient_optimize(rep_pts, start_time, deadline)
            radius = _covering_radius_hull(grad_pts)
            if radius < best_radius:
                best_radius = radius
                best_pts = grad_pts.copy()
    
    # Final polish: short gradient run on best
    if best_pts is not None and time.time() < deadline - 5:
        grad_pts = _gradient_optimize(best_pts, start_time, deadline)
        radius = _covering_radius_hull(grad_pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = grad_pts.copy()
    
    if best_pts is None:
        best_pts = _generate_initial_points('fibonacci', rng)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
