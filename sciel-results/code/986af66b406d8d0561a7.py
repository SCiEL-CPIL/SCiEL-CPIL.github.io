"""Direct geometric approach: optimize 130 points on unit sphere to minimize covering radius."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _facet_circumcenters(points):
    """Compute circumcenter directions and cos values for all hull facets.
    Returns (cos_vals, normals, valid_mask) arrays of length n_facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    facets = hull.simplices
    n_facets = len(facets)
    cos_vals = np.zeros(n_facets)
    normals = np.zeros((n_facets, 3))
    valid = np.zeros(n_facets, dtype=bool)
    for i, facet in enumerate(facets):
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        if p @ a < 0:
            p = -p
        cos_vals[i] = p @ a
        normals[i] = p
        valid[i] = True
    return cos_vals, normals, valid, facets

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi
    result = _facet_circumcenters(points)
    if result is None:
        return np.pi
    cos_vals, normals, valid, facets = result
    if not np.any(valid):
        return np.pi
    min_cos = np.min(cos_vals[valid])
    return np.arccos(np.clip(min_cos, -1.0, 1.0))

def _soft_objective_and_grad(points, beta=50.0):
    """
    Compute smooth surrogate objective and its gradient w.r.t. points.
    Objective: soft_max of angular distances = soft_min of cos_vals.
    Uses analytic derivative of circumcenter w.r.t. facet vertices.
    """
    points = np.asarray(points, dtype=np.float64)
    result = _facet_circumcenters(points)
    if result is None:
        return np.pi, np.zeros_like(points)
    cos_vals, normals, valid, facets = result
    n_valid = np.sum(valid)
    if n_valid == 0:
        return np.pi, np.zeros_like(points)
    
    # Work with -cos_vals: we want to minimize max angular distance = minimize max(-cos)
    # soft_max(-cos) = (1/beta)*log(sum(exp(-beta*cos_i)))
    neg_cos = -cos_vals[valid]
    neg_cos_clipped = np.clip(neg_cos, -50.0, 50.0)
    max_neg = np.max(neg_cos_clipped)
    exp_vals = np.exp(beta * (neg_cos_clipped - max_neg))
    sum_exp = np.sum(exp_vals)
    soft_val = (1.0 / beta) * (max_neg + np.log(sum_exp))
    
    # Weight for each valid facet: w_i = exp(-beta*cos_i)/sum
    weights = np.zeros(len(facets))
    weights[valid] = exp_vals / sum_exp
    
    # Gradient of soft_val w.r.t. cos_vals: d(soft)/d(cos_i) = -w_i (for valid i)
    # So d(soft)/d(point_j) = sum_i (-w_i) * d(cos_i)/d(point_j)
    
    grad_points = np.zeros_like(points)
    I3 = np.eye(3)
    
    for i in range(len(facets)):
        if not valid[i]:
            continue
        w_i = weights[i]
        if w_i < 1e-15:
            continue
        
        facet = facets[i]
        ia, ib, ic = facet[0], facet[1], facet[2]
        a = points[ia]
        b = points[ib]
        c = points[ic]
        
        u = a - b
        v = b - c
        n_vec = np.cross(u, v)
        n_norm = np.linalg.norm(n_vec)
        if n_norm < 1e-12:
            continue
        
        p_orig = n_vec / n_norm
        sign = 1.0 if (p_orig @ a) > 0 else -1.0
        p = sign * p_orig
        # Projection matrix: P = I - p p^T
        P = I3 - np.outer(p, p)
        
        # Analytic gradients of cos_val = p · a w.r.t. a, b, c
        # Derived from: dp = P dn / |n|, where n = cross(a-b, b-c)
        # d(cos)/da = [P (v × a) / |n| + p] * sign
        # d(cos)/db = [P (-(v × a) + (a × u)) / |n|] * sign
        # d(cos)/dc = [P (-(a × u)) / |n|] * sign
        va_cross_a = np.cross(v, a)
        a_cross_u = np.cross(a, u)
        
        grad_a = sign * (P @ va_cross_a / n_norm + p)
        grad_b = sign * (P @ (-va_cross_a + a_cross_u) / n_norm)
        grad_c = sign * (P @ (-a_cross_u) / n_norm)
        
        # d(soft)/d(point) = -w_i * d(cos_i)/d(point)
        grad_points[ia] -= w_i * grad_a
        grad_points[ib] -= w_i * grad_b
        grad_points[ic] -= w_i * grad_c
    
    return soft_val, grad_points

def _lbfgs_smooth_optimize(points, deadline, beta_start=20.0, beta_end=80.0, max_iters=300):
    """Use L-BFGS on the smooth surrogate to refine the configuration."""
    points = points.copy()
    beta = beta_start
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        val, _ = _soft_objective_and_grad(pts, beta=beta)
        return val
    
    def gradient(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        _, grad = _soft_objective_and_grad(pts, beta=beta)
        # Project gradient to tangent space (remove radial component)
        grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return grad.flatten()
    
    try:
        result = minimize(
            objective,
            points.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < objective(points.flatten()):
            new_points = result.x.reshape(N, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_points = new_points / norms
            return new_points
    except Exception:
        pass
    
    return points

def _weighted_worst_facets(points, k=20, beta=50.0):
    """
    Return weighted worst facets using soft-max weighting.
    Returns list of (indices, circumcenter, weight) for top k facets.
    """
    points = np.asarray(points, dtype=np.float64)
    result = _facet_circumcenters(points)
    if result is None:
        return []
    cos_vals, normals, valid, facets = result
    
    if not np.any(valid):
        return []
    
    neg_cos = -cos_vals[valid]
    neg_cos_clipped = np.clip(neg_cos, -50.0, 50.0)
    max_neg = np.max(neg_cos_clipped)
    exp_vals = np.exp(beta * (neg_cos_clipped - max_neg))
    sum_exp = np.sum(exp_vals)
    weights = np.zeros(len(facets))
    weights[valid] = exp_vals / sum_exp
    
    valid_indices = np.where(valid)[0]
    sorted_idx = valid_indices[np.argsort(weights[valid_indices])[::-1]]
    
    result_list = []
    for idx in sorted_idx[:k]:
        if weights[idx] < 1e-8:
            break
        result_list.append((facets[idx].copy(), normals[idx].copy(), weights[idx]))
    
    return result_list


def _worst_facets(points, k=5):
    """Return the k worst facets (by circumradius) as list of (indices, circumcenter)."""
    points = np.asarray(points, dtype=np.float64)
    result = _facet_circumcenters(points)
    if result is None:
        return []
    cos_vals, normals, valid, facets = result
    
    if not np.any(valid):
        return []
    
    valid_indices = np.where(valid)[0]
    sorted_idx = valid_indices[np.argsort(cos_vals[valid_indices])]
    
    return [(facets[idx].copy(), normals[idx].copy()) for idx in sorted_idx[:k]]


def _gradient_descent(points, deadline, max_iters=3000, base_step=0.05):
    """Optimize points using gradient descent toward circumcenters of worst facets with soft weighting."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    beta = 30.0
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.05, 0.1)
        else:
            step *= 0.85
        
        if step < 1e-10:
            break
        
        progress = it / max_iters
        beta = 30.0 + 70.0 * progress
        
        worst = _weighted_worst_facets(points, k=15, beta=beta)
        if not worst:
            break
        
        displacement = np.zeros_like(points)
        for facet, circ, weight in worst:
            for idx in facet:
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    tangent = direction - np.dot(direction, pt) * pt
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                        displacement[idx] += weight * tangent
        
        for idx in range(N):
            if np.linalg.norm(displacement[idx]) > 1e-10:
                new_pt = points[idx] + step * displacement[idx]
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def _random_restart_optimize(rng, deadline, max_iters=2000):
    """Generate random initial config and optimize."""
    points = rng.standard_normal((N, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return _gradient_descent(points, deadline, max_iters=max_iters)

def _polish(points, deadline, max_iters=500, base_step=0.005):
    """Final high-precision polish with very small steps."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.02, 0.01)
        else:
            step *= 0.9
        
        if step < 1e-12:
            break
        
        worst = _weighted_worst_facets(points, k=3, beta=100.0)
        if not worst:
            break
        
        displacement = np.zeros_like(points)
        for facet, circ, weight in worst:
            for idx in facet:
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    tangent = direction - np.dot(direction, pt) * pt
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                        displacement[idx] += weight * tangent
        
        for idx in range(N):
            if np.linalg.norm(displacement[idx]) > 1e-10:
                new_pt = points[idx] + step * displacement[idx]
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Start with Fibonacci spiral as baseline
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = fib_points / np.linalg.norm(fib_points, axis=1, keepdims=True)
    
    best_points = _gradient_descent(fib_points, deadline, max_iters=2000, base_step=0.04)
    best_radius = _covering_radius(best_points)
    
    # Multiple random restarts
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        if best_points is not None and restart % 2 == 0:
            perturbation = 0.15 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        max_iters = min(2000, int(remaining * 20))
        optimized = _gradient_descent(points, deadline, max_iters=max_iters, base_step=0.06)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # L-BFGS smooth optimization on the best configuration
    if best_points is not None and time.time() < deadline - 15.0:
        lbfgs_points = _lbfgs_smooth_optimize(best_points, deadline, beta_start=30.0, beta_end=80.0, max_iters=200)
        lbfgs_radius = _covering_radius(lbfgs_points)
        if lbfgs_radius < best_radius:
            best_radius = lbfgs_radius
            best_points = lbfgs_points
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, deadline, max_iters=800, base_step=0.003)
        best_radius = _covering_radius(best_points)
    
    # Second L-BFGS pass after polish
    if best_points is not None and time.time() < deadline - 3.0:
        lbfgs_points = _lbfgs_smooth_optimize(best_points, deadline, beta_start=50.0, beta_end=100.0, max_iters=150)
        lbfgs_radius = _covering_radius(lbfgs_points)
        if lbfgs_radius < best_radius:
            best_radius = lbfgs_radius
            best_points = lbfgs_points
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
