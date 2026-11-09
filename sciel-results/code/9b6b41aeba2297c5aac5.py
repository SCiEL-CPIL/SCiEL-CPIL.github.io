"""Optimize 130 points on unit sphere to minimize covering radius via convex hull facet analysis."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _facet_circumcenters(points):
    """Compute circumcenter directions and cos values for all hull facets."""
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
        if p_norm < 1e-14:
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
    """
    points = np.asarray(points, dtype=np.float64)
    result = _facet_circumcenters(points)
    if result is None:
        return np.pi, np.zeros_like(points)
    cos_vals, normals, valid, facets = result
    n_valid = np.sum(valid)
    if n_valid == 0:
        return np.pi, np.zeros_like(points)
    
    neg_cos = -cos_vals[valid]
    neg_cos_clipped = np.clip(neg_cos, -50.0, 50.0)
    max_neg = np.max(neg_cos_clipped)
    exp_vals = np.exp(beta * (neg_cos_clipped - max_neg))
    sum_exp = np.sum(exp_vals)
    soft_val = (1.0 / beta) * (max_neg + np.log(sum_exp))
    
    weights = np.zeros(len(facets))
    weights[valid] = exp_vals / sum_exp
    
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
        if n_norm < 1e-14:
            continue
        
        p_orig = n_vec / n_norm
        sign = 1.0 if (p_orig @ a) > 0 else -1.0
        p = sign * p_orig
        P = I3 - np.outer(p, p)
        
        va_cross_a = np.cross(v, a)
        a_cross_u = np.cross(a, u)
        
        grad_a = sign * (P @ va_cross_a / n_norm + p)
        grad_b = sign * (P @ (-va_cross_a + a_cross_u) / n_norm)
        grad_c = sign * (P @ (-a_cross_u) / n_norm)
        
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
    """Return weighted worst facets using soft-max weighting."""
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
    """Return the k worst facets (by circumradius)."""
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

def _lloyd_iterate(points, deadline, max_iters=2000):
    """
    Discrete Lloyd's algorithm on spherical Voronoi:
    For each point, move it toward the centroid of its Voronoi cell (the facet circumcenters).
    This equalizes the facet circumradii.
    """
    points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
        
        result = _facet_circumcenters(points)
        if result is None:
            break
        cos_vals, normals, valid, facets = result
        
        if not np.any(valid):
            break
        
        # For each point, collect all facets it belongs to
        # and compute the weighted centroid of their circumcenters
        new_points = points.copy()
        
        for idx in range(N):
            # Find all facets containing this point
            facet_indices = []
            for fi in range(len(facets)):
                if not valid[fi]:
                    continue
                if idx in facets[fi]:
                    facet_indices.append(fi)
            
            if not facet_indices:
                continue
            
            # Weight by how "bad" each facet is (lower cos = worse)
            weights = []
            centers = []
            for fi in facet_indices:
                # Inverse of angular distance: worse facets get more weight
                w = 1.0 / (1.0 + (1.0 - cos_vals[fi]) * 20.0)
                weights.append(w)
                centers.append(normals[fi])
            
            weights = np.array(weights)
            centers = np.array(centers)
            weights = weights / np.sum(weights)
            
            centroid = np.sum(weights[:, None] * centers, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-10:
                continue
            centroid = centroid / centroid_norm
            
            # Move point toward centroid along tangent
            direction = centroid - points[idx]
            tangent = direction - np.dot(direction, points[idx]) * points[idx]
            t_norm = np.linalg.norm(tangent)
            if t_norm > 1e-10:
                tangent = tangent / t_norm
                # Adaptive step: larger when centroid is far
                step_size = min(0.1, 0.02 * t_norm)
                new_pt = points[idx] + step_size * tangent
                new_pt = new_pt / np.linalg.norm(new_pt)
                new_points[idx] = new_pt
        
        points = new_points
    
    return points

def _mixed_optimize(points, deadline, max_iters=2000):
    """
    Alternate between gradient descent on worst facets and Lloyd iterations.
    """
    points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Every 5 iterations, do a Lloyd step
        if it % 5 == 0:
            # Lloyd iteration
            result = _facet_circumcenters(points)
            if result is None:
                break
            cos_vals, normals, valid, facets = result
            
            if not np.any(valid):
                break
            
            new_points = points.copy()
            for idx in range(N):
                facet_indices = []
                for fi in range(len(facets)):
                    if not valid[fi]:
                        continue
                    if idx in facets[fi]:
                        facet_indices.append(fi)
                
                if not facet_indices:
                    continue
                
                weights = []
                centers = []
                for fi in facet_indices:
                    w = 1.0 / (1.0 + (1.0 - cos_vals[fi]) * 20.0)
                    weights.append(w)
                    centers.append(normals[fi])
                
                weights = np.array(weights)
                centers = np.array(centers)
                weights = weights / np.sum(weights)
                
                centroid = np.sum(weights[:, None] * centers, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm < 1e-10:
                    continue
                centroid = centroid / centroid_norm
                
                direction = centroid - points[idx]
                tangent = direction - np.dot(direction, points[idx]) * points[idx]
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-10:
                    tangent = tangent / t_norm
                    step_size = min(0.08, 0.015 * t_norm)
                    new_pt = points[idx] + step_size * tangent
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    new_points[idx] = new_pt
            
            points = new_points
        
        # Gradient descent step
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
        
        worst = _weighted_worst_facets(points, k=10, beta=40.0)
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
        
        step = 0.03
        for idx in range(N):
            if np.linalg.norm(displacement[idx]) > 1e-10:
                new_pt = points[idx] + step * displacement[idx]
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Start with Fibonacci spiral as baseline
    fib_points = _fibonacci_spiral()
    
    # Phase 1: Gradient descent from Fibonacci
    best_points = _gradient_descent(fib_points, deadline, max_iters=1500, base_step=0.04)
    best_radius = _covering_radius(best_points)
    
    # Phase 2: Lloyd's algorithm refinement
    if time.time() < deadline - 40.0:
        lloyd_pts = _lloyd_iterate(best_points, deadline, max_iters=1000)
        lloyd_radius = _covering_radius(lloyd_pts)
        if lloyd_radius < best_radius:
            best_radius = lloyd_radius
            best_points = lloyd_pts
    
    # Phase 3: Mixed optimization (Lloyd + gradient)
    if time.time() < deadline - 30.0:
        mixed_pts = _mixed_optimize(best_points, deadline, max_iters=1500)
        mixed_radius = _covering_radius(mixed_pts)
        if mixed_radius < best_radius:
            best_radius = mixed_radius
            best_points = mixed_pts
    
    # Phase 4: Random restarts with Lloyd + gradient
    num_restarts = 4
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        if restart % 2 == 0:
            # Perturb best
            perturbation = 0.12 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Random
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Quick gradient descent
        max_iters = min(1000, int(remaining * 15))
        optimized = _gradient_descent(points, deadline, max_iters=max_iters, base_step=0.05)
        
        # Lloyd refinement
        if time.time() < deadline - 10.0:
            optimized = _lloyd_iterate(optimized, deadline, max_iters=500)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Phase 5: L-BFGS smooth optimization
    if best_points is not None and time.time() < deadline - 15.0:
        lbfgs_points = _lbfgs_smooth_optimize(best_points, deadline, beta_start=30.0, beta_end=80.0, max_iters=200)
        lbfgs_radius = _covering_radius(lbfgs_points)
        if lbfgs_radius < best_radius:
            best_radius = lbfgs_radius
            best_points = lbfgs_points
    
    # Phase 6: Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        best_points = _polish(best_points, deadline, max_iters=600, base_step=0.003)
        best_radius = _covering_radius(best_points)
    
    # Phase 7: Second L-BFGS pass
    if best_points is not None and time.time() < deadline - 3.0:
        lbfgs_points = _lbfgs_smooth_optimize(best_points, deadline, beta_start=50.0, beta_end=100.0, max_iters=150)
        lbfgs_radius = _covering_radius(lbfgs_points)
        if lbfgs_radius < best_radius:
            best_radius = lbfgs_radius
            best_points = lbfgs_points
    
    # Final polish after L-BFGS
    if best_points is not None and time.time() < deadline - 2.0:
        best_points = _polish(best_points, deadline, max_iters=300, base_step=0.002)
        best_radius = _covering_radius(best_points)
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
