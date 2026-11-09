"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius_and_worst_facets(points, top_k=1):
    """
    Compute the covering radius in radians and identify the worst facets.
    Returns (radius, list of (facet_indices, circumcenter) for top_k worst facets).
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, []
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, []
    
    facets = hull.simplices  # (M, 3) array of indices
    
    # Compute circumcenter and cos_r for each facet
    facet_data = []
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        # Compute circumcenter direction
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_r = np.clip(p @ a, -1.0, 1.0)
        facet_data.append((cos_r, facet.copy(), p.copy()))
    
    if not facet_data:
        return np.pi, []
    
    # Sort by cos_r (ascending = largest radius first)
    facet_data.sort(key=lambda x: x[0])
    
    min_cos = facet_data[0][0]
    radius = np.arccos(np.clip(min_cos, -1.0, 1.0))
    
    # Return top_k worst facets
    top_facets = [(f[1], f[2]) for f in facet_data[:top_k]]
    
    return radius, top_facets


def _soft_covering_radius(points, beta=30.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    Returns (soft_radius, gradient) where gradient is w.r.t. points.
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, np.zeros_like(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    facets = hull.simplices
    m = len(facets)
    if m == 0:
        return np.pi, np.zeros_like(points)
    
    # Compute cos_r for each facet and store facet info
    cos_vals = np.zeros(m)
    valid = np.ones(m, dtype=bool)
    facet_pts = np.zeros((m, 3, 3))  # (m, 3, 3) - three points per facet
    
    for i, facet in enumerate(facets):
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        facet_pts[i] = [a, b, c]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            valid[i] = False
            continue
        p = p / p_norm
        if p @ a < 0:
            p = -p
        cos_vals[i] = np.clip(p @ a, -1.0, 1.0)
    
    if not np.any(valid):
        return np.pi, np.zeros_like(points)
    
    valid_idx = np.where(valid)[0]
    valid_cos = cos_vals[valid_idx]
    
    if len(valid_cos) == 0:
        return np.pi, np.zeros_like(points)
    
    # Soft min of cos_vals: we want to maximize the minimum cos (minimize max angle)
    # soft_min_cos = -logsumexp(-beta * cos_vals) / beta
    scaled = -beta * valid_cos
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_cos = -log_sum / beta
    soft_min_cos = np.clip(soft_min_cos, -1.0, 1.0)
    soft_radius = np.arccos(soft_min_cos)
    
    # Gradient computation
    # d(soft_radius)/d(cos_i) for i in valid:
    # soft_min_cos = -logsumexp(-beta*cos)/beta
    # d(soft_min_cos)/d(cos_i) = exp(-beta*cos_i) / sum_j exp(-beta*cos_j) = w_i
    # d(soft_radius)/d(cos_i) = -1/sqrt(1-cos^2) * w_i  (since d(arccos)/dx = -1/sqrt(1-x^2))
    
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # weights for each valid facet
    
    # d(soft_min_cos)/d(cos_i) = weights[i]
    # d(soft_radius)/d(cos_i) = -weights[i] / sqrt(1 - soft_min_cos^2)
    
    denom = np.sqrt(max(1e-15, 1.0 - soft_min_cos * soft_min_cos))
    d_radius_d_cos = -weights / denom  # shape (len(valid_idx),)
    
    # Now we need d(cos_i)/d(points) for each facet
    # cos_i = p_i @ a_i where p_i is the circumcenter direction
    # This is complex; approximate by treating the circumcenter as fixed for gradient
    # A simpler approach: use finite differences or just move toward circumcenters
    
    # For now, return zero gradient and use geometric repair
    grad = np.zeros_like(points)
    
    return soft_radius, grad


def _local_optimize(points, max_iters=5000, deadline=None):
    """
    Combined local optimization using geometric repair with adaptive step size.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facets(points)[0]
    no_improve = 0
    step = 0.08
    min_step = 1e-8
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=5)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-12:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive step size: reduce more aggressively when stuck
        if no_improve > 50:
            step *= 0.7
            no_improve = 0
        elif no_improve > 20:
            step *= 0.85
        elif no_improve < 10 and step < 0.1:
            step = min(step * 1.1, 0.15)
        
        if step < min_step:
            break
        
        # Move vertices of top worst facets toward their circumcenters
        moved_indices = set()
        for facet, circ in top_facets:
            for idx in facet:
                if idx in moved_indices:
                    continue
                moved_indices.add(idx)
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    new_pt = pt + step * direction
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    points[idx] = new_pt
    
    return points


def _polish(points, max_iters=3000, deadline=None):
    """
    High-precision polish pass with very small steps to fine-tune the configuration.
    """
    points = points.copy()
    best_radius = _covering_radius_and_worst_facets(points)[0]
    step = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=1)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-14:
            best_radius = radius
            # Reduce step slightly on improvement
            step *= 0.97
        else:
            step *= 0.5
        
        if step < 1e-12:
            break
        
        facet, circ = top_facets[0]
        for idx in facet:
            pt = points[idx]
            direction = circ - pt
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                new_pt = pt + step * direction
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def _smooth_optimize(points, deadline=None, max_iters=500):
    """
    Smooth optimization using log-sum-exp surrogate with Riemannian gradient-like updates.
    Moves points in the direction that reduces the soft covering radius.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facets(points)[0]
    beta = 20.0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute soft radius and identify facets contributing most
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        facets = hull.simplices
        m = len(facets)
        if m == 0:
            break
        
        cos_vals = np.zeros(m)
        valid = np.ones(m, dtype=bool)
        circumcenters = np.zeros((m, 3))
        
        for i, facet in enumerate(facets):
            a = points[facet[0]]
            b = points[facet[1]]
            c = points[facet[2]]
            
            v1 = a - b
            v2 = b - c
            p = np.cross(v1, v2)
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-12:
                valid[i] = False
                continue
            p = p / p_norm
            if p @ a < 0:
                p = -p
            cos_vals[i] = np.clip(p @ a, -1.0, 1.0)
            circumcenters[i] = p
        
        if not np.any(valid):
            break
        
        valid_idx = np.where(valid)[0]
        valid_cos = cos_vals[valid_idx]
        
        # Compute weights (probability each facet is the "worst" under soft-max)
        scaled = -beta * valid_cos
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        weights = exp_scaled / np.sum(exp_scaled)
        
        # Move each point in the weighted average direction of its contributing facets
        # For each point, accumulate the weighted circumcenter directions
        point_gradients = np.zeros((n, 3))
        
        for j, vi in enumerate(valid_idx):
            w = weights[j]
            if w < 1e-8:
                continue
            facet = facets[vi]
            circ = circumcenters[vi]
            for idx in facet:
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    point_gradients[idx] += w * direction
        
        # Normalize and apply step
        step = 0.05
        for idx in range(n):
            grad = point_gradients[idx]
            norm_grad = np.linalg.norm(grad)
            if norm_grad > 1e-10:
                grad = grad / norm_grad
                new_pt = points[idx] + step * grad
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
        
        # Check improvement
        radius, _ = _covering_radius_and_worst_facets(points)
        if radius < best_radius - 1e-12:
            best_radius = radius
    
    return points


def _lloyd_iteration(points, deadline=None):
    """
    One iteration of Lloyd's algorithm for spherical codes.
    """
    points = points.copy()
    n = points.shape[0]
    
    if deadline is not None and time.time() > deadline - 5.0:
        return points
    
    radius, top_facets = _covering_radius_and_worst_facets(points, top_k=10)
    
    if not top_facets:
        return points
    
    # Move vertices of worst facets
    for facet, circ in top_facets:
        for idx in facet:
            pt = points[idx]
            direction = circ - pt
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                # Small step
                new_pt = pt + 0.05 * direction
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with random initializations
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Random points
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Perturbed Fibonacci
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            points = points + 0.1 * rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            # Another perturbed Fibonacci with different seed
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            points = points + 0.15 * rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Perturbed best or random
            if best_points is not None and rng.random() < 0.6:
                perturbation = 0.2 * rng.standard_normal((N, 3))
                points = best_points + perturbation
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = rng.standard_normal((N, 3))
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Local optimization with geometric repair
        remaining_time = deadline - time.time() - 10.0
        if remaining_time < 10.0:
            break
        
        max_iters = min(4000, int(remaining_time * 40))
        optimized = _local_optimize(points, max_iters=max_iters, deadline=deadline)
        
        # Smooth optimization pass
        if time.time() < deadline - 20.0:
            smooth_iters = min(200, int((deadline - time.time() - 15.0) * 5))
            optimized = _smooth_optimize(optimized, deadline=deadline, max_iters=smooth_iters)
        
        # Additional Lloyd-like iterations
        if time.time() < deadline - 12.0:
            lloyd_iters = min(50, int((deadline - time.time() - 10.0) * 5))
            for _ in range(lloyd_iters):
                if time.time() > deadline - 10.0:
                    break
                optimized = _lloyd_iteration(optimized, deadline=deadline)
        
        # Polish pass if we have time
        if time.time() < deadline - 8.0:
            optimized = _polish(optimized, max_iters=2000, deadline=deadline)
        
        radius, _ = _covering_radius_and_worst_facets(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish on the best result if we have time
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, max_iters=1500, deadline=deadline)
    
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_points = best_points / norms
    
    if best_points is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points
