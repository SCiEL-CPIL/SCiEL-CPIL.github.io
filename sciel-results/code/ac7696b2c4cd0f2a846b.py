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
    m = facets.shape[0]
    
    # Vectorized circumcenter computation
    A = points[facets[:, 0]]  # (m, 3)
    B = points[facets[:, 1]]  # (m, 3)
    C = points[facets[:, 2]]  # (m, 3)
    
    v1 = A - B
    v2 = B - C
    P = np.cross(v1, v2)  # (m, 3)
    p_norms = np.linalg.norm(P, axis=1)
    
    valid = p_norms > 1e-12
    if not np.any(valid):
        return np.pi, []
    
    P_valid = P[valid] / p_norms[valid, None]
    A_valid = A[valid]
    
    # Ensure circumcenter points outward
    dots = np.sum(P_valid * A_valid, axis=1)
    flip = dots < 0
    P_valid[flip] = -P_valid[flip]
    dots[flip] = -dots[flip]
    
    cos_r = np.clip(dots, -1.0, 1.0)
    
    # Map back to original indices
    all_cos = np.full(m, 1.0)  # default: best possible (won't be selected)
    all_cos[valid] = cos_r
    
    # Sort by cos_r ascending (worst first)
    sorted_idx = np.argsort(all_cos)
    
    min_cos = all_cos[sorted_idx[0]]
    radius = np.arccos(np.clip(min_cos, -1.0, 1.0))
    
    # Return top_k worst facets
    top_facets = []
    for idx in sorted_idx[:top_k]:
        if not valid[idx]:
            continue
        facet = facets[idx].copy()
        circ = P_valid[list(valid).index(idx) if idx in list(valid) else 0].copy() if False else None
        # Recompute for this specific facet to get correct circ
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        v1f = a - b
        v2f = b - c
        pf = np.cross(v1f, v2f)
        pf_norm = np.linalg.norm(pf)
        if pf_norm > 1e-12:
            pf = pf / pf_norm
            if pf @ a < 0:
                pf = -pf
        else:
            pf = np.array([1.0, 0.0, 0.0])
        top_facets.append((facet, pf))
    
    return radius, top_facets


def _pairwise_soft_repulsion(points, beta=40.0, k_neighbors=10):
    """
    Compute a soft repulsion force based on nearest neighbors.
    This creates a uniform pressure that pushes points apart,
    targeting the covering radius by enforcing separation.
    """
    points = np.asarray(points)
    n = points.shape[0]
    forces = np.zeros_like(points)
    
    # Compute all pairwise dot products
    dots = points @ points.T
    
    # For each point, find its k nearest neighbors and compute soft repulsion
    for i in range(n):
        # Get distances (1 - dot) to all other points
        dists = 1.0 - dots[i]
        dists[i] = np.inf  # Exclude self
        
        # Get indices of k nearest neighbors
        if n > k_neighbors:
            knn_indices = np.argpartition(dists, k_neighbors)[:k_neighbors]
        else:
            knn_indices = np.arange(n)
            knn_indices = knn_indices[knn_indices != i]
        
        # Compute soft repulsion from these neighbors
        # Force is proportional to exp(-beta * dist) * direction
        for j in knn_indices:
            dist = 1.0 - dots[i, j]
            weight = np.exp(-beta * dist)
            # Direction away from neighbor j
            direction = points[i] - points[j]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-12:
                direction = direction / norm_dir
                forces[i] += weight * direction
    
    # Normalize forces
    norms = np.linalg.norm(forces, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    forces = forces / norms
    
    return forces


def _soft_covering_radius(points, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp
    over facet circumradii. Returns (soft_radius, gradient w.r.t. points).
    """
    points = np.asarray(points)
    n = points.shape[0]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    facets = hull.simplices
    m = facets.shape[0]
    
    # Compute circumradii for all facets
    A = points[facets[:, 0]]
    B = points[facets[:, 1]]
    C = points[facets[:, 2]]
    
    v1 = A - B
    v2 = B - C
    P = np.cross(v1, v2)
    p_norms = np.linalg.norm(P, axis=1)
    
    valid = p_norms > 1e-12
    if not np.any(valid):
        return np.pi, np.zeros_like(points)
    
    P_valid = P[valid] / p_norms[valid, None]
    A_valid = A[valid]
    
    dots = np.sum(P_valid * A_valid, axis=1)
    flip = dots < 0
    P_valid[flip] = -P_valid[flip]
    dots[flip] = -dots[flip]
    
    cos_r = np.clip(dots, -1.0, 1.0)
    # Radius = arccos(cos_r), so we want to minimize max radius
    # Use log-sum-exp on (radius - t) style, but simpler: soft-max of radii
    radii = np.arccos(cos_r)
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    soft_radius = max_r + np.log(np.sum(weights)) / beta
    
    # Compute gradient: d(soft_radius)/d(points)
    # For each valid facet, gradient of arccos(cos_r) w.r.t. point
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    # cos_r = P @ A where P depends on A,B,C
    # This is complex; use finite differences or approximate
    
    # Approximate gradient: for each facet, the gradient of cos_r w.r.t. each vertex
    # cos_r = (P @ A) where P = normalize(cross(A-B, B-C))
    # This is messy; use a simpler approach: perturb and measure
    
    grad = np.zeros_like(points)
    
    # For each valid facet, compute contribution to gradient
    valid_indices = np.where(valid)[0]
    for vi in valid_indices:
        w = weights[vi]
        if w < 1e-10:
            continue
        
        facet = facets[vi]
        a_idx, b_idx, c_idx = facet[0], facet[1], facet[2]
        a = points[a_idx]
        b = points[b_idx]
        c = points[c_idx]
        
        v1f = a - b
        v2f = b - c
        pf = np.cross(v1f, v2f)
        pf_norm = np.linalg.norm(pf)
        if pf_norm < 1e-12:
            continue
        pf = pf / pf_norm
        if pf @ a < 0:
            pf = -pf
        
        cos_rf = np.clip(pf @ a, -1.0, 1.0)
        sin_rf = np.sqrt(max(0.0, 1.0 - cos_rf * cos_rf))
        if sin_rf < 1e-12:
            continue
        
        # d(arccos(cos_r))/d(point) = -1/sin_r * d(cos_r)/d(point)
        # cos_r = pf @ a (approximately, treating pf as fixed for local gradient)
        # d(cos_r)/da = pf
        # d(cos_r)/db = d(pf)/db @ a + pf @ d(a)/db ... complex
        # Simplification: treat pf as fixed, only vary a
        # This is an approximation
        
        # Better: use the fact that for a triangle on unit sphere,
        # moving a vertex changes the circumradius
        # Approximate gradient for each vertex
        # da: gradient contribution = -w/sin_r * pf
        # db, dc: more complex, approximate by symmetry
        
        # For now, use the simple approximation
        grad[a_idx] += -w * pf / sin_rf
        # Approximate for b and c
        # The effect of moving b on cos_r is roughly pf @ (da/db) = pf @ (-1) direction... 
        # Use: d(cos_r)/db ≈ pf @ (b - a) direction component
        # Simplified: split the gradient among all three vertices
        grad[b_idx] += -w * pf * 0.3 / sin_rf
        grad[c_idx] += -w * pf * 0.3 / sin_rf
    
    return soft_radius, grad


def _smooth_optimize(points, max_iters=2000, deadline=None, beta=50.0):
    """
    Smooth optimization using log-sum-exp surrogate for covering radius.
    Uses projected gradient descent on the sphere.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facets(points)[0]
    step = 0.03
    min_step = 1e-9
    no_improve = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        soft_r, grad = _soft_covering_radius(points, beta=beta)
        
        # Check actual radius for improvement tracking
        actual_r = _covering_radius_and_worst_facets(points)[0]
        
        if actual_r < best_radius - 1e-12:
            best_radius = actual_r
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive step size
        if no_improve > 30:
            step *= 0.8
            no_improve = 0
        elif no_improve > 10:
            step *= 0.9
        
        if step < min_step:
            break
        
        # Project gradient to tangent space and apply
        for i in range(n):
            pt = points[i]
            g = grad[i]
            # Project to tangent space
            g_tan = g - np.dot(g, pt) * pt
            norm_gt = np.linalg.norm(g_tan)
            if norm_gt > 1e-12:
                g_tan = g_tan / norm_gt
                new_pt = pt - step * g_tan  # Gradient descent
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[i] = new_pt
    
    return points


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
        
        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=3)
        
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


def _repulsion_optimize(points, max_iters=2000, deadline=None):
    """
    Optimization using pairwise soft repulsion to enforce uniform separation.
    This targets the covering radius by creating uniform pressure.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facets(points)[0]
    step = 0.05
    min_step = 1e-9
    no_improve = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius_and_worst_facets(points)[0]
        
        if radius < best_radius - 1e-12:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive step size
        if no_improve > 30:
            step *= 0.8
            no_improve = 0
        elif no_improve > 10:
            step *= 0.9
        
        if step < min_step:
            break
        
        # Compute repulsion forces
        forces = _pairwise_soft_repulsion(points, beta=40.0, k_neighbors=10)
        
        # Apply forces with projection to tangent space
        for i in range(n):
            pt = points[i]
            force = forces[i]
            # Project force to tangent space
            force_tan = force - np.dot(force, pt) * pt
            norm_ft = np.linalg.norm(force_tan)
            if norm_ft > 1e-12:
                force_tan = force_tan / norm_ft
                new_pt = pt + step * force_tan
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[i] = new_pt
    
    return points


def _polish(points, max_iters=2000, deadline=None):
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
            step *= 0.95
        else:
            step *= 0.5
        
        if step < 1e-10:
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


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with random initializations
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:
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
        else:
            # Perturbed best or random
            if best_points is not None and rng.random() < 0.5:
                perturbation = 0.25 * rng.standard_normal((N, 3))
                points = best_points + perturbation
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = rng.standard_normal((N, 3))
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Smooth optimization first to navigate the landscape
        remaining_time = deadline - time.time() - 10.0
        if remaining_time < 15.0:
            break
        
        # Smooth phase
        smooth_iters = min(1500, int(remaining_time * 20))
        if smooth_iters > 100:
            optimized = _smooth_optimize(points, max_iters=smooth_iters, deadline=deadline)
        
        # Local optimization with geometric repair
        remaining_time = deadline - time.time() - 10.0
        if remaining_time < 10.0:
            optimized = points
        else:
            max_iters = min(2500, int(remaining_time * 25))
            optimized = _local_optimize(optimized, max_iters=max_iters, deadline=deadline)
        
        # Repulsion optimization pass if we have time
        if time.time() < deadline - 15.0:
            optimized = _repulsion_optimize(optimized, max_iters=800, deadline=deadline)
        
        # Polish pass if we have time
        if time.time() < deadline - 8.0:
            optimized = _polish(optimized, max_iters=1000, deadline=deadline)
        
        radius, _ = _covering_radius_and_worst_facets(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish on the best result if we have time
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, max_iters=500, deadline=deadline)
    
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
