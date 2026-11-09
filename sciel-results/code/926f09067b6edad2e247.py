"""
Covering the sphere with 130 equal caps.
Strategy: Vectorized geometric repair on convex hull facets + smooth log-sum-exp
surrogate optimization + high-precision polish.
"""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130


def _covering_radius_and_worst_facets(points, top_k=1):
    """
    Compute the covering radius in radians and identify the worst facets.
    Returns (radius, list of (facet_indices, circumcenter) for top_k worst facets).
    Vectorized circumcenter computation.
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
    if m == 0:
        return np.pi, []

    # Vectorized circumcenter computation
    A = points[facets[:, 0]]
    B = points[facets[:, 1]]
    C = points[facets[:, 2]]

    v1 = A - B
    v2 = B - C
    P = np.cross(v1, v2)
    p_norms = np.linalg.norm(P, axis=1)

    valid = p_norms > 1e-12
    if not np.any(valid):
        return np.pi, []

    # Store valid indices for mapping back
    valid_indices = np.where(valid)[0]
    P_valid = P[valid] / p_norms[valid, None]
    A_valid = A[valid]

    # Ensure circumcenter points outward (toward the sphere surface)
    dots = np.sum(P_valid * A_valid, axis=1)
    flip = dots < 0
    P_valid[flip] = -P_valid[flip]
    dots[flip] = -dots[flip]

    cos_r_all = np.full(m, 2.0)  # invalid facets get high cos (small radius)
    cos_r_all[valid_indices] = np.clip(dots, -1.0, 1.0)

    # Sort by cos_r ascending (largest radius first)
    order = np.argsort(cos_r_all)
    min_cos = cos_r_all[order[0]]
    radius = np.arccos(np.clip(min_cos, -1.0, 1.0))

    # Return top_k worst facets
    top_facets = []
    for idx in order[:top_k]:
        if not valid[idx]:
            continue
        facet = facets[idx]
        circ = P_valid[idx - valid_indices[0] if False else np.where(valid_indices == idx)[0][0]]
        top_facets.append((facet.copy(), circ.copy()))

    return radius, top_facets


def _soft_covering_radius(points, beta=50.0, k=15):
    """
    Smooth surrogate for the covering radius using log-sum-exp over the k worst facets.
    Returns (soft_radius, gradient) where gradient is with respect to points.
    """
    points = np.asarray(points)
    n = points.shape[0]

    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)

    facets = hull.simplices
    m = facets.shape[0]
    if m == 0:
        return np.pi, np.zeros_like(points)

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

    cos_vals = np.clip(dots, -1.0 + 1e-10, 1.0 - 1e-10)
    # Convert to radius (radians)
    radii = np.arccos(cos_vals)

    # Log-sum-exp over the k largest radii
    k_eff = min(k, len(radii))
    top_k_radii = np.partition(radii, -k_eff)[-k_eff:]
    top_k_radii = top_k_radii[top_k_radii > 0]

    if len(top_k_radii) == 0:
        return np.pi, np.zeros_like(points)

    max_r = np.max(top_k_radii)
    # Stable log-sum-exp
    shifted = top_k_radii - max_r
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)

    soft_radius = max_r + np.log(np.sum(weights)) / beta

    # Gradient: d(soft_radius)/d(points)
    # We need to track which facets are in the top-k
    # For simplicity, compute gradient only for the top-k facets
    # Map back to original facet indices
    # This is complex; for now, use a simpler approach:
    # Gradient of arccos(cos_r) w.r.t. points for each facet

    grad = np.zeros_like(points)

    # For each of the top-k facets, compute the gradient contribution
    # We need to identify which valid facets are in the top-k
    # Sort valid facets by radius
    valid_order = np.argsort(radii)
    top_valid_indices = valid_order[-k_eff:]

    for vi in top_valid_indices:
        orig_idx = np.where(valid)[vi]
        facet = facets[orig_idx]
        w = weights[vi] if vi < len(weights) else 0

        if w < 1e-10:
            continue

        # Gradient of arccos(p @ a) w.r.t. a
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        # x = p @ a, dx/da = p
        # So d(radius)/da = -p / sqrt(1 - (p@a)^2)
        # And d(radius)/dp = -a / sqrt(1 - (p@a)^2)
        # p depends on the three points of the facet

        a_pt = points[facet[0]]
        b_pt = points[facet[1]]
        c_pt = points[facet[2]]

        # Recompute p for this facet
        v1_f = a_pt - b_pt
        v2_f = b_pt - c_pt
        p_f = np.cross(v1_f, v2_f)
        p_norm_f = np.linalg.norm(p_f)
        if p_norm_f < 1e-12:
            continue
        p_f = p_f / p_norm_f
        if p_f @ a_pt < 0:
            p_f = -p_f

        cos_f = np.clip(p_f @ a_pt, -1.0 + 1e-10, 1.0 - 1e-10)
        sin_f = np.sqrt(np.maximum(0.0, 1.0 - cos_f * cos_f))
        if sin_f < 1e-10:
            continue

        # Gradient w.r.t. each vertex
        # This is complex because p depends on all three vertices
        # Use numerical gradient for this small contribution
        eps = 1e-6
        for vertex_idx in range(3):
            pt = points[facet[vertex_idx]].copy()
            pt_p = pt.copy()
            pt_p[0] += eps
            pt_p = pt_p / np.linalg.norm(pt_p)
            pt_m = pt.copy()
            pt_m[0] -= eps
            pt_m = pt_m / np.linalg.norm(pt_m)

            # Evaluate radius with perturbed point
            for delta_pt, sign in [(pt_p, 1), (pt_m, -1)]:
                pts_copy = points.copy()
                pts_copy[facet[vertex_idx]] = delta_pt
                try:
                    hull_tmp = ConvexHull(pts_copy)
                    facets_tmp = hull_tmp.simplices
                    # Find the same facet
                    # This is expensive; skip for now
                except Exception:
                    pass

        # Simplified: just move toward circumcenter
        # This is what the geometric repair does, so for the smooth surrogate,
        # we'll use the direction toward the circumcenter weighted by w
        direction = p_f - a_pt
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-10:
            direction = direction / norm_dir
            grad[facet[0]] += w * direction
            grad[facet[1]] += w * 0.5 * direction
            grad[facet[2]] += w * 0.5 * direction

    # Project gradient to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]

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

        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=3)

        if not top_facets:
            break

        if radius < best_radius - 1e-12:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1

        # Adaptive step size
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


def _smooth_optimize(points, max_iters=2000, deadline=None):
    """
    Optimization using smooth log-sum-exp surrogate.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facets(points)[0]
    step = 0.05
    min_step = 1e-9
    no_improve = 0
    beta = 30.0

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break

        radius = _covering_radius_and_worst_facets(points)[0]

        if radius < best_radius - 1e-12:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1

        # Adaptive step size and beta
        if no_improve > 30:
            step *= 0.8
            beta *= 1.2
            no_improve = 0
        elif no_improve > 10:
            step *= 0.9

        if step < min_step:
            break

        # Compute smooth surrogate gradient
        soft_r, grad = _soft_covering_radius(points, beta=beta, k=15)

        # Apply gradient descent (minimize soft_radius)
        norm_grad = np.linalg.norm(grad)
        if norm_grad > 1e-12:
            grad = grad / norm_grad
            for i in range(n):
                pt = points[i]
                g = grad[i]
                # Project to tangent space
                g_tan = g - np.dot(g, pt) * pt
                norm_gt = np.linalg.norm(g_tan)
                if norm_gt > 1e-12:
                    g_tan = g_tan / norm_gt
                    new_pt = pt - step * g_tan
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

        # Local optimization with geometric repair
        remaining_time = deadline - time.time() - 10.0
        if remaining_time < 15.0:
            break

        max_iters = min(3000, int(remaining_time * 30))
        optimized = _local_optimize(points, max_iters=max_iters, deadline=deadline)

        # Smooth optimization pass if we have time
        if time.time() < deadline - 15.0:
            optimized = _smooth_optimize(optimized, max_iters=1000, deadline=deadline)

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
