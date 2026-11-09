"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius_and_worst_facet(points):
    """
    Compute the covering radius in radians and identify the worst facet.
    Returns (radius, worst_facet_indices, circumcenter).
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, None, None
    
    hull = ConvexHull(points)
    facets = hull.simplices  # (M, 3) array of indices
    
    min_cos = 2.0
    worst_facet = None
    worst_circ = None
    
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
        
        cos_r = p @ a
        cos_r = np.clip(cos_r, -1.0, 1.0)
        if cos_r < min_cos:
            min_cos = cos_r
            worst_facet = facet.copy()
            worst_circ = p.copy()
    
    if min_cos >= 2.0:
        # Fallback
        min_cos = -1.0
    
    radius = np.arccos(np.clip(min_cos, -1.0, 1.0))
    return radius, worst_facet, worst_circ


def _geometric_repair(points, max_iters=2000):
    """
    Direct geometric optimization: iteratively find the worst facet
    and move its vertices toward the circumcenter.
    """
    points = points.copy()
    n = points.shape[0]
    
    for it in range(max_iters):
        radius, worst_facet, worst_circ = _covering_radius_and_worst_facet(points)
        
        if worst_facet is None:
            break
        
        # Get the three vertices of the worst facet
        i, j, k = worst_facet
        a = points[i].copy()
        b = points[j].copy()
        c = points[k].copy()
        
        # Move each vertex a small step toward the circumcenter
        # Use a small step size to avoid oscillation
        step = 0.05
        
        for idx, pt in [(i, a), (j, b), (k, c)]:
            # Spherical linear interpolation toward circumcenter
            # Simple approach: move in direction of circumcenter
            direction = worst_circ - pt
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                # Move by step in that direction, then renormalize
                new_pt = pt + step * direction
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def _local_optimize(points, max_iters=5000):
    """
    Combined local optimization using geometric repair with adaptive step size.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facet(points)[0]
    no_improve = 0
    step = 0.1
    
    for it in range(max_iters):
        radius, worst_facet, worst_circ = _covering_radius_and_worst_facet(points)
        
        if worst_facet is None:
            break
        
        if radius < best_radius:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 200:
            # Reduce step size
            step *= 0.5
            no_improve = 0
            if step < 1e-6:
                break
        
        i, j, k = worst_facet
        
        # Move vertices toward circumcenter
        for idx in [i, j, k]:
            pt = points[idx]
            direction = worst_circ - pt
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                new_pt = pt + step * direction
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with random initializations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
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
                perturbation = 0.3 * rng.standard_normal((N, 3))
                points = best_points + perturbation
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = rng.standard_normal((N, 3))
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Local optimization with geometric repair
        optimized = _local_optimize(points, max_iters=3000)
        
        radius, _, _ = _covering_radius_and_worst_facet(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
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
