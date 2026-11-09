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


def _local_optimize(points, max_iters=5000, deadline=None):
    """
    Combined local optimization using geometric repair with adaptive step size.
    Iteratively finds the worst facet and moves its vertices toward the circumcenter.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facet(points)[0]
    no_improve = 0
    step = 0.1
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        radius, worst_facet, worst_circ = _covering_radius_and_worst_facet(points)
        
        if worst_facet is None:
            break
        
        if radius < best_radius - 1e-10:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 150:
            # Reduce step size
            step *= 0.5
            no_improve = 0
            if step < 1e-7:
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
    # Safety margin: 150s limit, leave 10s buffer
    deadline = t_start + 140.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with diverse initializations
    # We use a mix of Fibonacci spiral, random, and perturbed best
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
            # Another random set
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Perturbed best or random
            if best_points is not None and rng.random() < 0.6:
                # Perturb the best solution found so far
                # Use a moderate perturbation to escape local minima
                perturbation = 0.2 * rng.standard_normal((N, 3))
                points = best_points + perturbation
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            else:
                # Random points
                points = rng.standard_normal((N, 3))
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Local optimization with geometric repair
        # Allocate remaining time minus buffer for this restart
        remaining = deadline - time.time()
        # Leave some time for final normalization and safety
        opt_deadline = t_start + 140.0 - (num_restarts - restart) * 1.0 
        # Actually, let's just use the main deadline minus a small buffer for the next restart
        # Or better, just run until the main deadline is close
        # The _local_optimize function checks the deadline
        optimized = _local_optimize(points, max_iters=5000, deadline=deadline - 5.0)
        
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

