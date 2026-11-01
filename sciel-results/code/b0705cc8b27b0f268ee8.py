import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sample(n, seed=None):
    """Generate n random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _compute_hull_data(p):
    """
    Compute convex hull facets and their circumcenters/radii.
    Returns facets (indices), circumcenters (unit vectors), radii (radians).
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    a = p[facets[:, 0]]  # (F, 3)
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b  # (F, 3)
    v2 = b - c
    n = np.cross(v1, v2)  # (F, 3)
    norm_n = np.linalg.norm(n, axis=1)  # (F,)
    
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        # Normalized normal vectors
        cc = n[valid] / norm_n[valid, np.newaxis]
        
        # Ensure circumcenter faces outward (dot with vertex > 0)
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        # Circumradius is arccos(dot(circumcenter, vertex))
        # Since all vertices of a facet are equidistant from the circumcenter in terms of angle
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
        
    return facets, circumcenters, radii

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    facets, circumcenters, radii = _compute_hull_data(config)
    if radii is None:
        return 180.0 # Invalid or degenerate
    return np.degrees(np.max(radii))

def _lloyd_iteration(config, step_size=0.1, max_iter=500, deadline=None, tol=1e-7):
    """
    Lloyd-like iteration to minimize the covering radius.
    Moves vertices of large facets towards their circumcenters.
    """
    points = config.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        facets, circumcenters, radii = _compute_hull_data(points)
        if radii is None:
            break
            
        r_max = np.max(radii)
        
        # If radius is very small, stop
        if r_max < tol:
            break
            
        # Identify "bad" facets: those contributing to the max radius
        # We use a threshold to include facets close to the max to smooth the gradient
        threshold = r_max - 0.05 * r_max
        bad_mask = radii > threshold
        
        if not np.any(bad_mask):
            break
            
        # Calculate forces
        # For each bad facet, we want to move its 3 vertices towards the circumcenter
        # Force on vertex v is proportional to (cc - v)
        # We sum these forces for each point
        
        forces = np.zeros_like(points)
        
        # Vectorized force accumulation
        # facets_bad: (B, 3) indices of vertices
        # cc_bad: (B, 3) circumcenters
        # r_bad: (B,) radii
        
        facets_bad = facets[bad_mask]
        cc_bad = circumcenters[bad_mask]
        r_bad = radii[bad_mask]
        
        # Weight the force by how bad the facet is? 
        # Simple approach: uniform weight for all bad facets.
        # Or weight by (r_i - r_avg). Let's stick to uniform for stability first.
        
        # For each bad facet i, add (cc_i - v_j) to force of v_j for j in 0,1,2
        # We can do this with np.add.at
        
        # Flatten indices: (B, 3)
        idx_flat = facets_bad.ravel()
        # Repeat cc for each vertex: (B, 3) -> (B, 3, 3) -> (B*3, 3)
        cc_repeated = np.repeat(cc_bad, 3, axis=0)
        
        # Vertices corresponding to these indices
        v_flat = points[idx_flat]
        
        # Force vector for each vertex in the bad facets
        f_flat = cc_repeated - v_flat
        
        # Accumulate forces
        np.add.at(forces, idx_flat, f_flat)
        
        # Project forces onto tangent plane
        # F_tan = F - (F . p) p
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update points
        points = points + step_size * forces_tan
        
        # Renormalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    return points

def _refine_lloyd(config, deadline):
    """
    Multi-stage Lloyd refinement with decreasing step size.
    """
    points = config.copy()
    
    # Stages: (step_size, max_iter)
    stages = [
        (0.5, 50),
        (0.2, 100),
        (0.1, 200),
        (0.05, 300),
        (0.02, 500),
        (0.01, 1000),
    ]
    
    best_points = points.copy()
    best_score = _covering_radius_deg_hull(points)
    
    for step, iters in stages:
        if time.time() > deadline - 5:
            break
            
        points = _lloyd_iteration(points, step_size=step, max_iter=iters, deadline=deadline - 2)
        
        current_score = _covering_radius_deg_hull(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()
        else:
            # If score worsened, revert and try smaller step or break?
            # Lloyd should generally improve, but numerical issues might occur.
            # Revert to best known
            points = best_points.copy()
            
    return best_points, best_score

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. We target 140s.
    deadline = start_time + 140.0
    
    best_config = None
    best_score = float('inf')
    
    # 1. Generate multiple initial configurations
    # Strategy: Mix of Fibonacci and Random
    inits = []
    
    # Fibonacci
    inits.append(_fib_sample(N))
    
    # Random starts
    for i in range(10):
        inits.append(_random_sample(N, seed=42 + i))
        
    # 2. Run Lloyd refinement on each
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Quick check: ensure not in a hemisphere
        # If all points are in a hemisphere, the convex hull will be a flat polygon or similar,
        # and covering radius will be large.
        # A simple check: if the origin is inside the convex hull.
        # ConvexHull will raise an error if points are coplanar, but not necessarily if they are in a hemisphere.
        # However, Lloyd iteration will likely fail or produce bad results if the hull is degenerate.
        
        # Run refinement
        # Allocate time based on remaining budget
        # Rough estimate: each Lloyd stage takes some time.
        # We'll just use the deadline.
        
        refined_config, score = _refine_lloyd(init, deadline - 5)
        
        if score < best_score:
            best_score = score
            best_config = refined_config.copy()
            
        # If we found a very good solution, we can stop early?
        # Theoretical lower bound is around 18.3 deg.
        # If we get below 18.35, it's excellent.
        if best_score < 18.31:
            break

    # 3. Final Polish
    # If we have time, do a very high precision Lloyd run on the best config
    if best_config is not None and time.time() < deadline - 2:
        # Very small step, many iterations
        final_config = _lloyd_iteration(best_config, step_size=0.005, max_iter=2000, deadline=deadline - 1)
        final_score = _covering_radius_deg_hull(final_config)
        
        if final_score < best_score:
            best_score = final_score
            best_config = final_config

    # Ensure valid output
    if best_config is None:
        # Fallback to Fibonacci if everything failed
        best_config = _fib_sample(N)
        
    # Final normalization
    best_config = np.array([_normalize(p) for p in best_config])
    
    return best_config
