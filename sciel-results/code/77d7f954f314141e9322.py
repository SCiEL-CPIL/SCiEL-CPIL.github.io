import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

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

def _covering_radius(p):
    """
    Compute the covering radius in degrees.
    Uses convex hull facets to find the largest empty cap.
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return 180.0
    
    max_r = 0.0
    for tri in facets:
        a, b, c = p[tri[0]], p[tri[1]], p[tri[2]]
        # Normal vector
        n_vec = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n_vec)
        if n_norm < 1e-15:
            continue
        n_vec = n_vec / n_norm
        
        # Check orientation (should point outward)
        # Midpoint of triangle
        mid = (a + b + c) / 3.0
        if np.dot(n_vec, mid) < 0:
            n_vec = -n_vec
            
        # Circumradius is the angle from circumcenter to vertex
        # Circumcenter on the sphere is the normalized normal (for spherical triangle)
        # Actually, the circumcenter of a spherical triangle is the point equidistant from all 3 vertices.
        # For a small triangle, it's close to the normalized normal.
        # But we need the exact angular distance.
        # The angular distance from the circumcenter to any vertex is the circumradius.
        # Let's compute the distance from the normalized normal to vertex a.
        # This is an approximation if the triangle is large, but for covering problems,
        # the facets are usually small enough.
        # A more robust way: solve for the point x on the sphere such that x.a = x.b = x.c.
        # This is the intersection of the sphere with the plane defined by the 3 points? No.
        # The circumcenter of a spherical triangle is the normalized vector (a x b) x (b x c) + ...?
        # Actually, the circumcenter is the normalized solution to:
        # x . a = x . b = x . c
        # This means x is orthogonal to (a-b) and (b-c). So x is parallel to (a-b) x (b-c).
        # So the circumcenter is indeed the normalized normal vector (with correct sign).
        
        cos_r = np.clip(np.dot(n_vec, a), -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
            
    return np.degrees(max_r)

def _pressure_objective(params, deadline):
    """
    Objective function for pressure minimization.
    We want to maximize the volume of the convex hull, which is equivalent to minimizing pressure.
    Actually, for a fixed number of points on the sphere, maximizing the volume of the convex hull
    tends to spread the points out.
    However, the covering radius is not exactly the same as hull volume.
    But they are correlated. A better configuration (smaller covering radius) usually has a larger hull volume.
    
    Let's try to minimize the sum of squared facet areas? No, that would collapse the hull.
    Let's try to maximize the volume.
    """
    if time.time() > deadline:
        return 0.0
        
    p = params.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / norms
    
    try:
        hull = ConvexHull(p)
        volume = hull.volume
    except Exception:
        return 0.0
        
    # We want to maximize volume, so minimize -volume
    return -volume

def _refine_pressure(config, deadline):
    """
    Refine configuration using L-BFGS-B to maximize hull volume (minimize pressure).
    """
    start_time = time.time()
    
    # Initial params
    x0 = config.flatten()
    
    # Bounds: each coordinate in [-1, 1] (since we normalize, the magnitude doesn't matter much,
    # but L-BFGS needs bounds. We can set them to [-2, 2] to allow some flexibility before normalization)
    bounds = [(-2.0, 2.0)] * (N * 3)
    
    # We use a wrapper that normalizes inside
    def obj(x):
        if time.time() > deadline:
            return 0.0
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        try:
            hull = ConvexHull(p)
            return -hull.volume
        except Exception:
            return 0.0
            
    # Use L-BFGS-B
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000})
    except Exception:
        res = None
        
    if res is not None and res.success:
        p_opt = res.x.reshape(N, 3)
        norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
        p_opt = p_opt / norms
        return p_opt
    else:
        return config

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
    best_score = _covering_radius(points)
    
    for step, iters in stages:
        if time.time() > deadline - 5:
            break
            
        # Lloyd iteration
        for _ in range(iters):
            if time.time() > deadline - 2:
                break
                
            try:
                hull = ConvexHull(points)
                facets = hull.simplices
            except Exception:
                break
                
            forces = np.zeros_like(points)
            
            for tri in facets:
                a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
                n_vec = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n_vec)
                if n_norm < 1e-15:
                    continue
                n_vec = n_vec / n_norm
                
                mid = (a + b + c) / 3.0
                if np.dot(n_vec, mid) < 0:
                    n_vec = -n_vec
                    
                cos_r = np.clip(np.dot(n_vec, a), -1.0, 1.0)
                r = np.arccos(cos_r)
                
                # Move vertices towards circumcenter
                # Weight by radius?
                weight = r
                forces[tri[0]] += weight * (n_vec - a)
                forces[tri[1]] += weight * (n_vec - b)
                forces[tri[2]] += weight * (n_vec - c)
                
            # Project to tangent plane
            dot_fp = np.sum(forces * points, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * points
            
            # Update
            points = points + step * forces_tan
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        current_score = _covering_radius(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()
        else:
            points = best_points.copy()
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_config = None
    best_score = float('inf')
    
    # Initial configurations
    inits = []
    inits.append(_fib_sample(N))
    for i in range(5):
        inits.append(_random_sample(N, seed=100 + i))
        
    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # 1. Pressure minimization (L-BFGS)
        refined_p = _refine_pressure(init, deadline - 10)
        score_p = _covering_radius(refined_p)
        
        if score_p < best_score:
            best_score = score_p
            best_config = refined_p.copy()
            
        # 2. Lloyd refinement from the pressure-refined config
        if time.time() < deadline - 10:
            refined_l, score_l = _refine_lloyd(refined_p, deadline - 5)
            
            if score_l < best_score:
                best_score = score_l
                best_config = refined_l.copy()
                
        # If we get very good, stop
        if best_score < 18.31:
            break
            
    # Final polish
    if best_config is not None and time.time() < deadline - 5:
        final_config, final_score = _refine_lloyd(best_config, deadline - 1)
        if final_score < best_score:
            best_score = final_score
            best_config = final_config
            
    if best_config is None:
        best_config = _fib_sample(N)
        
    # Final normalization
    best_config = np.array([_normalize(p) for p in best_config])
    
    return best_config
