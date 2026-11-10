"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _spherical_circumcenter(a, b, c):
    """
    Compute the circumcenter of a spherical triangle with vertices a, b, c on the unit sphere.
    The circumcenter is the point p on the sphere such that p·a = p·b = p·c (all equal).
    
    For three points on the unit sphere, the circumcenter lies in the plane containing a, b, c
    and the origin. We find p = αa + βb + γc such that p·a = p·b = p·c and |p| = 1.
    
    The condition p·a = p·b = p·c means p is in the intersection of the planes
    {x : x·(a-b) = 0} and {x : x·(a-c) = 0}, so p is parallel to (a-b) × (a-c).
    But we need to be careful: this gives the normal to the plane through a,b,c, which
    may not be the right direction for the spherical circumcenter.
    
    Actually, let me use a different approach. The circumcenter p satisfies:
    p·a = p·b = p·c = cos(r) for some r.
    
    This means (p - cos(r)*a) is orthogonal to a, and similarly for b, c.
    But more directly: p·(a-b) = 0 and p·(a-c) = 0.
    So p is in the null space of the matrix with rows (a-b) and (a-c).
    The null space is spanned by n = (a-b) × (a-c).
    
    So p = ±n/|n|. We choose the sign that gives the correct circumcenter.
    For the covering radius, we want the circumcenter that is farthest from the vertices
    (the one on the "outside" of the triangle relative to the origin).
    
    Let me verify: if p = n/|n| where n = (a-b) × (a-c), then:
    p·(a-b) = 0 and p·(a-c) = 0, so p·a = p·b and p·a = p·c.
    So p·a = p·b = p·c, which is what we want.
    
    The issue is which sign to choose. For a convex hull facet, the circumcenter
    that matters for the covering radius is the one that is "outside" the triangle,
    i.e., the one that is farthest from the origin in the direction away from the
    triangle's interior.
    
    Actually, for a point on the unit sphere, both ±n/|n| are on the sphere.
    The circumcenter for the covering radius is the one that maximizes the angular
    distance to the vertices, which corresponds to minimizing p·a (since angle = arccos(p·a)).
    """
    n = np.cross(a - b, a - c)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        # Degenerate: return the centroid direction
        centroid = a + b + c
        norm_c = np.linalg.norm(centroid)
        if norm_c < 1e-12:
            return a
        return centroid / norm_c
    
    p1 = n / norm_n
    p2 = -p1
    
    # Choose the circumcenter that gives the larger angular distance to the nearest vertex
    # i.e., the smaller minimum dot product
    min_dot1 = min(np.dot(p1, a), np.dot(p1, b), np.dot(p1, c))
    min_dot2 = min(np.dot(p2, a), np.dot(p2, b), np.dot(p2, c))
    
    # We want the one with smaller min dot product (larger angle)
    if min_dot1 <= min_dot2:
        return p1
    else:
        return p2


def _compute_circumcenters(pts, hull):
    """
    Compute circumcenters for all triangular facets of the convex hull.
    Returns array of circumcenters (one per simplex) and the angular radius for each.
    """
    n_facets = len(hull.simplices)
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        cc = _spherical_circumcenter(a, b, c)
        circumcenters[i] = cc
        
        # Angular distance from circumcenter to nearest vertex
        dots = np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)])
        min_dot = np.min(dots)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        radii[i] = np.arccos(min_dot)
    
    return circumcenters, radii


def _cover_radius(pts):
    """
    Compute the covering radius using the convex hull facets.
    The covering radius is the maximum circumradius over all facets.
    """
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    _, radii = _compute_circumcenters(pts, hull)
    return np.max(radii)


def _hill_climb_optimize(pts, max_iters=2000, initial_step=0.01, time_budget=60.0):
    """
    Hill climbing optimization: iteratively move each point to reduce the local
    maximum of the covering radius. Use the convex hull to identify which facets
    are critical.
    """
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius(pts)
    no_improve_count = 0
    start_time = time.time()
    
    for it in range(max_iters):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Compute circumcenters and radii for all facets
        circumcenters, radii = _compute_circumcenters(pts, hull)
        
        # Find the maximum radius facet
        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        
        # Get the three vertices of the critical facet
        critical_simplex = hull.simplices[max_idx]
        
        # For each vertex of the critical facet, try moving it to reduce the radius
        # The direction to move is away from the circumcenter
        cc = circumcenters[max_idx]
        
        improved = False
        for idx in critical_simplex:
            v = pts[idx]
            
            # Direction from circumcenter to vertex (push vertex away from cc)
            dir_vec = v - cc
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-15:
                continue
            dir_vec = dir_vec / dir_norm
            
            # Try a small step in this direction
            new_pts = pts.copy()
            new_pts[idx] = v + step * dir_vec
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_radius = _cover_radius(new_pts)
            
            if new_radius < best_radius - 1e-10:
                pts = new_pts
                best_radius = new_radius
                improved = True
                break
        
        if not improved:
            # Try smaller steps
            step *= 0.8
            if step < 1e-8:
                break
            no_improve_count += 1
            if no_improve_count > 30:
                break
        else:
            no_improve_count = 0
            # Slowly increase step if making progress
            if it % 50 == 0 and step < initial_step:
                step = min(step * 1.1, initial_step)
    
    return pts, best_radius


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(N, rng):
    """Generate N random points on the unit sphere."""
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_fibonacci(N, rng, noise_scale=0.1):
    """Generate a perturbed Fibonacci spiral."""
    base = _fibonacci_spiral(N)
    noise = rng.randn(N, 3) * noise_scale
    pts = base + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius(best_pts)
    
    # Hill climbing optimization from Fibonacci start
    if time.time() < deadline - 30:
        time_budget = min(30.0, deadline - 30 - (time.time() - start_time))
        optimized, opt_radius = _hill_climb_optimize(pts, max_iters=3000, initial_step=0.01, time_budget=time_budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Try multiple random restarts with hill climbing
    rng = np.random.RandomState(42)
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate different initial configurations
        if restart % 3 == 0:
            # Perturbed Fibonacci
            random_pts = _perturbed_fibonacci(N, rng, noise_scale=0.05 + 0.02 * restart)
        elif restart % 3 == 1:
            # Pure random
            random_pts = _random_sphere_points(N, rng)
        else:
            # Fibonacci with different phase
            i = np.arange(N) + 0.5 + restart * 0.1
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + restart * 0.3
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            random_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        remaining = deadline - 20 - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Estimate time budget for this restart
        time_per_restart = remaining / (max_restarts - restart)
        time_budget = min(time_per_restart * 0.8, 20.0)
        
        optimized, opt_radius = _hill_climb_optimize(random_pts, max_iters=2000, initial_step=0.01, time_budget=time_budget)
        
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final refinement: more iterations on the best configuration
    if time.time() < deadline - 5:
        time_budget = deadline - 5 - (time.time() - start_time)
        if time_budget > 5:
            optimized, opt_radius = _hill_climb_optimize(best_pts, max_iters=5000, initial_step=0.005, time_budget=time_budget)
            if opt_radius < best_radius:
                best_pts = optimized
                best_radius = opt_radius
    
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    # Normalize to ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
