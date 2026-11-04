"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
TIME_LIMIT = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 130
TIME_LIMIT = 115.0


def _covering_radius(pts):
    """Compute the covering radius in radians from the convex hull."""
    try:
        hull = ConvexHull(pts)
        # Get the facets (each is a set of point indices)
        # For each facet, compute the circumcenter and its distance to the facet vertices
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the vertices of this facet
            verts = pts[simplex]
            # Compute the circumcenter of the simplex
            # For a triangle, solve for the point equidistant from all vertices
            # Use the formula: circumcenter is the intersection of perpendicular bisectors
            # In 3D for a triangle, project to 2D or use the general formula
            
            # General approach: find point c minimizing max |c - v_i| s.t. c on plane
            # For a triangle with vertices a, b, c, the circumcenter is:
            # Using the formula with cross products
            a, b, c = verts[0], verts[1], verts[2]
            
            # Compute the circumcenter of triangle abc
            # Using the formula: O = (|a|^2*(b×c) + |b|^2*(c×a) + |c|^2*(a×b)) / (2*(a×b)·c)
            # But this assumes the triangle is in a plane through origin, which it's not necessarily.
            # Better: work in the plane of the triangle.
            
            # Translate so that a is at origin
            b_rel = b - a
            c_rel = c - a
            
            # Normal to the plane
            n = np.cross(b_rel, c_rel)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # Project b_rel and c_rel onto the plane (they already are)
            # Solve for circumcenter in 2D coordinates
            # Let u = b_rel, v = c_rel
            # Circumcenter in the plane: O = a + alpha*u + beta*v
            # |O|^2 = |O - u|^2 = |O - v|^2
            # This gives: 2*alpha*(u·u) + 2*beta*(u·v) = |u|^2
            #              2*alpha*(u·v) + 2*beta*(v·v) = |v|^2
            
            uu = np.dot(b_rel, b_rel)
            vv = np.dot(c_rel, c_rel)
            uv = np.dot(b_rel, c_rel)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-12:
                continue
            
            alpha = (vv * uv - vv * uv) / (2 * det)  # This is wrong, let me redo
            # Actually:
            # 2*alpha*uu + 2*beta*uv = uu  => alpha*uu + beta*uv = uu/2
            # 2*alpha*uv + 2*beta*vv = vv  => alpha*uv + beta*vv = vv/2
            
            # Using Cramer's rule:
            alpha = (0.5 * uu * vv - 0.5 * vv * uv) / det
            beta = (0.5 * uu * uv - 0.5 * uv * vv) / det
            
            # Wait, let me just use numpy's solve
            A = np.array([[uu, uv], [uv, vv]])
            b_vec = np.array([0.5 * uu, 0.5 * vv])
            try:
                coeffs = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                continue
            alpha, beta = coeffs
            
            circumcenter = a + alpha * b_rel + beta * c_rel
            
            # Distance from circumcenter to any vertex
            r = np.linalg.norm(circumcenter - a)
            if r > max_r:
                max_r = r
        
        return max_r
    except Exception:
        # Fallback: compute max distance from origin to hull vertices
        # This is an upper bound on the covering radius
        return np.max(np.linalg.norm(pts, axis=1))


def _soft_covering_radius(pts, beta=100.0):
    """
    Soft approximation of the covering radius using log-sum-exp.
    This is differentiable and can be used for gradient-based optimization.
    """
    # Sample points on the sphere
    n_samples = 5000
    # Use Fibonacci spiral for good coverage
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each sample to each point
    # Use dot product since both are unit vectors
    dots = samples @ pts.T  # shape (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # angular distances
    
    # For each sample, find the minimum distance to any point
    min_dists = np.min(angles, axis=1)
    
    # Soft max using log-sum-exp
    # We want to approximate max(min_dists)
    # Use: log(sum(exp(beta * x))) / beta approximates max(x) for large beta
    scaled = beta * min_dists
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return log_sum_exp / beta


def _gradient_soft_covering_radius(pts, beta=100.0, n_samples=3000):
    """
    Compute the gradient of the soft covering radius with respect to pts.
    """
    # Sample points on the sphere
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # dots[s, k] = samples[s] . pts[k]
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each sample, find the nearest point
    min_dots_idx = np.argmax(dots, axis=1)  # index of nearest point (max dot = min angle)
    
    # For the soft max, we need the weights
    # min_angle[s] = arccos(dots[s, min_idx[s]])
    min_dots = dots[np.arange(n_samples), min_dots_idx]
    min_angles = np.arccos(min_dots)
    
    # Soft max weights
    scaled = beta * min_angles
    max_val = np.max(scaled)
    weights = np.exp(scaled - max_val)
    weights /= np.sum(weights)
    
    # Gradient with respect to pts[k]
    # d/dpts[k] of soft_max = sum_s weights[s] * d/dpts[k] (min_angle[s])
    # min_angle[s] = arccos(dots[s, k_s]) where k_s = min_dots_idx[s]
    # d/dpts[k] arccos(d) = -1/sqrt(1-d^2) * d/dpts[k](d)
    # d/dpts[k](dots[s,k]) = samples[s]
    
    grad = np.zeros_like(pts)
    for s in range(n_samples):
        k = min_dots_idx[s]
        d = dots[s, k]
        denom = np.sqrt(np.maximum(1.0 - d * d, 1e-12))
        grad[k] += weights[s] * (-samples[s] / denom)
    
    # Normalize gradient to lie in the tangent space (perpendicular to pts[k])
    for k in range(N):
        grad[k] -= np.dot(grad[k], pts[k]) * pts[k]
    
    return grad


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lloyd_iterate(pts, n_iters=50, beta=50.0):
    """
    Spherical Lloyd's algorithm: move each point to the area-weighted centroid
    of its Voronoi cell, approximated by the circumcenter of its largest facet.
    """
    pts = pts.copy()
    
    for it in range(n_iters):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell approximation
        # The Voronoi cell of point k is the set of points on the sphere closer to k than to any other
        # Approximate by the intersection of half-spaces, but that's expensive.
        # Instead, use the circumcenters of adjacent facets as attractors.
        
        # For each facet, compute the circumcenter and assign it to the facet's vertices
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            
            b_rel = b - a
            c_rel = c - a
            
            n = np.cross(b_rel, c_rel)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            
            uu = np.dot(b_rel, b_rel)
            vv = np.dot(c_rel, c_rel)
            uv = np.dot(b_rel, c_rel)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-12:
                continue
            
            A = np.array([[uu, uv], [uv, vv]])
            b_vec = np.array([0.5 * uu, 0.5 * vv])
            try:
                coeffs = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                continue
            alpha, beta_coeff = coeffs
            
            circumcenter = a + alpha * b_rel + beta_coeff * c_rel
            cc_norm = np.linalg.norm(circumcenter)
            if cc_norm < 1e-12:
                continue
            circumcenter = circumcenter / cc_norm
            
            # Weight by the area of the facet
            area = 0.5 * n_norm
            
            for idx in simplex:
                new_pts[idx] += area * circumcenter
                counts[idx] += area
        
        # Normalize
        valid = counts > 1e-12
        for k in range(N):
            if valid[k]:
                new_pts[k] /= counts[k]
            else:
                new_pts[k] = pts[k]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        new_pts = new_pts / norms
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        
        if diff < 1e-10:
            break
    
    return pts


def construct():
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Different initializations
        if restart == 0:
            pts = _fibonacci_sphere(N)
        elif restart == 1:
            pts = _fibonacci_sphere(N, offset=0.3)
        elif restart == 2:
            pts = _random_sphere(N, seed=42)
        elif restart == 3:
            pts = _random_sphere(N, seed=123)
        elif restart == 4:
            pts = _fibonacci_sphere(N, offset=1.0)
        elif restart == 5:
            pts = _random_sphere(N, seed=777)
        elif restart == 6:
            pts = _fibonacci_sphere(N, offset=2.5)
        else:
            pts = _random_sphere(N, seed=999)
        
        # Phase 1: Lloyd's algorithm
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - 20:
            pts = _lloyd_iterate(pts, n_iters=100)
        
        # Phase 2: Gradient-based refinement using soft covering radius
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - 15:
            def objective(x):
                p = x.reshape(N, 3)
                # Normalize to unit sphere
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                p = p / norms
                return _soft_covering_radius(p, beta=80.0, n_samples=2000)
            
            def gradient(x):
                p = x.reshape(N, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                p = p / norms
                g = _gradient_soft_covering_radius(p, beta=80.0, n_samples=2000)
                return g.flatten()
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    objective,
                    pts.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'gtol': 1e-8}
                )
                pts = result.x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                pts = pts / norms
            except Exception:
                pass
        
        # Phase 3: Final Lloyd refinement
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - 10:
            pts = _lloyd_iterate(pts, n_iters=50)
        
        # Evaluate
        radius = _covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final check
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
