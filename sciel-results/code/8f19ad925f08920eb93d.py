"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 135.0

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the exact covering radius using Convex Hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback to sampling if hull fails
        return _covering_radius_sample(pts)
    
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Calculate circumcenter of the spherical triangle
        # The circumcenter O is the point on the sphere equidistant from a, b, c.
        # It lies in the direction of the normal to the plane of the triangle? No.
        # It is the intersection of the sphere with the line passing through the 3D circumcenter of triangle abc.
        
        # 3D circumcenter of triangle abc:
        # Solve for O such that |O-a|^2 = |O-b|^2 = |O-c|^2 and O is in the plane of abc.
        # Actually, we just need the direction.
        
        ab = b - a
        ac = c - a
        ab_ac = np.dot(ab, ac)
        ab_ab = np.dot(ab, ab)
        ac_ac = np.dot(ac, ac)
        
        det = ab_ab * ac_ac - ab_ac * ab_ac
        if abs(det) < 1e-10:
            # Degenerate
            center = (a + b + c) / 3.0
            center = center / np.linalg.norm(center)
        else:
            # Coefficients for O = a + u*ab + v*ac
            # 2 u ab_ab + 2 v ab_ac = ab_ab
            # 2 u ab_ac + 2 v ac_ac = ac_ac
            
            # u = (ab_ab * ac_ac - ab_ac * ab_ab) / (2 * det) ? No.
            # Cramer's rule:
            # u = (ab_ab * ac_ac - ab_ac * ab_ab) / (2 * (ab_ab * ac_ac - ab_ac^2))
            # Wait, the RHS is [ab_ab, ac_ac].
            # Matrix M = [[2*ab_ab, 2*ab_ac], [2*ab_ac, 2*ac_ac]]
            # u = det([[ab_ab, 2*ab_ac], [ac_ac, 2*ac_ac]]) / det(M)
            # u = (ab_ab * 2*ac_ac - 2*ab_ac * ac_ac) / (2 * det)
            # u = (ab_ab * ac_ac - ab_ac * ac_ac) / det  <-- This doesn't look right.
            
            # Let's just use np.linalg.solve
            M = np.array([[2*ab_ab, 2*ab_ac], [2*ab_ac, 2*ac_ac]])
            b_vec = np.array([ab_ab, ac_ac])
            try:
                uv = np.linalg.solve(M, b_vec)
                center_3d = a + uv[0]*ab + uv[1]*ac
                center = center_3d / np.linalg.norm(center_3d)
            except:
                center = (a + b + c) / 3.0
                center = center / np.linalg.norm(center)
        
        # Distance from center to vertices
        dots = np.clip([np.dot(center, a), np.dot(center, b), np.dot(center, c)], -1, 1)
        r = np.arccos(np.min(dots))
        max_r = max(max_r, r)
        
    return max_r

def _covering_radius_sample(pts, n_samples=10000):
    pts = _normalize(pts)
    rng = np.random.RandomState(42)
    # Uniform points on sphere
    u = rng.uniform(0, 1, n_samples)
    v = rng.uniform(0, 1, n_samples)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    samples = np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
    
    dots = samples @ pts.T
    max_dots = np.max(dots, axis=1)
    min_angles = np.arccos(np.clip(max_dots, -1, 1))
    return np.max(min_angles)

def _spherical_lloyd_iteration(pts, n_iter=20):
    """Perform Spherical Lloyd iterations to minimize covering radius."""
    pts = _normalize(pts)
    for _ in range(n_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (facets containing it)
        # The centroid of the Voronoi cell on the sphere is the new position.
        # Approximation: Average the normals of the facets? No.
        # The Voronoi cell of point i is the set of points on the sphere closer to i than any other.
        # The vertices of the Voronoi cell are the circumcenters of the facets containing i.
        # The centroid of the cell is the average of the vertices of the cell, normalized.
        
        # Map: point index -> list of facet vertices (circumcenters)
        point_to_vertices = {i: [] for i in range(N)}
        
        for simplex in hull.simplices:
            a_idx, b_idx, c_idx = simplex
            a, b, c = pts[a_idx], pts[b_idx], pts[c_idx]
            
            # Compute circumcenter of triangle abc
            ab = b - a
            ac = c - a
            ab_ac = np.dot(ab, ac)
            ab_ab = np.dot(ab, ab)
            ac_ac = np.dot(ac, ac)
            
            M = np.array([[2*ab_ab, 2*ab_ac], [2*ab_ac, 2*ac_ac]])
            b_vec = np.array([ab_ab, ac_ac])
            try:
                uv = np.linalg.solve(M, b_vec)
                center_3d = a + uv[0]*ab + uv[1]*ac
                center = center_3d / np.linalg.norm(center_3d)
            except:
                center = (a + b + c) / 3.0
                center = center / np.linalg.norm(center)
            
            # Add this circumcenter to the Voronoi cell of each vertex
            point_to_vertices[a_idx].append(center)
            point_to_vertices[b_idx].append(center)
            point_to_vertices[c_idx].append(center)
        
        new_pts = np.zeros_like(pts)
        for i in range(N):
            if point_to_vertices[i]:
                verts = np.array(point_to_vertices[i])
                new_pts[i] = np.mean(verts, axis=0)
            else:
                new_pts[i] = pts[i]
        
        new_pts = _normalize(new_pts)
        # Check for improvement
        if np.linalg.norm(new_pts - pts) < 1e-8:
            break
        pts = new_pts
        
    return pts

def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_points(seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _normalize(pts)

def construct():
    t_start = time.time()
    
    best_pts = None
    best_cr = np.inf
    
    # Strategy: Try multiple initializations and optimize each
    initializations = [
        _fibonacci_points(),
        _random_points(1),
        _random_points(2),
        _random_points(3)
    ]
    
    for init_pts in initializations:
        if time.time() > DEADLINE - 15:
            break
            
        # 1. Spherical Lloyd
        pts = _spherical_lloyd_iteration(init_pts, n_iter=50)
        
        # 2. Local Gradient Descent on a smooth surrogate
        # We use a simple gradient descent on the sample-based objective
        # to polish the Lloyd result.
        
        def objective_and_grad(x):
            x = _normalize(x.reshape(N, 3))
            # Sample based
            rng = np.random.RandomState(1)
            n_samples = 5000
            u = rng.uniform(0, 1, n_samples)
            v = rng.uniform(0, 1, n_samples)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            samples = np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
            
            dots = samples @ x.T
            # Soft max dot
            t = 0.05
            neg = -dots / t
            max_neg = np.max(neg, axis=1, keepdims=True)
            lse = max_neg.squeeze() + np.log(np.sum(np.exp(neg - max_neg), axis=1))
            soft_max_dot = -t * lse
            
            # Objective: -mean(soft_max_dot)
            val = -np.mean(soft_max_dot)
            
            # Gradient
            exp_neg = np.exp(neg - max_neg)
            sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
            weights = exp_neg / sum_exp
            
            grad_raw = -t * (weights.T @ samples)
            grad = grad_raw - (grad_raw * x).sum(axis=1, keepdims=True) * x
            
            return val, grad.flatten()
        
        # Run a few iterations of gradient descent
        x = pts.flatten()
        for _ in range(100):
            if time.time() > DEADLINE - 5:
                break
            val, grad = objective_and_grad(x)
            if np.linalg.norm(grad) < 1e-8:
                break
            # Simple step
            lr = 0.01
            x_new = x - lr * grad
            x_new = x_new.reshape(N, 3)
            x_new = _normalize(x_new)
            x = x_new.flatten()
            
        pts = _normalize(x.reshape(N, 3))
        
        # Evaluate
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    return best_pts
# EVOLVE-BLOCK-END
