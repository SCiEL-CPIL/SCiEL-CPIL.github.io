"""
Covering the sphere with 50 equal caps.
Uses scipy.spatial.ConvexHull to compute exact covering radius via facet circumradii.
Optimizes with L-BFGS-B on a smooth surrogate, then polishes with exact evaluations.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _exact_covering_radius(pts):
    """Compute the exact covering radius in radians using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the three vertices of the triangular facet
            v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Compute the circumcenter of the triangle on the unit sphere
            # The circumcenter of a spherical triangle is along the normal to the plane
            # containing the three points, at distance 1/cos(alpha) from the origin
            # where alpha is the angular radius.
            # 
            # Alternative: compute the circumradius directly.
            # For a triangle with vertices a, b, c on the unit sphere,
            # the circumradius R satisfies: sin(R) = |a x b x c| / (2 * |a x b| * |b x c| * |c x a|)
            # Actually, let's use a more robust method.
            #
            # The circumcenter C of the triangle (a,b,c) on the sphere is the point
            # equidistant from all three vertices. It lies along the vector:
            #   C_dir = (a x b) x (b x c)  (or similar cross product combination)
            # Actually, the normal to the plane containing a, b, c is n = (b-a) x (c-a)
            # The circumcenter on the sphere is in the direction of n (or -n),
            # and the angular distance from C to a is arccos(C . a).
            #
            # Let's compute n = (b - a) x (c - a), normalize it.
            # Then C = n / |n| (or -n/|n|, choose the one that gives positive dot with a)
            # The circumradius is arccos(|C . a|).
            
            ab = v1 - v0
            ac = v2 - v0
            n = np.cross(ab, ac)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # Choose sign so that n points towards the interior (away from origin side)
            # Actually, for a convex hull, the outward normal points away from the origin.
            # The circumcenter should be on the same side as the facet normal for the covering.
            # Let's just compute both and take the one with larger distance.
            c1 = n
            c2 = -n
            d1 = np.abs(np.dot(c1, v0))
            d2 = np.abs(np.dot(c2, v0))
            # Both should give the same |dot|, so circumradius is arccos(d1)
            r = np.arccos(np.clip(d1, -1.0, 1.0))
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        return np.pi  # Fallback: return a large value

def _smooth_energy(pts, beta=30.0, M=3000):
    """
    Smooth surrogate for the covering radius using soft-max over a grid.
    Returns (energy, gradient).
    """
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute dots: (M, N)
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # soft-max over i for each grid point
    shifted = beta * dots
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    g = np.sum(weights * dots, axis=1)  # soft-max of dots
    
    g = np.clip(g, -1.0 + 1e-12, 1.0 - 1e-12)
    d = np.arccos(g)
    
    # soft-max over grid points
    shifted2 = beta * d
    max_shifted2 = np.max(shifted2, keepdims=True)
    exp_shifted2 = np.exp(shifted2 - max_shifted2)
    weights2 = exp_shifted2 / np.sum(exp_shifted2)
    E = np.sum(weights2 * d)
    
    # Gradient computation
    # dE/dg = sum_y weights2 * (-1/sqrt(1-g^2))
    dg = -weights2 / np.sqrt(np.maximum(1.0 - g * g, 1e-12))
    
    # dg/ddots_j = weights[:, j] for each grid point
    # dE/dpts = sum_y dg[y] * sum_j (dg/ddots_j) * d(dots_j)/dpts
    # dots_j = grid_y . pts_j
    # d(dots_j)/dpts_j = grid_y
    # So dE/dpts_j = sum_y dg[y] * weights[y, j] * grid_y
    
    # grad shape: (N, 3)
    grad = np.zeros((N, 3))
    # For each grid point y, contribution to grad[j] = dg[y] * weights[y, j] * grid[y]
    # grad[j] = sum_y (dg[y] * weights[y, j]) * grid[y]
    # This is: grad = (weights * dg[:, None]).T @ grid
    weighted_weights = weights * dg[:, None]  # (M, N)
    grad = weighted_weights.T @ grid  # (N, 3)
    
    return E, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configuration
    best_pts = _fibonacci_sphere(N)
    best_radius = _exact_covering_radius(best_pts)
    
    # Try multiple initializations
    rng = np.random.default_rng(42)
    inits = [best_pts.copy()]
    for _ in range(5):
        random_pts = rng.standard_normal((N, 3))
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        inits.append(random_pts)
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        current_pts = init_pts.copy()
        
        # Optimize the smooth surrogate
        def obj_and_grad(x):
            pts = x.reshape(N, 3)
            # Renormalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            pts = pts / norms
            E, grad = _smooth_energy(pts)
            # Project gradient to tangent space
            grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            return E, grad.ravel()
        
        x0 = current_pts.ravel()
        try:
            result = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'gtol': 1e-6})
            if result.success or result.fun < best_radius:
                current_pts = result.x.reshape(N, 3)
                norms = np.linalg.norm(current_pts, axis=1, keepdims=True)
                current_pts = current_pts / np.maximum(norms, 1e-12)
                radius = _exact_covering_radius(current_pts)
                if radius < best_radius:
                    best_radius = radius
                    best_pts = current_pts.copy()
        except Exception:
            pass
        
        # Local polish: move points towards worst facets
        if time.time() > deadline - 10:
            break
        
        for it in range(100):
            if time.time() > deadline - 5:
                break
            
            # Find worst facet
            try:
                hull = ConvexHull(current_pts)
                max_r = 0.0
                worst_simplex = None
                for simplex in hull.simplices:
                    v0, v1, v2 = current_pts[simplex[0]], current_pts[simplex[1]], current_pts[simplex[2]]
                    ab = v1 - v0
                    ac = v2 - v0
                    n = np.cross(ab, ac)
                    n_norm = np.linalg.norm(n)
                    if n_norm < 1e-12:
                        continue
                    n = n / n_norm
                    d = np.abs(np.dot(n, v0))
                    r = np.arccos(np.clip(d, -1.0, 1.0))
                    if r > max_r:
                        max_r = r
                        worst_simplex = simplex
                if worst_simplex is None:
                    break
                
                # Move the three vertices of the worst facet towards the circumcenter
                v0, v1, v2 = current_pts[worst_simplex[0]], current_pts[worst_simplex[1]], current_pts[worst_simplex[2]]
                ab = v1 - v0
                ac = v2 - v0
                n = np.cross(ab, ac)
                n = n / np.linalg.norm(n)
                
                # Move each vertex slightly towards the circumcenter direction
                step = 0.02
                for idx in worst_simplex:
                    current_pts[idx] += step * n
                    current_pts[idx] /= np.linalg.norm(current_pts[idx])
                
                new_radius = _exact_covering_radius(current_pts)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = current_pts.copy()
            except Exception:
                break
    
    # Final renormalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-12)
    
    return best_pts
