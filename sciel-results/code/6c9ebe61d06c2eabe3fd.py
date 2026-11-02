import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def compute_hull_data(p):
        """
        Computes the convex hull of the points and returns the circumradii of the facets.
        The covering radius is the maximum of these circumradii.
        """
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None
            
        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        
        # Calculate normal vectors
        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)
        
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            # Normalize normals to get circumcenters (since points are on unit sphere, 
            # the circumcenter of a facet of the convex hull of points on the sphere 
            # is the normalized normal vector, provided the facet is small enough / convexity holds)
            cc = n[valid] / norm_n[valid, np.newaxis]
            
            # Ensure circumcenter is on the same side as the points (dot product check)
            # For a convex hull of points on the sphere, the circumcenter of a facet 
            # (triangle abc) is the point on the sphere such that the plane abc is the 
            # tangent plane at that point? No.
            # The circumcenter of the triangle abc in 3D space lies on the line passing 
            # through the origin perpendicular to the plane abc.
            # Since a, b, c are on the unit sphere, the circumcenter of the triangle abc 
            # in 3D is exactly the normalized normal vector n/|n|.
            # The distance from the origin to the plane is 1/|n| * |n| = 1? No.
            # The distance from origin to plane is |dot(a, n)| / |n|.
            # Since a is on the unit sphere and n is normal to the plane containing a,
            # the projection of a onto n is the circumcenter.
            # Let's verify: The circumcenter C of triangle abc satisfies |C-a|=|C-b|=|C-c|.
            # C = k * n_hat.
            # |k n_hat - a|^2 = k^2 - 2k (a.n_hat) + 1.
            # We want this to be equal for a, b, c.
            # Since a,b,c are in the plane, a.n_hat = b.n_hat = c.n_hat = h.
            # So the distance is the same for any k? No.
            # Wait, the circumcenter of the triangle in the plane is different from the 
            # point on the sphere that defines the cap.
            # The "covering radius" is the angular distance from any point y to the nearest 
            # chosen point.
            # The furthest point from the set of points on the sphere is the circumcenter 
            # of the spherical triangle formed by three consecutive points on the hull.
            # The angular distance is the angle between the circumcenter vector and any 
            # of the three points.
            
            dots = np.sum(cc * a[valid], axis=1)
            # If the dot product is negative, the normal is pointing inwards.
            cc[dots < 0] *= -1
            # Re-calculate dots after flip if necessary, but cc is now pointing outwards
            # relative to the origin for the "cap" side?
            # Actually, for a convex hull containing the origin, the normals point outwards.
            # The circumcenter of the spherical triangle is the normalized normal.
            
            circumcenters[valid] = cc
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
            
        return facets, circumcenters, radii

    def get_covering_radius(pts):
        """Returns the covering radius in radians."""
        _, _, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def soft_max_objective(params, beta=100.0):
        """
        Smoothed objective for global optimization.
        Uses log-sum-exp to approximate the max of the facet radii.
        """
        X = normalize(params.reshape(N, 3))
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        
        r_max = np.max(radii)
        if r_max < 1e-15:
            return 0.0
            
        # Shift to avoid overflow in exp
        shifted = beta * (radii - r_max)
        shifted = np.clip(shifted, -50, 50)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def hard_objective(params):
        """
        Hard objective for local optimization (Nelder-Mead).
        """
        X = normalize(params.reshape(N, 3))
        return get_covering_radius(X)

    # 1. Global Search using Differential Evolution
    # We optimize over the 3N coordinates.
    # Bounds: [-1, 1] for each coordinate.
    bounds = [(-1.0, 1.0)] * (N * 3)
    
    # Use a moderate population size and tolerance to balance time vs quality.
    # maxiter is set to allow convergence within the time budget.
    # We use the soft-max objective to provide a smoother landscape for DE.
    
    best_result = None
    best_val = float('inf')
    
    # Run DE with a timeout check inside the objective is hard, 
    # so we rely on maxiter and the fact that DE is parallelizable or fast enough.
    # We will run DE for a fixed number of iterations.
    
    try:
        de_res = differential_evolution(
            soft_max_objective, 
            bounds, 
            args=(100.0,), # beta for soft max
            maxiter=50, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            polish=False # We will do our own polishing
        )
        
        if de_res.success or de_res.fun < best_val:
            best_val = de_res.fun
            best_params = de_res.x
            best_result = normalize(best_params.reshape(N, 3))
            
    except Exception as e:
        pass

    # Fallback initializations if DE fails or to provide multiple starts
    initializations = []
    
    if best_result is not None:
        initializations.append(best_result)
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)
    
    # Random
    np.random.seed(123)
    rand_pts = np.random.randn(N, 3)
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
    initializations.append(rand_pts)
    
    # 2. Local Refinement using Nelder-Mead on the hard objective
    # Nelder-Mead is robust to non-smoothness and does not require gradients.
    
    current_best_pts = None
    current_best_radius = float('inf')
    
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
            
        # Normalize start
        start_pts = normalize(init_pts)
        
        try:
            # Nelder-Mead on the flattened vector
            res = minimize(
                hard_objective, 
                start_pts.flatten(), 
                method='Nelder-Mead',
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            new_pts = normalize(res.x.reshape(N, 3))
            new_radius = get_covering_radius(new_pts)
            
            if new_radius < current_best_radius:
                current_best_radius = new_radius
                current_best_pts = new_pts
                
        except Exception:
            continue

    # 3. Final Polishing with L-BFGS-B on Soft-Max
    # This helps with the final convergence precision where Nelder-Mead might be slow or stuck.
    
    if current_best_pts is not None and time.time() < deadline - 10.0:
        def obj_polish(params, beta=1000.0):
            return soft_max_objective(params, beta=beta)
        
        try:
            res = minimize(
                obj_polish, 
                current_best_pts.flatten(), 
                method='L-BFGS-B',
                options={
                    'maxiter': 200, 
                    'ftol': 1e-15, 
                    'gtol': 1e-14,
                    'maxls': 500
                }
            )
            
            new_pts = normalize(res.x.reshape(N, 3))
            new_radius = get_covering_radius(new_pts)
            
            if new_radius < current_best_radius:
                current_best_radius = new_radius
                current_best_pts = new_pts
                
        except Exception:
            pass

    # 4. One last Nelder-Mead pass on the best result found
    if current_best_pts is not None and time.time() < deadline - 2.0:
        try:
            res = minimize(
                hard_objective, 
                current_best_pts.flatten(), 
                method='Nelder-Mead',
                options={
                    'maxiter': 200, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            new_pts = normalize(res.x.reshape(N, 3))
            new_radius = get_covering_radius(new_pts)
            
            if new_radius < current_best_radius:
                current_best_radius = new_radius
                current_best_pts = new_pts
                
        except Exception:
            pass

    # Fallback if something went wrong
    if current_best_pts is None:
        current_best_pts = fib_pts

    # Final validation
    norms = np.linalg.norm(current_best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(current_best_pts)):
        current_best_pts = fib_pts

    return current_best_pts
