"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Constants
_DEADLINE = 140.0  # Safety margin before 150s kill
_N_SAMPLES = 20000  # Reduced samples for speed, relying on gradient accuracy
_BETA = 50.0        # Sharpness of LogSumExp approximation

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

def _generate_random_samples(n_samples, rng):
    """Generate random points on the sphere."""
    pts = rng.normal(size=(n_samples, 3))
    return _normalize(pts)

def _soft_min_obj_grad(flat_pts, samples):
    """
    Compute the LogSumExp approximation of the covering radius and its gradient.
    
    Objective: Approximate R = max_y min_i d(y, p_i)
    Using dots: min_i d(y, p_i) = arccos(max_i y.p_i)
    Let M_y = max_i y.p_i. We want to minimize max_y arccos(M_y).
    Since arccos is decreasing, this is equivalent to minimizing min_y M_y (roughly, 
    but we actually want to maximize the minimum of the max dot products).
    
    Wait, let's look at the geometry again.
    Covering radius R is the smallest r such that union of caps of radius r covers sphere.
    For a point y, distance to nearest p_i is d(y, p_i).
    We want to minimize max_y min_i d(y, p_i).
    
    Let f(y) = min_i d(y, p_i).
    We want to minimize max_{y in S^2} f(y).
    
    Using the LogSumExp trick for max:
    max_y g(y) ≈ (1/beta) * log( sum_y exp(beta * g(y)) )
    
    Here g(y) = min_i d(y, p_i).
    This is still a min inside.
    
    Alternative formulation:
    min_i d(y, p_i) = arccos( max_i y.p_i ).
    Let h(y) = max_i y.p_i.
    We want to minimize max_y arccos(h(y)).
    Since arccos is monotonic decreasing, minimizing max_y arccos(h(y)) is equivalent to 
    maximizing min_y h(y).
    
    So the goal is to **maximize** the minimum over all y of (max over i of y.p_i).
    Let Z = min_y max_i y.p_i.
    We want to maximize Z.
    
    Using LogSumExp for min:
    min_y u(y) ≈ - (1/beta) * log( sum_y exp(-beta * u(y)) )
    
    So we want to maximize:
    Z ≈ - (1/beta) * log( sum_y exp(-beta * max_i y.p_i) )
    
    And max_i y.p_i can be approximated by LogSumExp for max as well, or computed exactly 
    since N=50 is small. Computing max_i y.p_i exactly is fast (vectorized argmax).
    
    Let M_y = max_i (y . p_i).
    We want to maximize J = - (1/beta) * log( sum_y exp(-beta * M_y) ).
    
    Gradient of J wrt p_k:
    dJ/dp_k = - (1/beta) * d/dp_k [ log( sum_y exp(-beta M_y) ) ]
    = - (1/beta) * [ 1 / (sum_y exp(-beta M_y)) ] * sum_y [ exp(-beta M_y) * d/dp_k (-beta M_y) ]
    = (1 / sum_y exp(-beta M_y)) * sum_y [ exp(-beta M_y) * d/dp_k M_y ]
    
    M_y = max_i y.p_i.
    d/dp_k M_y = y_k if k is the argmax index for y, else 0.
    (Subgradient if tie, but probability of tie is 0 for continuous y).
    
    So dJ/dp_k = sum_y [ w_y * y_k * I(k == argmax_y) ]
    where w_y = exp(-beta M_y) / sum_j exp(-beta M_j).
    
    We want to MAXIMIZE J, so we MINIMIZE -J.
    Objective for minimizer: F = -J.
    dF/dp_k = - dJ/dp_k = - sum_y [ w_y * y_k * I(k == argmax_y) ]
    
    We also need to project this gradient to the tangent space of the sphere for each p_k.
    grad_tangent_k = grad_k - (grad_k . p_k) p_k.
    """
    pts = flat_pts.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute M_y = max_i y . p_i
    dots = samples @ pts.T  # (n_samples, N)
    M = dots.max(axis=1)    # (n_samples,)
    argmax_idx = dots.argmax(axis=1) # (n_samples,)
    
    # Compute weights w_y
    # To avoid overflow/underflow, subtract max exponent
    exponents = -_BETA * M
    max_exp = exponents.max()
    exp_terms = np.exp(exponents - max_exp)
    sum_exp = exp_terms.sum()
    weights = exp_terms / sum_exp
    
    # Compute gradient of J (which we maximize)
    # dJ/dp_k = sum_y w_y * y_k * I(k == argmax_y)
    # We can compute this by scattering the weighted samples to their nearest point.
    # grad_J = zeros((N, 3))
    # for each y: grad_J[argmax_idx[y]] += weights[y] * samples[y]
    
    grad_J = np.zeros((N, 3))
    np.add.at(grad_J, argmax_idx, weights[:, np.newaxis] * samples)
    
    # We minimize -J, so gradient is -grad_J
    grad_F = -grad_J
    
    # Project to tangent space
    # For each k, grad_F[k] -= (grad_F[k] . pts[k]) * pts[k]
    dots_grad = np.einsum('ij,ij->i', grad_F, pts)
    grad_F -= dots_grad[:, np.newaxis] * pts
    
    # Objective value: F = -J = (1/beta) * log(sum_y exp(-beta M_y))
    # Note: The constant max_exp cancels out in the log? 
    # log(sum exp(e_i - m)) = log(sum exp(e_i)) - m
    # So J = -1/beta * (log(sum_exp_original) - max_exp) ... wait.
    # sum_y exp(-beta M_y) = exp(max_exp) * sum_y exp(e_y - max_exp)
    # log(sum) = max_exp + log(sum_exp_terms)
    # J = -1/beta * (max_exp + log(sum_exp_terms))
    # F = 1/beta * (max_exp + log(sum_exp_terms))
    
    obj_val = (max_exp + np.log(sum_exp)) / _BETA
    
    return obj_val, grad_F.flatten()

def _refine_with_voronoi(pts, time_limit, start_time):
    """
    Refine points by moving them to the circumcenter of the facet of the convex hull
    that corresponds to the largest Voronoi cell (or largest circumradius).
    """
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_radius = np.inf
    
    # Estimate radius initially
    # Use a quick sample check
    rng = np.random.default_rng(42)
    samples = _generate_random_samples(10000, rng)
    dots = samples @ pts.T
    max_dots = dots.max(axis=1)
    min_max_dot = np.min(max_dots)
    best_radius = np.degrees(np.arccos(np.clip(min_max_dot, -1.0, 1.0)))
    
    while time.time() - start_time < time_limit:
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # For each point, find the facets it belongs to
        # The Voronoi cell of p_i is defined by the facets of the convex hull opposite to p_i?
        # Actually, the covering radius is the max circumradius of the facets of the convex hull.
        # Each facet (triangle) in the hull has a circumcenter.
        # The distance from the origin to this circumcenter (if it's a valid cap) or the angular radius
        # of the cap defined by the facet vertices determines the local covering radius.
        
        # For a facet with vertices v1, v2, v3, the circumcenter c is in the plane.
        # The cap defined by these 3 points has radius r such that cos(r) = c . n?
        # Actually, the "empty cap" condition means the cap centered at the circumcenter of the facet 
        # (projected to sphere) containing the 3 vertices is empty of other points.
        # The covering radius is the maximum of the angular radii of these caps.
        
        # Let's compute the circumradius for each facet.
        # Facets are in hull.simplices
        radii = []
        circumcenters = []
        
        for simplex in hull.simplices:
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # Compute circumcenter of triangle v1, v2, v3 on unit sphere
            # This is the point c on the sphere such that |c-v1|=|c-v2|=|c-v3|
            # c is in the plane defined by v1, v2, v3.
            # Normal to plane: n = (v2-v1) x (v3-v1)
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            # The circumcenter c lies along the normal from the centroid? No.
            # c = a*v1 + b*v2 + c*v3 such that c.n = 1 (if we want it on the sphere? No, c is on sphere)
            # Actually, the circumcenter of the spherical triangle is the intersection of the sphere
            # and the plane spanned by v1, v2, v3? No, the plane containing v1, v2, v3 does not pass through origin.
            # The circumcenter is the point on the sphere equidistant from v1, v2, v3.
            # It lies in the plane defined by v1, v2, v3.
            # Let c = x*v1 + y*v2 + z*v3.
            # c . v1 = c . v2 = c . v3 = cos(r)
            
            # Solve for c:
            # The circumcenter is given by the normalized sum of the unit vectors from the incenter?
            # A robust way:
            # c is proportional to (v1 * sin(2*angle1) + v2 * sin(2*angle2) + v3 * sin(2*angle3))?
            # Or simply solve the linear system for the plane intersection.
            
            # Plane equation: n . x = d, where d = n . v1
            d = np.dot(n, v1)
            
            # The circumcenter c is the intersection of the sphere |x|=1 and the plane n.x=d?
            # No, that's the center of the circle cut by the plane.
            # The circumcenter of the *spherical* triangle is the point on the sphere.
            # It is the normalized vector:
            # c_unnorm = v1 * (v2 x v3) + v2 * (v3 x v1) + v3 * (v1 x v2)
            # This is the standard formula for the circumcenter of a triangle on the sphere.
            
            c_unnorm = v1 * np.dot(v2, v3) - v1 * np.dot(v2, v3) # Wait, v1 x v2 x v3?
            # Formula: c = (v1 * |v2 x v3|^2 + v2 * |v3 x v1|^2 + v3 * |v1 x v2|^2) ? No.
            
            # Let's use the property that c is in the plane spanned by v1, v2, v3.
            # c = a v1 + b v2 + c v3
            # c.v1 = c.v2 = c.v3
            # (a v1 + b v2 + c v3).v1 = (a v1 + b v2 + c v3).v2
            # a + b(v2.v1) + c(v3.v1) = a(v1.v2) + b + c(v3.v2)
            # Similarly for v3.
            
            # This is a 3x3 linear system.
            # Let G be the Gram matrix of v1, v2, v3.
            G = np.array([
                [np.dot(v1, v1), np.dot(v1, v2), np.dot(v1, v3)],
                [np.dot(v2, v1), np.dot(v2, v2), np.dot(v2, v3)],
                [np.dot(v3, v1), np.dot(v3, v2), np.dot(v3, v3)]
            ])
            # We want G @ [a,b,c]^T = [1,1,1]^T ? No.
            # c . v_i = cos(r) for all i.
            # G @ coeffs = cos(r) * [1,1,1]^T
            # So coeffs is proportional to G^{-1} @ [1,1,1]^T.
            
            try:
                coeffs = np.linalg.solve(G, np.ones(3))
            except np.linalg.LinAlgError:
                continue
                
            c_vec = coeffs[0]*v1 + coeffs[1]*v2 + coeffs[2]*v3
            c_norm = np.linalg.norm(c_vec)
            if c_norm < 1e-10:
                continue
            c_center = c_vec / c_norm
            
            # The radius is arccos(c_center . v1)
            cos_r = np.dot(c_center, v1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            
            radii.append(r)
            circumcenters.append(c_center)
            
        if not radii:
            break
            
        radii = np.array(radii)
        circumcenters = np.array(circumcenters)
        
        max_r_idx = np.argmax(radii)
        max_r = radii[max_r_idx]
        max_r_deg = np.degrees(max_r)
        
        if max_r_deg < best_radius - 1e-5:
            best_radius = max_r_deg
            # Move the points forming this facet towards the circumcenter?
            # Or move all points?
            # A common heuristic: move the vertices of the largest facet towards the circumcenter.
            # Or move all points slightly towards the circumcenter of the largest facet?
            # Better: Move each point to the centroid of its Voronoi cell?
            # For now, let's try moving the vertices of the max facet towards its circumcenter.
            
            # Identify which points are in this facet
            facet_vertices = hull.simplices[max_r_idx]
            
            new_pts = pts.copy()
            target = circumcenters[max_r_idx]
            
            # Move vertices towards target
            for idx in facet_vertices:
                # Spherical interpolation or simple vector addition
                diff = target - pts[idx]
                # Project to tangent
                proj = diff - np.dot(diff, pts[idx]) * pts[idx]
                new_pts[idx] = pts[idx] + 0.1 * proj
                new_pts[idx] = _normalize(new_pts[idx])
            
            # Check if this improves the global radius (approximate with samples)
            dots = samples @ new_pts.T
            max_dots = dots.max(axis=1)
            min_max_dot = np.min(max_dots)
            approx_radius = np.degrees(np.arccos(np.clip(min_max_dot, -1.0, 1.0)))
            
            if approx_radius < best_radius + 1.0: # Allow some noise
                 pts = new_pts
                 # Re-evaluate exact radius for best tracking?
                 # We already know the max facet radius was max_r_deg.
                 # If we moved points, the facets changed.
                 # We'll just trust the loop to re-evaluate.
            
            # If no improvement or stagnation, break?
            # For now, just iterate.
            
        else:
            # Stagnation
            break
            
    return best_pts, best_radius

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Strategy:
    1. Multi-start optimization using LogSumExp approximation of covering radius.
    2. Refine using Voronoi/ConvexHull facet circumradii.
    """
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    n_restarts = 3
    
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 20:
            break
        
        # 1. Initialization: Fibonacci spiral + random perturbation
        rng = np.random.default_rng(100 + i)
        
        idx = np.arange(N) + 0.5
        z = 1.0 - 2.0 * idx / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
        r = np.sqrt(1.0 - z * z)
        base_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        noise = rng.normal(scale=0.2, size=(N, 3))
        init_pts = _normalize(base_pts + noise)
        
        # 2. LogSumExp Optimization
        flat_init = init_pts.flatten()
        samples = _generate_random_samples(_N_SAMPLES, rng)
        
        try:
            res = minimize(
                _soft_min_obj_grad,
                flat_init,
                args=(samples,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            opt_pts = _normalize(res.x.reshape(N, 3))
        except Exception:
            opt_pts = init_pts
        
        # 3. Voronoi Refinement
        remaining_time = _DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 5:
            refined_pts, refined_radius = _refine_with_voronoi(
                opt_pts, time_limit=remaining_time, start_time=start_time
            )
            opt_pts = refined_pts
            current_radius = refined_radius
        else:
            # Quick sample estimate
            dots = samples @ opt_pts.T
            max_dots = dots.max(axis=1)
            min_max_dot = np.min(max_dots)
            current_radius = np.degrees(np.arccos(np.clip(min_max_dot, -1.0, 1.0)))
            
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = opt_pts.copy()
            
    if best_pts is None:
        # Fallback
        idx = np.arange(N) + 0.5
        z = 1.0 - 2.0 * idx / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
        
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
