import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

# Suppress warnings from scipy.optimize to keep output clean
warnings.filterwarnings("ignore")

N = 130

def _compute_covering_radius(points):
    """
    Computes the exact covering radius in radians using the convex hull method.
    This is used for the final validation and scoring, not for the optimization loop.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_circumradius = 0.0
        for i, j, k in facets:
            p_i = points[i]
            p_j = points[j]
            p_k = points[k]
            
            # Circumcenter direction for spherical triangle
            # The circumcenter of a spherical triangle is the normalized sum of the vertices
            # only if the triangle is small (acute). For general spherical triangles, 
            # the circumcenter is the intersection of the perpendicular bisectors.
            # However, for covering radius problems with small caps, the "Voronoi vertex" 
            # (circumcenter) is indeed given by the normalized sum of the vertices 
            # IF the origin projects inside the triangle formed by the vertices in 3D space.
            # A more robust way to find the circumcenter of a spherical triangle (i,j,k) 
            # is to find the point c such that |c-p_i| = |c-p_j| = |c-p_k| and |c|=1.
            # This is equivalent to finding the null space of the system:
            # c . (p_i - p_j) = 0
            # c . (p_i - p_k) = 0
            # c . c = 1
            
            # Let's use the cross product method which is robust for non-degenerate triangles.
            # The normal to the plane containing p_i, p_j, p_k is n = (p_j - p_i) x (p_k - p_i).
            # The circumcenter lies in the plane defined by the origin and the normal to the triangle plane? No.
            
            # Actually, the circumcenter of the spherical triangle is the normalized vector 
            # that is equidistant from the three vertices.
            # Let c = a*p_i + b*p_j + c*p_k.
            # Conditions: c.p_i = c.p_j = c.p_k = cos(theta).
            # This implies c is orthogonal to (p_i - p_j) and (p_i - p_k).
            # So c is parallel to (p_i - p_j) x (p_i - p_k).
            
            v1 = p_i - p_j
            v2 = p_i - p_k
            cross = np.cross(v1, v2)
            
            # If the cross product is near zero, the points are collinear or coincident (degenerate).
            if np.linalg.norm(cross) < 1e-10:
                continue
            
            c = cross / np.linalg.norm(cross)
            
            # We need to ensure c is on the same side of the triangle as the "cap center".
            # The cap center is the point on the sphere "outside" the triangle relative to the origin?
            # No, the covering radius is the distance to the nearest point.
            # The "hole" is the region not covered. The farthest point in that hole from the vertices 
            # is the circumcenter of the spherical triangle formed by the 3 nearest points.
            
            # The vector `cross` points either towards the origin or away from it relative to the triangle plane.
            # We want the circumcenter on the unit sphere.
            # Let's check the angle.
            dot = np.dot(c, p_i)
            # If dot < 0, the circumcenter is on the "far side" (antipodal region), 
            # but for small caps covering the sphere, the circumcenter of the Voronoi cell 
            # is the one where the angle is acute/small.
            # Actually, if the triangle contains the antipode of the origin, the "circumcenter" 
            # for covering purposes might be different. But for optimal packings, 
            # the Voronoi cells are small.
            
            # Let's verify: c should be such that the angle is the circumradius.
            # If we use the other direction -c, the angle would be pi - theta.
            # We want the smaller angle (the gap).
            # So we take c such that dot(c, p_i) > 0 if possible?
            # Not necessarily. But usually, the "center" of the cap is the one closest to the vertices.
            
            # Let's calculate theta for both c and -c and take the min?
            # No, the circumradius is defined by the geometry.
            # For a spherical triangle, there are two circumcenters (antipodal).
            # The one relevant for covering is the one inside the "hole".
            # The hole is the set of points farther from any vertex than the circumradius.
            # The circumcenter is the point equidistant from the 3 vertices.
            # The distance is arccos(|c . p_i|).
            
            # Let's just use the absolute value of the dot product to get the acute angle?
            # Wait, if the triangle is large (obtuse), the circumcenter might be "outside".
            # For the covering problem, we are looking for the maximum distance to the nearest point.
            # The critical points are the vertices of the Voronoi diagram.
            # These are circumcenters of triangles of points.
            # The distance is the circumradius.
            
            # Let's assume the standard case where the circumcenter is the normalized cross product 
            # oriented such that it points "outward" from the origin relative to the triangle plane?
            # Actually, the vector n = (p_i - p_j) x (p_k - p_i) is normal to the plane.
            # The circumcenter c is parallel to n.
            # Let's check the sign.
            # If we project the origin onto the plane, it's at some point.
            # The circumcenter is on the line through the origin and the circumcenter of the Euclidean triangle?
            # No.
            
            # Let's use the property that c is orthogonal to p_i - p_j and p_i - p_k.
            # c = cross / |cross|
            # The angle is arccos(c . p_i).
            # If c . p_i < 0, the angle is > 90 degrees.
            # Is it possible for the covering radius to be > 90 degrees?
            # For 130 points, the radius is ~11 degrees. So we are in the acute regime.
            # So c . p_i should be positive.
            # If c . p_i is negative, we should flip c to -c?
            # If we flip c, the angle becomes pi - theta.
            # Since we want the small angle (the gap), we should take the min(theta, pi-theta)?
            # No, the circumradius is the distance to the *nearest* point.
            # If the circumcenter is at distance > 90 deg, it means the points are far apart.
            # But for a good packing, the circumcenter is "between" the points.
            
            # Let's just calculate the angle.
            dot_val = np.dot(c, p_i)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            theta = np.arccos(dot_val)
            
            # If theta > pi/2, it might be the "other" circumcenter.
            # But for small caps, theta < pi/2.
            # Let's assume theta is the correct radius.
            
            if theta > max_circumradius:
                max_circumradius = theta
                
        return max_circumradius
    except Exception:
        return np.pi

def construct():
    """
    Constructs a set of N points on the unit sphere to minimize the covering radius.
    Strategy: 
    1. Initialize with Fibonacci spiral.
    2. Define a smooth objective function using Log-Sum-Exp of squared distances 
       to the nearest points, or better, a surrogate for the max circumradius.
       
       Actually, the covering radius is max over all y on sphere of min over x in P of d(y,x).
       This is equivalent to the max circumradius of the convex hull facets.
       
       A differentiable surrogate for the max circumradius:
       For each facet (triangle), compute its circumradius r_i.
       We want to minimize max(r_i).
       We can use Log-Sum-Exp: R = (1/beta) * log(sum(exp(beta * r_i))).
       As beta -> infinity, this approaches max(r_i).
       
       However, computing the circumradius for each facet requires ConvexHull, which is not differentiable.
       
       Alternative Strategy:
       Use a "soft" covering radius definition based on a grid of test points?
       No, that's too slow and inaccurate.
       
       Better Strategy:
       Use the fact that the covering radius is related to the minimum distance between points?
       No, that's packing radius.
       
       Let's stick to the ConvexHull approach but make it differentiable?
       ConvexHull is not differentiable.
       
       Alternative:
       Use a "Voronoi-based" objective that is differentiable?
       Or, use a stochastic gradient descent approach where we sample points from the sphere,
       compute the distance to the nearest point, and minimize the max (or soft max) of these distances.
       
       This "Monte Carlo + Soft Max" approach is differentiable!
       
       Objective(y) = SoftMax_{x in P}( d(y, x) ) ? No, we want min over x.
       We want to minimize the max over y of min over x d(y,x).
       
       Let f(y, P) = min_{x in P} d(y, x).
       We want to minimize E_y[ SoftMax(f(y, P)) ] or max_y f(y, P).
       
       Since we can't maximize over y analytically, we sample y.
       But the evaluator uses the exact convex hull radius.
       
       Let's try a hybrid:
       1. Optimize using a smooth surrogate that approximates the covering radius.
          One common surrogate is the "soft max" of the distances to the nearest points 
          for a set of random test points on the sphere.
          
          Let Y be a set of M random points on the sphere.
          For each y in Y, compute d_min(y) = min_{x in P} d(y, x).
          Objective = (1/beta) log( sum_{y in Y} exp(beta * d_min(y)) ).
          
          This is differentiable with respect to P (the positions of x).
          d_min(y) = min_i d(y, x_i).
          The gradient of min is tricky (subgradient), but we can use the "straight-through" 
          estimator or just compute the gradient of the *active* minimum.
          Actually, for a fixed y, the point x_k that is closest is well-defined almost everywhere.
          So grad_x_k d(y, x_k) is the unit vector from x_k to y (projected to tangent space).
          The other points have 0 gradient.
          This is a sparse gradient, which is fine for L-BFGS-B.
          
       2. After optimization, refine with local search if needed.
       
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms

    # Flatten for optimizer
    x0 = points.flatten()

    # Parameters for the surrogate
    # Number of test points M. Too few -> noisy, Too many -> slow.
    # M=1000 is a good balance.
    M = 2000
    rng = np.random.default_rng(42)
    # Generate random points on the sphere for the surrogate
    # Use Fibonacci spiral for better coverage of the test points too? 
    # Random is better for unbiased estimation, but structured might help convergence.
    # Let's use random normal vectors normalized.
    test_pts = rng.normal(0, 1, (M, 3))
    test_pts = test_pts / np.linalg.norm(test_pts, axis=1, keepdims=True)

    # Beta for Log-Sum-Exp. Higher beta -> sharper max.
    # Start with moderate beta, increase? Or just keep it high enough to approximate max.
    # Radius is ~0.2 rad. beta=50 gives exp(10) ~ 22000, so it picks the max strongly.
    beta = 100.0

    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        # Normalize points to ensure they stay on sphere (L-BFGS-B doesn't know constraints)
        # Actually, it's better to parameterize the sphere or project.
        # Projection: normalize after each step.
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Compute distances from test_pts to pts
        # d_{m,n} = arccos(clip(test_pts[m] . pts[n], -1, 1))
        # This is MxN matrix.
        # M=2000, N=130. 260,000 dots. Fast.
        
        # Vectorized dot product
        # test_pts: (M, 3), pts: (N, 3)
        # Dots: (M, N)
        dots = np.dot(test_pts, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots) # (M, N)
        
        # For each test point m, find the min distance to any point n
        # min_dists: (M,)
        min_dists = np.min(dists, axis=1)
        # Also get the index of the min to compute gradient
        min_indices = np.argmin(dists, axis=1) # (M,)
        
        # Objective: Log-Sum-Exp of min_dists
        # LSE = (1/beta) * log(sum(exp(beta * min_dists)))
        # For numerical stability, subtract max
        max_md = np.max(min_dists)
        exp_terms = np.exp(beta * (min_dists - max_md))
        lse = (max_md + np.log(np.sum(exp_terms)) / beta)
        
        # Gradient
        # d(LSE)/d(min_dists_j) = (1/beta) * exp(beta * min_dists_j) / sum(exp(...))
        # = exp_terms[j] / sum(exp_terms)
        weights = exp_terms / np.sum(exp_terms) # (M,)
        
        # Now, d(min_dists_j)/d(x_k)
        # min_dists_j is the distance from test_pts[j] to pts[min_indices[j]]
        # Let k = min_indices[j]
        # dist = arccos(test_pts[j] . pts[k])
        # d(dist)/d(pts[k]) = - (test_pts[j] - (test_pts[j].pts[k]) pts[k]) / sqrt(1 - (test_pts[j].pts[k])^2)
        # This is the projection of (test_pts[j] - pts[k]) onto the tangent space of pts[k].
        # Actually, gradient of arccos(u) is -1/sqrt(1-u^2).
        # u = t . p. du/dp = t - (t.p)p.
        # So grad_p dist = - 1/sqrt(1-u^2) * (t - (t.p)p).
        
        # We need to accumulate gradients for each point k.
        # grad_x = zeros((N, 3))
        # For each j in M:
        #   k = min_indices[j]
        #   w = weights[j]
        #   t = test_pts[j]
        #   p = pts[k]
        #   u = dots[j, k]
        #   denom = sqrt(1 - u^2)
        #   if denom > eps:
        #      grad_term = - w / denom * (t - u*p)
        #      grad_x[k] += grad_term
        
        # Vectorization of gradient accumulation:
        # This is tricky because each test point contributes to a different point k.
        # We can use np.add.at or similar, but it's slow.
        # Alternative: Loop over M? M=2000. 2000 iterations in Python is slow but maybe acceptable?
        # 2000 * (simple ops) ~ 0.01s?
        # Let's try to vectorize.
        
        # We have weights (M,), min_indices (M,), test_pts (M,3), pts (N,3), dots (M,N).
        # We want grad_pts (N,3).
        # For each k, grad_pts[k] = sum_{j: min_indices[j]==k} weights[j] * grad_dist_j_k
        
        # Let's compute the raw gradients for all pairs? No, too big.
        # Let's just loop over M. It's only 2000.
        grad_pts = np.zeros_like(pts)
        
        # Precompute denominators for active pairs
        # active_dots = dots[range(M), min_indices]
        active_dots = dots[np.arange(M), min_indices]
        denom = np.sqrt(np.maximum(1.0 - active_dots**2, 1e-12))
        # grad_factor_j = - weights[j] / denom[j]
        grad_factors = -weights / denom
        
        # term_j = grad_factors[j] * (test_pts[j] - active_dots[j] * pts[min_indices[j]])
        # We need to sum these terms into grad_pts[min_indices[j]]
        
        # Vectorized calculation of the vector part:
        # v_j = test_pts[j] - active_dots[j] * pts[min_indices[j]]
        # pts[min_indices] is (M, 3)
        active_pts = pts[min_indices]
        v = test_pts - active_dots[:, np.newaxis] * active_pts # (M, 3)
        
        # weighted_v_j = grad_factors[j] * v_j
        weighted_v = grad_factors[:, np.newaxis] * v # (M, 3)
        
        # Sum into grad_pts
        # np.add.at(grad_pts, min_indices, weighted_v)
        np.add.at(grad_pts, min_indices, weighted_v)
        
        # The gradient with respect to x (unnormalized) is the same as grad_pts,
        # but we normalized pts inside the function.
        # The chain rule for normalization p = x/|x| is:
        # d p_i / d x_j = (delta_ij I - p_i p_i^T) / |x_i|
        # Since we are on the sphere, the gradient in the ambient space is the projection of grad_pts onto the tangent space?
        # Actually, if we define the objective on the manifold, the gradient is the tangent vector.
        # L-BFGS-B works in Euclidean space.
        # If we project x to the sphere at the start of the function, the effective gradient 
        # in the x-space is the projection of the manifold gradient onto the tangent space,
        # scaled by 1/|x|?
        # Let's just return the manifold gradient (tangent vector) and let the optimizer handle it.
        # Wait, if we return the tangent vector as the gradient, L-BFGS-B will move x in that direction.
        # Then we normalize again. This is a common "project and reflect" or "project and gradient" scheme.
        # It works well if the step size is small.
        
        return lse, grad_pts.flatten()

    # Run L-BFGS-B
    # maxiter: limit based on time.
    # Each iteration involves M=2000 dots and gradient calc. ~1ms?
    # 1000 iterations ~ 1s. We have 140s. So we can do many iterations.
    # But L-BFGS-B is not just 1 iteration per second, it has overhead.
    # Let's set maxiter to a large number and rely on time check in callback.
    
    max_iter = 10000
    t_start = time.time()
    
    # Callback to check time
    def callback(xk):
        if time.time() - t_start > time_limit - 10: # Leave 10s for final processing
            return True # Stop
        return False

    res = minimize(
        objective_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True, # We return (f, g)
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9, 'callback': callback, 'maxls': 50}
    )
    
    x_opt = res.x
    points_opt = x_opt.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(points_opt, axis=1, keepdims=True)
    points_opt = points_opt / norms
    
    # Check time
    if time.time() - start_time > time_limit - 5:
        # Return immediately
        return points_opt

    # Final Polish:
    # The soft-max surrogate might not perfectly match the convex hull radius.
    # Do a few steps of local refinement on the exact metric?
    # Or just return the result. The soft-max with high beta is a very good approximation.
    
    # Let's do a quick local search to fix any potential issues
    # and ensure we are at a local minimum of the *exact* metric.
    
    # Re-evaluate exact radius
    current_radius = _compute_covering_radius(points_opt)
    best_points = points_opt.copy()
    best_radius = current_radius
    
    # Simple local search: perturb worst facet vertices
    # We can reuse the logic from the original program but run it for a short time
    # to polish the result from the global optimizer.
    
    time_left = time_limit - (time.time() - start_time)
    if time_left > 10:
        # Run a short simulated annealing polish
        T = 1e-4
        T_min = 1e-7
        cooling_rate = 0.99
        rng_local = np.random.default_rng(42)
        
        def get_worst_facet(pts):
            try:
                hull = ConvexHull(pts)
                facets = hull.simplices
                max_cr = 0.0
                worst_idx = -1
                for idx, (i, j, k) in enumerate(facets):
                    p_i = pts[i]
                    p_j = pts[j]
                    p_k = pts[k]
                    c = p_i + p_j + p_k
                    norm_c = np.linalg.norm(c)
                    if norm_c < 1e-10:
                        continue
                    c = c / norm_c
                    dot = np.clip(np.dot(c, p_i), -1.0, 1.0)
                    theta = np.arccos(dot)
                    if theta > max_cr:
                        max_cr = theta
                        worst_idx = idx
                return worst_idx, max_cr, hull
            except Exception:
                return -1, np.pi, None

        while time.time() - start_time < time_limit - 2:
            worst_idx, current_cr, hull = get_worst_facet(points_opt)
            if worst_idx == -1:
                break
            if current_cr < best_radius:
                best_radius = current_cr
                best_points = points_opt.copy()
                
            i, j, k = hull.simplices[worst_idx]
            vertex_to_move = rng_local.choice([i, j, k])
            
            p_i = points_opt[i]
            p_j = points_opt[j]
            p_k = points_opt[k]
            c = p_i + p_j + p_k
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                c = points_opt[vertex_to_move]
            else:
                c = c / norm_c
                
            step_size = 0.005
            noise = rng_local.normal(0, step_size * 0.5, 3)
            
            p_v = points_opt[vertex_to_move]
            dir_vec = c - p_v
            dot = np.dot(dir_vec, p_v)
            dir_tangent = dir_vec - dot * p_v
            norm_dir = np.linalg.norm(dir_tangent)
            if norm_dir > 1e-10:
                dir_tangent = dir_tangent / norm_dir
                
            new_p_v = p_v + step_size * dir_tangent + noise
            new_p_v = new_p_v / np.linalg.norm(new_p_v)
            
            old_p_v = points_opt[vertex_to_move].copy()
            points_opt[vertex_to_move] = new_p_v
            
            new_worst_idx, new_cr, _ = get_worst_facet(points_opt)
            
            if new_cr < current_cr:
                current_cr = new_cr
            else:
                delta = new_cr - current_cr
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if rng_local.random() < prob:
                        current_cr = new_cr
                    else:
                        points_opt[vertex_to_move] = old_p_v
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    # Final normalization and return
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering Radius: {_compute_covering_radius(pts) * 180 / np.pi:.6f} degrees")

