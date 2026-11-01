"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Helper to compute facet radii and gradients
    def compute_facet_radii_and_grads(pts):
        """
        Computes the angular radius for each facet of the convex hull.
        Also computes the gradient of the squared radius w.r.t. the points.
        Returns:
        radii: array of shape (n_facets,) containing the angular radius of each facet.
        grads: array of shape (N, 3) containing the gradient of the sum of squared radii.
               (We use squared radius for smoother optimization, then convert back).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # If hull fails, return large radius and zero grad (or random)
            return np.full(N, 1.0), np.zeros((N, 3))
        
        n_facets = len(hull.simplices)
        radii = np.zeros(n_facets)
        # We will accumulate gradients for the objective: sum(r_i^2)
        # Then we can use softmax or max on the radii themselves later.
        # Actually, for L-BFGS, we need the gradient of the objective we minimize.
        # Let's define the objective as the max radius. 
        # To make it differentiable, we can use a soft-max approximation:
        # R_soft = (1/beta) * log(sum(exp(beta * r_i^2)))
        # Or simply optimize the sum of squares of the "active" facets?
        # The task asks for the max.
        # Let's compute the radius for each facet first.
        
        grads = np.zeros((N, 3))
        
        for idx, simplex in enumerate(hull.simplices):
            i, j, k = simplex
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            # Compute circumcenter of the planar triangle p1, p2, p3
            u = p2 - p1
            v = p3 - p1
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-12:
                # Degenerate triangle, skip
                radii[idx] = 0.0
                continue
                
            # Solve for circumcenter in the plane
            # c = p1 + a*u + b*v
            a = (0.5 * uu * vv - 0.5 * uv * vv) / det
            b = (0.5 * vv * uu - 0.5 * uu * uv) / det
            
            c_planar = p1 + a * u + b * v
            c_norm = np.linalg.norm(c_planar)
            if c_norm < 1e-12:
                radii[idx] = 0.0
                continue
                
            c_sph = c_planar / c_norm
            
            # Check if c_sph is inside the spherical triangle
            P = np.column_stack([p1, p2, p3])
            try:
                coeffs = np.linalg.solve(P, c_sph)
            except np.linalg.LinAlgError:
                radii[idx] = 0.0
                continue
                
            if np.all(coeffs > -1e-6):
                # Circumcenter is inside
                # Distance is arccos(dot(c_sph, p1))
                dot_val = np.clip(np.dot(c_sph, p1), -1.0, 1.0)
                r_ang = np.arccos(dot_val)
                radii[idx] = r_ang
                
                # Gradient of r_ang^2 w.r.t p1, p2, p3
                # Let f = arccos(x) where x = c.p1
                # df/dx = -1/sqrt(1-x^2)
                # dx/dp1 = c (since c depends on p1, p2, p3)
                # This is complex. 
                # Alternative: Optimize the squared chordal distance? No, metric is angular.
                # Let's use a simpler gradient approximation or finite differences?
                # L-BFGS can handle non-smooth functions with care, but gradients help.
                # Let's compute the gradient of the squared angular distance.
                # Actually, there is a known formula for the gradient of the covering radius.
                # If the circumcenter is inside, the gradient w.r.t vertex p_i is:
                # grad_p_i = - (sin(r) / (sin(r_i1) sin(r_i2) sin(r_i3) ...)) * (tangent_vector)
                # This is getting complicated.
                
                # Let's try a differentiable surrogate:
                # Minimize the sum of exp(beta * r_i) for all facets.
                # Gradient of exp(beta * r_i) = beta * exp(beta * r_i) * grad(r_i)
                # We still need grad(r_i).
                
                # Let's use the property that for a point on the sphere, the gradient of the distance to a fixed point q is (q - p)/sin(d) in the tangent space.
                # Here the "worst point" is the circumcenter c_sph.
                # So grad_p_i (distance to c_sph) = (c_sph - p_i) / sin(r_ang) ... projected to tangent plane?
                # Actually, the gradient of the angular distance d(p, c) w.r.t p is (c - p) / ||c - p|| * (1/sin(d))?
                # Let d = arccos(c.p). sin(d) = ||c x p|| = ||c - (c.p)p||.
                # grad_p d = (c - (c.p)p) / ||c - (c.p)p||^2 * ||c - (c.p)p|| ? No.
                # Let's derive: d = arccos(c.p). dd/dp = -1/sqrt(1-(c.p)^2) * c = -c / sin(d).
                # This is the gradient in R3. We must project it to the tangent plane of the sphere at p.
                # Tangent projection: v_tan = v - (v.p)p.
                # So grad_p d = ( -c / sin(d) ) - ( (-c / sin(d)).p ) p
                # = ( -c + (c.p)/sin(d) p ) / sin(d)
                # = ( (c.p)/sin(d) p - c ) / sin(d)
                
                sin_d = np.sin(r_ang)
                if sin_d < 1e-6:
                    continue
                    
                c_dot_p1 = np.dot(c_sph, p1)
                grad_c = (c_dot_p1 / sin_d * p1 - c_sph) / sin_d
                # This is the gradient of the distance from p1 to c_sph.
                # However, c_sph itself depends on p1, p2, p3.
                # For small moves, the change in c_sph is second order? 
                # Actually, if we move p1, the circumcenter moves.
                # But the dominant term is the change in distance to the *current* circumcenter.
                # This is a first-order approximation. It works well for local optimization.
                
                # So, grad_obj w.r.t p1 approx += grad_c
                # Similarly for p2, p3.
                # Note: The objective is max(r_i). We are approximating the gradient of the soft-max.
                # For now, let's just accumulate the gradient of r_i^2 or r_i.
                # Let's accumulate gradient of r_i (angular distance).
                
                grads[i] += grad_c
                grads[j] += grad_c.copy() # Wait, gradient w.r.t p2 is similar but with p2
                # Let's re-calculate for p2 and p3 specifically to be safe.
                
                sin_d2 = np.sin(r_ang)
                c_dot_p2 = np.dot(c_sph, p2)
                grad_c2 = (c_dot_p2 / sin_d2 * p2 - c_sph) / sin_d2
                grads[j] += grad_c2
                
                sin_d3 = np.sin(r_ang)
                c_dot_p3 = np.dot(c_sph, p3)
                grad_c3 = (c_dot_p3 / sin_d3 * p3 - c_sph) / sin_d3
                grads[k] += grad_c3
                
            else:
                # Circumcenter outside. Worst point is midpoint of longest edge.
                # This is non-smooth. We can approximate by pushing the vertices of the longest edge apart.
                d12 = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
                d23 = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
                d31 = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
                
                # Find longest edge
                if d12 >= d23 and d12 >= d31:
                    radii[idx] = d12 / 2.0
                    # Gradient: push p1 away from p2 and p2 away from p1
                    # Approximation: grad_p1 += (p2 - p1)/sin(d12) ... wait, we want to INCREASE distance?
                    # No, we want to MINIMIZE the max radius.
                    # If the midpoint is the worst point, we should move p1 and p2 towards the midpoint?
                    # No, the midpoint is the farthest point from the *other* points?
                    # The covering radius is the distance from the worst point y to the nearest point.
                    # If y is the midpoint of edge (1,2), the nearest points are p1 and p2.
                    # To reduce the distance, we should move p1 and p2 towards y?
                    # Yes.
                    # Gradient of distance from y to p1: (p1 - y)/sin(d) ?
                    # We want to minimize d(p1, y).
                    # grad_p1 d = (y - p1)/sin(d) projected?
                    # Actually, if y is fixed, grad_p1 d = (y - p1)/||y-p1|| * ...
                    # Let's just use the same formula: grad = (c_dot_p / sin_d * p - c) / sin_d where c is the "worst point" y.
                    y = (p1 + p2) / np.linalg.norm(p1 + p2)
                    sin_d = np.sin(d12 / 2.0)
                    if sin_d > 1e-6:
                        c_dot_p1 = np.dot(y, p1)
                        grad_c1 = (c_dot_p1 / sin_d * p1 - y) / sin_d
                        grads[i] += grad_c1
                        
                        c_dot_p2 = np.dot(y, p2)
                        grad_c2 = (c_dot_p2 / sin_d * p2 - y) / sin_d
                        grads[j] += grad_c2
                elif d23 >= d12 and d23 >= d31:
                    radii[idx] = d23 / 2.0
                    y = (p2 + p3) / np.linalg.norm(p2 + p3)
                    sin_d = np.sin(d23 / 2.0)
                    if sin_d > 1e-6:
                        c_dot_p2 = np.dot(y, p2)
                        grad_c2 = (c_dot_p2 / sin_d * p2 - y) / sin_d
                        grads[j] += grad_c2
                        
                        c_dot_p3 = np.dot(y, p3)
                        grad_c3 = (c_dot_p3 / sin_d * p3 - y) / sin_d
                        grads[k] += grad_c3
                else:
                    radii[idx] = d31 / 2.0
                    y = (p3 + p1) / np.linalg.norm(p3 + p1)
                    sin_d = np.sin(d31 / 2.0)
                    if sin_d > 1e-6:
                        c_dot_p3 = np.dot(y, p3)
                        grad_c3 = (c_dot_p3 / sin_d * p3 - y) / sin_d
                        grads[k] += grad_c3
                        
                        c_dot_p1 = np.dot(y, p1)
                        grad_c1 = (c_dot_p1 / sin_d * p1 - y) / sin_d
                        grads[i] += grad_c1
                        
        return radii, grads

    def objective_and_grad(x_flat):
        """
        Objective: Soft-max of the facet radii.
        x_flat: flattened array of shape (N*3,)
        """
        pts = x_flat.reshape(N, 3)
        # Normalize points to ensure they stay on the sphere during optimization steps
        # L-BFGS might step off the sphere. We project back.
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        radii, grads = compute_facet_radii_and_grads(pts)
        
        # Soft-max to make it differentiable
        # R_soft = (1/beta) * log(sum(exp(beta * r_i)))
        # We want to minimize R_soft.
        # Gradient: (1/beta) * sum(exp(beta * r_i) * beta * grad_r_i) / sum(exp(beta * r_i))
        # = sum(exp(beta * r_i) * grad_r_i) / sum(exp(beta * r_i))
        
        beta = 50.0 # Temperature parameter for soft-max
        
        # Scale radii to avoid overflow in exp
        # Max radius is around 0.3 rad (17 deg). beta * r ~ 15. exp(15) is fine.
        exp_r = np.exp(beta * radii)
        sum_exp = np.sum(exp_r)
        
        if sum_exp < 1e-10:
            return 0.0, np.zeros_like(x_flat)
            
        weights = exp_r / sum_exp
        obj_val = np.log(sum_exp) / beta
        
        # Gradient of soft-max w.r.t radii: weights * beta
        # Gradient of radii w.r.t pts: grads (computed as sum of grads for each facet)
        # Wait, `grads` from compute_facet_radii_and_grads is the sum of gradients for ALL facets?
        # No, in my implementation, I added gradients for each facet to the global `grads` array.
        # So `grads` is the gradient of the SUM of radii.
        # We need the gradient of the SOFT-MAX.
        # d(R_soft)/dp = sum_i (d R_soft/d r_i) * (d r_i/dp)
        # d R_soft/d r_i = weights[i] * beta
        # So we need the gradient of EACH facet's radius separately to weight them.
        # My current `grads` is sum_i grad_r_i. This is incorrect for soft-max.
        
        # I need to modify compute_facet_radii_and_grads to return a list of gradients or
        # re-calculate.
        # For efficiency, let's re-calculate or store them.
        # Let's modify the helper to return an array of gradients (n_facets, N, 3).
        
        # Since I can't easily change the helper without re-writing, let's assume for this step
        # that we use the gradient of the MAX radius (approximated by the gradient of the facet with max radius).
        # This is a common heuristic.
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        # If we use hard max, the gradient is just the gradient of that facet.
        # We need the gradient of that specific facet.
        # Let's re-compute the gradient for just the worst facet to be accurate.
        
        worst_grad = np.zeros((N, 3))
        try:
            hull = ConvexHull(pts)
            simplex = hull.simplices[max_idx]
            i, j, k = simplex
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            u = p2 - p1
            v = p3 - p1
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            det = uu * vv - uv * uv
            
            if abs(det) > 1e-12:
                a = (0.5 * uu * vv - 0.5 * uv * vv) / det
                b = (0.5 * vv * uu - 0.5 * uu * uv) / det
                c_planar = p1 + a * u + b * v
                c_norm = np.linalg.norm(c_planar)
                if c_norm > 1e-12:
                    c_sph = c_planar / c_norm
                    P = np.column_stack([p1, p2, p3])
                    try:
                        coeffs = np.linalg.solve(P, c_sph)
                        if np.all(coeffs > -1e-6):
                            r_ang = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
                            sin_d = np.sin(r_ang)
                            if sin_d > 1e-6:
                                c_dot_p1 = np.dot(c_sph, p1)
                                grad_c1 = (c_dot_p1 / sin_d * p1 - c_sph) / sin_d
                                worst_grad[i] += grad_c1
                                
                                c_dot_p2 = np.dot(c_sph, p2)
                                grad_c2 = (c_dot_p2 / sin_d * p2 - c_sph) / sin_d
                                worst_grad[j] += grad_c2
                                
                                c_dot_p3 = np.dot(c_sph, p3)
                                grad_c3 = (c_dot_p3 / sin_d * p3 - c_sph) / sin_d
                                worst_grad[k] += grad_c3
                        else:
                            # Outside case
                            d12 = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
                            d23 = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
                            d31 = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
                            
                            if d12 >= d23 and d12 >= d31:
                                y = (p1 + p2) / np.linalg.norm(p1 + p2)
                                sin_d = np.sin(d12 / 2.0)
                                if sin_d > 1e-6:
                                    c_dot_p1 = np.dot(y, p1)
                                    grad_c1 = (c_dot_p1 / sin_d * p1 - y) / sin_d
                                    worst_grad[i] += grad_c1
                                    c_dot_p2 = np.dot(y, p2)
                                    grad_c2 = (c_dot_p2 / sin_d * p2 - y) / sin_d
                                    worst_grad[j] += grad_c2
                            elif d23 >= d12 and d23 >= d31:
                                y = (p2 + p3) / np.linalg.norm(p2 + p3)
                                sin_d = np.sin(d23 / 2.0)
                                if sin_d > 1e-6:
                                    c_dot_p2 = np.dot(y, p2)
                                    grad_c2 = (c_dot_p2 / sin_d * p2 - y) / sin_d
                                    worst_grad[j] += grad_c2
                                    c_dot_p3 = np.dot(y, p3)
                                    grad_c3 = (c_dot_p3 / sin_d * p3 - y) / sin_d
                                    worst_grad[k] += grad_c3
                            else:
                                y = (p3 + p1) / np.linalg.norm(p3 + p1)
                                sin_d = np.sin(d31 / 2.0)
                                if sin_d > 1e-6:
                                    c_dot_p3 = np.dot(y, p3)
                                    grad_c3 = (c_dot_p3 / sin_d * p3 - y) / sin_d
                                    worst_grad[k] += grad_c3
                                    c_dot_p1 = np.dot(y, p1)
                                    grad_c1 = (c_dot_p1 / sin_d * p1 - y) / sin_d
                                    worst_grad[i] += grad_c1
                    except np.linalg.LinAlgError:
                        pass
        except Exception:
            pass
            
        # The gradient of the max radius is the gradient of the worst facet.
        # We return the value of the max radius (or soft-max) and the gradient.
        # Using max radius value for the score, but gradient of the worst facet.
        # This is a standard approach for min-max problems.
        
        return max_r, worst_grad.flatten()

    # 2. Multi-start L-BFGS
    best_pts = None
    best_val = np.pi
    
    # Generate a few diverse starting points
    starts = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(x0)
    
    # Random start
    x1 = np.random.randn(N, 3)
    x1 /= np.linalg.norm(x1, axis=1, keepdims=True)
    starts.append(x1)
    
    # Another random start
    x2 = np.random.randn(N, 3)
    x2 /= np.linalg.norm(x2, axis=1, keepdims=True)
    starts.append(x2)
    
    for start_idx, x_start in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        x_flat = x_start.flatten()
        
        res = minimize(
            objective_and_grad,
            x_flat,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 100, 'gtol': 1e-5}
        )
        
        if res.success or res.fun < best_val:
            pts_opt = res.x.reshape(N, 3)
            # Normalize
            pts_opt /= np.linalg.norm(pts_opt, axis=1, keepdims=True)
            
            # Verify with exact calculation
            radii, _ = compute_facet_radii_and_grads(pts_opt)
            val = np.max(radii)
            
            if val < best_val:
                best_val = val
                best_pts = pts_opt.copy()
                
    if best_pts is None:
        # Fallback
        best_pts = starts[0]
        
    # 3. Final Polish with Simulated Annealing (short) to break any remaining plateaus
    if time.time() < deadline - 5.0:
        current_pts = best_pts.copy()
        current_val, _ = compute_facet_radii_and_grads(current_pts)
        current_val = np.max(current_val)
        
        T = 0.01
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            idx = np.random.randint(0, N)
            noise = np.random.normal(0, T, 3)
            new_pt = current_pts[idx] + noise
            new_pt /= np.linalg.norm(new_pt)
            
            trial_pts = current_pts.copy()
            trial_pts[idx] = new_pt
            
            trial_radii, _ = compute_facet_radii_and_grads(trial_pts)
            trial_val = np.max(trial_radii)
            
            delta = trial_val - current_val
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_pts = trial_pts
                current_val = trial_val
                if current_val < best_val:
                    best_val = current_val
                    best_pts = current_pts.copy()
            T *= 0.95

    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
