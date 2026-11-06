"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius_and_worst_indices(pts):
        """
        Computes the exact covering radius and identifies the facet responsible for it.
        Returns (radius_degrees, worst_face_indices, circumcenter).
        """
        try:
            hull = ConvexHull(pts)
            max_r_rad = 0.0
            worst_face = None
            worst_center = None
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                v1 = p0 - p1
                v2 = p0 - p2
                c_vec = np.cross(v1, v2)
                
                if np.linalg.norm(c_vec) < 1e-10:
                    continue
                    
                c_vec /= np.linalg.norm(c_vec)
                
                # Determine correct hemisphere for circumcenter
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                
                if np.dot(c_vec, centroid) < 0:
                    c_vec = -c_vec
                    
                # Check if circumcenter is inside the spherical triangle
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                
                # Orient normals towards centroid
                if np.dot(centroid, n1) < 0: n1 = -n1
                if np.dot(centroid, n2) < 0: n2 = -n2
                if np.dot(centroid, n3) < 0: n3 = -n3
                
                if (np.dot(c_vec, n1) >= -1e-8 and 
                    np.dot(c_vec, n2) >= -1e-8 and 
                    np.dot(c_vec, n3) >= -1e-8):
                    
                    dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                    if dist > max_r_rad:
                        max_r_rad = dist
                        worst_face = face
                        worst_center = c_vec
                        
            return np.degrees(max_r_rad), worst_face, worst_center
        except Exception:
            return 180.0, None, None

    def soft_max_covering_energy(pts, temperature=0.1):
        """
        Computes a smooth approximation of the covering radius using a soft-max 
        over the distances to the nearest points.
        We use a dense grid of points to sample the sphere.
        """
        # Use a fixed grid for evaluation to ensure differentiability and speed
        # 20000 points is a good balance between accuracy and speed for gradient estimation
        n_grid = 20000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid_pts = normalize(grid_pts)
        
        # Distance matrix: (n_grid, N)
        # d[i, j] = angle between grid_pts[i] and pts[j]
        # cos(d) = dot(grid_pts[i], pts[j])
        dots = grid_pts @ pts.T
        
        # We want to minimize the max distance.
        # Let u = 1 - cos(d) = 1 - dot. Since d is small, u is proportional to d^2.
        # Minimizing max(d) is similar to minimizing max(u) for small angles.
        # To make it smooth and differentiable, we use a soft-max on the angles.
        # However, arccos is non-differentiable at 1. 
        # Better approach: Minimize the soft-max of the squared chordal distances or similar.
        # Let's use the soft-max of the angles directly, but careful with gradients.
        # Actually, a common robust approach is to minimize the sum of exp(k * dist) or similar.
        # Or simply: SoftMax of the distances.
        
        # To avoid arccos issues, we can work with cosines.
        # Max distance <-> Min cosine.
        # So we want to maximize the minimum cosine.
        # MinCos = min_j (dots[i, j]) for each i.
        # We want to maximize min_i (MinCos_i).
        # This is equivalent to minimizing -min_i (MinCos_i).
        # Let's use a soft-min on the cosines to make it smooth.
        
        # min_j (dots[i, j]) approximated by -1/k * log(sum_j exp(-k * dots[i, j]))
        # We want to maximize this value over i.
        # So we minimize - (soft_min_cosines).
        
        k = 100.0 # Sharpness of the soft-min
        
        # Compute soft-min of cosines for each grid point
        # dots is (n_grid, N)
        # We want min over axis 1.
        # log-sum-exp trick for stability
        # -1/k * log(sum(exp(-k * dots)))
        
        # Scale to avoid overflow/underflow
        # Since dots are in [-1, 1], -k*dots is in [-k, k].
        # For k=100, exp(-100) is very small, exp(100) is huge.
        # We can just use np.min for the exact value if we don't need gradients from the grid,
        # but we DO need gradients.
        
        # Let's compute the gradient of the soft-min explicitly or use autograd-like logic?
        # No, we need to return a value and potentially a gradient for L-BFGS-B.
        # L-BFGS-B can use numerical gradients if we don't provide them, but that's slow.
        # Let's try providing the gradient.
        
        # Objective: F(pts) = - (1/n_grid) * sum_i [ -1/k * log(sum_j exp(-k * dot(g_i, p_j))) ]
        #           = (1/n_grid) * sum_i [ 1/k * log(sum_j exp(-k * dot(g_i, p_j))) ]
        
        # Let S_i = sum_j exp(-k * dot(g_i, p_j))
        # F = (1/(k*n_grid)) * sum_i log(S_i)
        
        # Gradient with respect to p_j:
        # dF/dp_j = (1/(k*n_grid)) * sum_i (1/S_i) * dS_i/dp_j
        # dS_i/dp_j = exp(-k * dot(g_i, p_j)) * (-k * g_i)
        # dF/dp_j = (1/(k*n_grid)) * sum_i (1/S_i) * exp(-k * dot(g_i, p_j)) * (-k * g_i)
        #          = -(1/n_grid) * sum_i [ exp(-k * dot(g_i, p_j)) / S_i * g_i ]
        
        # Let w_i = exp(-k * dot(g_i, p_j)) / S_i. Note that w_i depends on j.
        # This formulation is tricky because the weight for each grid point i depends on which point p_j we are differentiating.
        
        # Alternative: Minimize the soft-max of the DISTANCES directly?
        # Let's stick to the cosine soft-min. It's a standard technique for "max-min" problems.
        
        # To speed up gradient calculation, we can compute the weights for all points at once.
        # weights[i, j] = exp(-k * dots[i, j]) / S_i
        # Then grad[j] = - (1/n_grid) * sum_i weights[i, j] * g_i
        
        # However, computing the full (n_grid, N) weight matrix might be memory intensive if n_grid is large.
        # 20000 * 50 = 1M floats. That's 8MB. Acceptable.
        
        # Compute S_i
        # To avoid overflow in exp, we can subtract the max of -k*dots for each row?
        # Actually, dots are <= 1. -k*dots >= -k.
        # If dots is close to -1, -k*dots is close to k. exp(k) is huge.
        # Let's shift: let M_i = max_j (-k * dots[i, j]) = -k * min_j(dots[i, j])
        # Then exp(-k*dots - M_i) is bounded by 1.
        # S_i = exp(M_i) * sum_j exp(-k*dots[i,j] - M_i)
        # log(S_i) = M_i + log(sum_j exp(...))
        
        # Let's implement the objective and gradient.
        
        # 1. Compute dots (n_grid, N)
        # 2. Compute -k * dots
        neg_k_dots = -k * dots
        
        # 3. For each row i, find max value M_i for stability
        M_i = np.max(neg_k_dots, axis=1, keepdims=True)
        
        # 4. Compute stable exponentials
        # A[i, j] = exp(neg_k_dots[i, j] - M_i)
        A = np.exp(neg_k_dots - M_i)
        
        # 5. Sum over j to get sum_exp
        sum_exp = np.sum(A, axis=1, keepdims=True)
        
        # 6. Objective value
        # log_S_i = M_i + log(sum_exp)
        # F = (1/(k*n_grid)) * sum_i log_S_i
        log_S_i = M_i.squeeze() + np.log(sum_exp.squeeze())
        obj_val = np.mean(log_S_i) / k
        
        # 7. Gradient
        # weights W[i, j] = A[i, j] / sum_exp[i]
        W = A / sum_exp
        
        # grad[j] = -(1/n_grid) * sum_i W[i, j] * g_i
        # This is equivalent to: - (1/n_grid) * (W.T @ grid_pts)
        grad = -(W.T @ grid_pts) / n_grid
        
        # The gradient is in Cartesian coordinates.
        # We are optimizing on the sphere. The gradient of the objective on the sphere
        # is the projection of the Cartesian gradient onto the tangent plane.
        # proj_grad = grad - (grad . p) * p
        
        # We need to return the gradient in the parameter space.
        # If we use L-BFGS-B on the 3D coordinates with a constraint ||p||=1,
        # it's tricky. It's better to parameterize the sphere or use the tangent space gradient.
        # However, scipy.optimize.minimize doesn't handle non-linear constraints well for gradient methods unless we use SLSQP or trust-constr.
        # L-BFGS-B only handles bound constraints.
        
        # Strategy:
        # 1. Use SLSQP with equality constraint ||p_i||^2 - 1 = 0.
        # 2. Or, parameterize using quaternions or axis-angle? Too complex.
        # 3. Or, simply use L-BFGS-B on the 3D coords, and after each step, re-normalize the points.
        #    This is a "projected gradient" approach, but L-BFGS-B assumes the gradient is for the unconstrained problem.
        #    If we pass the projected gradient, it might work reasonably well if we re-normalize.
        #    Let's try passing the Cartesian gradient and letting the optimizer do its thing, 
        #    but we must ensure the points stay on the sphere.
        
        # Actually, a common trick is to optimize in 3D and normalize at the end of each function call?
        # No, the optimizer keeps internal state.
        
        # Better: Use `scipy.optimize.minimize` with `method='L-BFGS-B'` on the 3D coordinates,
        # but define the function to normalize the input internally before computing the energy?
        # If we do that, the gradient returned must be consistent with the normalization.
        
        # Let's try a different approach: Optimize on the sphere using a parameterization that covers the sphere without singularities?
        # Or just use SLSQP. It's slower but handles constraints.
        # Given N=50, 150 variables. SLSQP might be too slow for 120s if it takes many iterations.
        
        # Let's stick to L-BFGS-B with a "normalize inside" strategy.
        # The function `soft_max_covering_energy` will normalize `pts` first.
        # The gradient must be the gradient of `f(normalize(pts))`.
        # df/dpts = (df/dnorm_pts) * J_norm
        # J_norm is the Jacobian of the normalization.
        # This is complex.
        
        # Simpler: Just use numerical gradients with L-BFGS-B?
        # 150 variables. Numerical gradient requires 150 evaluations.
        # Each evaluation is ~0.1s. 150 * 0.1 = 15s per gradient step.
        # Too slow for many iterations.
        
        # Let's use the analytic gradient but handle the sphere constraint by projecting the gradient.
        # And use a custom wrapper that re-normalizes the points after each L-BFGS-B step?
        # L-BFGS-B doesn't support callbacks that modify the state directly in the same way.
        
        # Alternative: Use `scipy.optimize.minimize` with `method='BFGS'` and line search?
        
        # Let's try a simpler, more robust method:
        # 1. Initialize with Fibonacci spiral.
        # 2. Use L-BFGS-B to minimize the energy.
        # 3. In the function, normalize the points.
        # 4. Return the energy and the gradient projected onto the tangent space, but mapped back to Cartesian?
        
        # Actually, if we optimize x in R^3 and the function is f(x/||x||), the gradient is:
        # grad_x f = (1/||x||) * (grad_u f - (grad_u f . u) * u) where u = x/||x||
        # This is the gradient in the ambient space.
        # If we pass this to L-BFGS-B, it will take a step in this direction.
        # The step will move x off the sphere.
        # But since the function normalizes x internally, the "effective" position is always on the sphere.
        # This effectively performs gradient descent on the sphere, but the optimizer's internal model
        # might be confused because the function is not smooth in R^3 at the origin (but we avoid origin).
        
        # Let's implement this:
        # 1. Function takes x (150,). Reshapes to (50, 3).
        # 2. Normalizes to get u.
        # 3. Computes energy E(u).
        # 4. Computes grad_u E.
        # 5. Projects grad_u to tangent space: g_tan = grad_u - (grad_u . u) * u.
        # 6. Maps back to ambient gradient: grad_x = g_tan / ||x||. (Approximation, since ||x|| changes).
        #    Actually, if we assume x is close to u, ||x|| ~ 1.
        #    So grad_x ~ g_tan.
        
        # This should work reasonably well.
        
        return obj_val, grad

    def optimize_covering(init_pts, max_iter=100):
        """
        Run L-BFGS-B optimization to minimize the soft-max covering energy.
        """
        x0 = init_pts.flatten()
        
        def fun_and_grad(x):
            pts = x.reshape(N, 3)
            # Normalize internally to handle any drift
            pts = normalize(pts)
            energy, grad_3d = soft_max_covering_energy(pts)
            
            # Project gradient to tangent space and map back
            # grad_3d is the gradient w.r.t u (on sphere).
            # We need gradient w.r.t x.
            # If x = r * u, and we assume r=1 locally, grad_x = grad_u.
            # But to be safe, we project grad_u onto the tangent space of the sphere at u.
            # g_tan = grad_u - (grad_u . u) * u
            dots_u = np.sum(grad_3d * pts, axis=1, keepdims=True)
            g_tan = grad_3d - dots_u * pts
            
            # The gradient w.r.t x is approximately g_tan (since the normalization map is an isometry on the sphere locally)
            # Actually, the chain rule is more complex, but this is a standard heuristic for sphere optimization with L-BFGS-B.
            
            return energy, g_tan.flatten()
        
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        # Normalize the result
        best_pts = normalize(res.x.reshape(N, 3))
        return best_pts

    # Main Strategy:
    # 1. Initialize with Fibonacci spiral
    # 2. Run L-BFGS-B optimization
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = fib_pts.copy()
    best_cr, _, _ = compute_covering_radius_and_worst_indices(best_pts)
    
    # Phase 1: Optimization from Fibonacci
    if time.time() < deadline - 20:
        opt_pts = optimize_covering(best_pts, max_iter=200)
        cr_opt, _, _ = compute_covering_radius_and_worst_indices(opt_pts)
        if cr_opt < best_cr:
            best_cr = cr_opt
            best_pts = opt_pts
            
    # Phase 2: Multi-start with random perturbations
    rng = np.random.default_rng(42)
    
    for k in range(5):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        init_pts = rng.normal(size=(N, 3))
        init_pts = normalize(init_pts)
        
        # Short optimization
        opt_pts = optimize_covering(init_pts, max_iter=100)
        cr_opt, _, _ = compute_covering_radius_and_worst_indices(opt_pts)
        
        if cr_opt < best_cr:
            best_cr = cr_opt
            best_pts = opt_pts
            
    return best_pts
# EVOLVE-BLOCK-END
