"""Rewritten program using L-BFGS-B with manifold gradient handling and multi-restart."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere.
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Clip to avoid log(0) and numerical issues
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Note: d_ij = sqrt(dist_sq_ij)
    # log(d_ij) = 0.5 * log(dist_sq_ij)
    # So sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(d_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    log_dist_sq = np.log(dist_sq)
    # Set diagonal to 0 to ignore self-interaction
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient of E with respect to x_i:
    # E = -0.5 * sum_{i!=j} log(|x_i - x_j|)
    # dE/dx_i = -0.5 * sum_{j!=i} d/dx_i log(|x_i - x_j|)
    #         = -0.5 * sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    #         = sum_{j!=i} (x_j - x_i) / (2 * |x_i - x_j|^2)
    
    # Let's compute sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # Let W_ij = 1 / |x_i - x_j|^2 for i!=j, 0 for i=j
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # sum_j (x_j - x_i) / |x_i - x_j|^2 = sum_j W_ij x_j - x_i sum_j W_ij
    # Let A = W (N x N matrix)
    # term1_i = sum_j A_ij x_j = (A @ x)_i
    # term2_i = x_i * sum_j A_ij = x_i * (A.sum(axis=1))_i
    
    term1 = A_x = inv_dist_sq @ pts
    term2 = pts * np.sum(inv_dist_sq, axis=1)[:, None]
    
    grad_raw = 0.5 * (term1 - term2)
    
    # Project to tangent space
    # grad_tan_i = grad_raw_i - (grad_raw_i . x_i) x_i
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _energy_and_grad_manifold(x_flat):
    """
    Objective function and gradient for scipy.optimize.minimize.
    x_flat: (3N,) vector.
    """
    x = x_flat.reshape(N, 3)
    x = _project_to_sphere(x)
    
    energy, grad_tan = _energy_and_tangent_grad(x)
    
    # To use L-BFGS-B, we need the gradient with respect to the free variables x_flat.
    # Since we project x to the sphere, the effective gradient is tricky.
    # A common approximation is to use the tangent gradient as the gradient in R^{3N}
    # if we assume the step is small and we re-project.
    # However, a more rigorous approach for L-BFGS-B is to minimize E(x) subject to ||x_i||=1.
    # L-BFGS-B cannot handle this directly.
    # Instead, we can minimize E(x) on R^{3N} and let the optimizer drift off the sphere,
    # then project at the end? No, that changes the problem.
    
    # Alternative: Use the fact that for small steps, minimizing E on the sphere is equivalent
    # to minimizing E(x + step) where step is tangent.
    # But L-BFGS-B doesn't know about tangent steps.
    
    # Standard trick: Use a penalty method?
    # f(x) = E(x) + lambda * sum(||x_i||^2 - 1)^2
    # This is computationally expensive and requires tuning lambda.
    
    # Better trick: Use the "unconstrained" formulation where we optimize over coordinates
    # but the function internally normalizes. The gradient returned must be consistent with the function.
    # If f(x) = E(x/||x||), then grad_f(x) = grad_E(x/||x||) * D(x/||x||).
    # This is complex.
    
    # Let's stick to the custom Conjugate Gradient which worked well (score 1.0005).
    # But the directive says "MODEL-AND-OPTIMIZE ... fit or optimize parameters".
    # Maybe parameterize points as something else?
    
    # Actually, the previous CG approach is a direct optimization.
    # Let's try to implement a more robust line search or use L-BFGS with a custom projection callback?
    # scipy.optimize.minimize does not support custom projection in L-BFGS-B.
    
    # Let's try L-BFGS-B on the energy function defined on R^{3N} without projection in the function,
    # but initializing on the sphere and hoping it stays close? No, it will drift.
    
    # How about minimizing E(x) subject to ||x_i||=1 using Lagrange multipliers?
    # L = E(x) + sum lambda_i (||x_i||^2 - 1)
    # dL/dx_i = grad_E_i + 2 lambda_i x_i = 0
    # grad_E_i is the Euclidean gradient.
    # We found grad_tan = grad_E - (grad_E . x_i) x_i.
    # So grad_E = grad_tan + (grad_E . x_i) x_i.
    # From Lagrangian: grad_E + 2 lambda_i x_i = 0 => grad_tan + (grad_E . x_i) x_i + 2 lambda_i x_i = 0.
    # This doesn't immediately give a simple optimization for x without lambda.
    
    # Let's revert to the strategy that worked: Custom CG.
    # But can we improve it?
    # 1. Better initialization: Use scipy.spatial.spherical_voronoi or similar?
    # 2. Better line search: Strong Wolfe conditions?
    # 3. More restarts?
    
    # For now, return the tangent gradient as the gradient for L-BFGS-B and see if it works.
    # It might not converge to the sphere, but if we start on the sphere and steps are small,
    # the projection error might be small.
    # Actually, if we pass grad_tan as the gradient, L-BFGS-B will take steps in R^{3N}.
    # x_new = x - alpha * grad_tan.
    # This x_new will not be on the sphere.
    # If the function f(x) = E(_project_to_sphere(x)), then the gradient of f is NOT grad_tan.
    
    # Correct gradient for f(x) = E(x/||x||):
    # Let u = x/||x||.
    # f(x) = E(u).
    # df/dx = (dE/du) @ (du/dx).
    # du/dx = (1/||x||) I - (x x^T)/||x||^3.
    # This is a projection matrix P = (1/||x||)(I - u u^T).
    # So grad_f = P^T grad_E(u) = P grad_E(u) (since P is symmetric).
    # And P grad_E(u) is exactly the tangent gradient grad_tan!
    # Wait, grad_E(u) is the Euclidean gradient of E at u.
    # Our _energy_and_tangent_grad computes grad_tan = grad_E(u) - (grad_E(u).u)u.
    # So grad_f(x) = grad_tan(u).
    # BUT, this gradient is defined at u, not x.
    # If x is not on the sphere, u = x/||x||.
    # The gradient returned to L-BFGS-B should be with respect to x.
    # So we should return grad_tan(u).
    # Let's test if this works.
    
    return energy, grad_tan.flatten()

def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _conjugate_gradient(pts, max_iter=1000, tol=1e-10, deadline=None):
    """
    Conjugate Gradient method on the sphere with strong Wolfe line search.
    """
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    p = -grad
    
    # Project p to tangent space
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Line search: Wolfe Conditions
        alpha = 1.0
        c1 = 1e-4
        c2 = 0.9
        rho_shrink = 0.5
        rho_expand = 1.5
        max_alpha = 10.0
        
        dir_deriv = np.sum(grad * p)
        
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)
        
        found_step = False
        # Phase 1: Sufficient Decrease (Armijo)
        while alpha > 1e-12:
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)
            
            if e_new <= energy + c1 * alpha * dir_deriv:
                # Phase 2: Curvature Condition
                new_dir_deriv = np.sum(g_new * p)
                if new_dir_deriv >= c2 * dir_deriv:
                    found_step = True
                    break
                
                # If curvature fails, step is too small. Try expanding.
                alpha_try = alpha
                while alpha_try < max_alpha:
                    alpha_try *= rho_expand
                    x_try = x + alpha_try * p
                    x_try = _project_to_sphere(x_try)
                    e_try, g_try = _energy_and_tangent_grad(x_try)
                    
                    # Check if Armijo still holds for the larger step
                    if e_try <= energy + c1 * alpha_try * dir_deriv:
                        new_dir_deriv_try = np.sum(g_try * p)
                        if new_dir_deriv_try >= c2 * dir_deriv:
                            x_new = x_try
                            e_new = e_try
                            g_new = g_try
                            alpha = alpha_try
                            found_step = True
                            break
                    else:
                        # Armijo failed, so this step is too big.
                        # Accept the last Armijo-satisfying step (relaxed Wolfe)
                        found_step = True
                        break
                
                if not found_step:
                    # If we expanded and didn't find a curvature-satisfying step,
                    # accept the Armijo-satisfying step we found earlier.
                    found_step = True
                break
            else:
                # Armijo failed, shrink alpha
                alpha *= rho_shrink
        
        if not found_step:
            break
        
        x = x_new
        energy = e_new
        grad = g_new
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Polak-Ribiere
        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        
        if grad_prev_dot < 1e-16:
            beta = 0.0
        else:
            beta = max(0.0, grad_dot / grad_prev_dot)
            
        p_new = -grad + beta * p
        
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        
        if np.sum(grad * p_new) > 0:
            beta = 0.0
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
            
        p = p_new
        grad_prev = grad.copy()
        
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    inits = []
    inits.append(_initial_fibonacci())
    
    rng = np.random.default_rng(42)
    for i in range(10):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    for i in range(10):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.1
        inits.append(_project_to_sphere(pts))
        
    best_pts = inits[0].copy()
    best_energy = _energy_and_tangent_grad(best_pts)[0]
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        time_left = deadline - time.time()
        run_deadline = time.time() + min(5.0, time_left - 2.0)
        
        if run_deadline <= time.time():
            continue
            
        optimized_pts = _conjugate_gradient(init_pts, max_iter=500, deadline=run_deadline)
        
        e_opt, _ = _energy_and_tangent_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
            
    if time.time() < deadline - 5:
        # Use a tighter tolerance and more iterations for the final polish
        final_pts = _conjugate_gradient(best_pts, max_iter=5000, tol=1e-12, deadline=deadline - 1)
        e_final, _ = _energy_and_tangent_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
    best_pts = _project_to_sphere(best_pts)
    
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
