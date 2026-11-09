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
    
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    term1 = inv_dist_sq @ pts
    term2 = pts * np.sum(inv_dist_sq, axis=1)[:, None]
    
    grad_raw = 0.5 * (term1 - term2)
    
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _simple_gradient_descent(pts, steps=50, step_size=0.01, deadline=None):
    x = pts.copy()
    for _ in range(steps):
        if deadline and time.time() > deadline:
            break
        energy, grad = _energy_and_tangent_grad(x)
        x_new = x - step_size * grad
        x_new = _project_to_sphere(x_new)
        x = x_new
    return x

def _conjugate_gradient(pts, max_iter=1000, tol=1e-10, deadline=None):
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    p = -grad
    
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
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
        while alpha > 1e-12:
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)
            
            if e_new <= energy + c1 * alpha * dir_deriv:
                new_dir_deriv = np.sum(g_new * p)
                if new_dir_deriv >= c2 * dir_deriv:
                    found_step = True
                    break
                
                alpha_try = alpha
                while alpha_try < max_alpha:
                    alpha_try *= rho_expand
                    x_try = x + alpha_try * p
                    x_try = _project_to_sphere(x_try)
                    e_try, g_try = _energy_and_tangent_grad(x_try)
                    
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
                        found_step = True
                        break
                
                if not found_step:
                    found_step = True
                break
            else:
                alpha *= rho_shrink
        
        if not found_step:
            break
        
        x = x_new
        energy = e_new
        grad = g_new
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
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

def _lbfgs_manifold_polish(pts, deadline=None):
    """
    Use L-BFGS-B in a retraction-based manifold optimization framework.
    We parameterize each point as x_i = normalize(y_i) where y_i are unconstrained.
    The gradient with respect to y_i is the projection of the tangent gradient.
    """
    x = pts.copy()
    x = _project_to_sphere(x)
    
    def obj_and_grad(y_flat):
        y = y_flat.reshape(N, 3)
        # Retraction: normalize
        x_local = _project_to_sphere(y)
        
        energy, grad_tan = _energy_and_tangent_grad(x_local)
        
        # Gradient with respect to y:
        # x = y / |y|
        # dx/dy_i = (I - x_i x_i^T) / |y_i|
        # So grad_y = grad_tan * |y| (since (I - xx^T) is the projection)
        norms_y = np.linalg.norm(y, axis=1, keepdims=True)
        norms_y = np.where(norms_y < 1e-12, 1.0, norms_y)
        grad_y = grad_tan * norms_y
        
        return energy, grad_y.flatten()
    
    y_init = x.flatten()
    
    result = minimize(
        obj_and_grad,
        y_init,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'ftol': 1e-15,
            'gtol': 1e-14,
            'maxls': 50
        }
    )
    
    if result.success or result.fun < _energy_and_tangent_grad(x)[0]:
        y_opt = result.x.reshape(N, 3)
        x_opt = _project_to_sphere(y_opt)
        return x_opt
    
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    inits = []
    inits.append(_initial_fibonacci())
    
    rng = np.random.default_rng(42)
    
    # 5 pure random
    for i in range(5):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    # 5 small jitter Fibonacci
    for i in range(5):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.01
        inits.append(_project_to_sphere(pts))
        
    # 5 large jitter Fibonacci
    for i in range(5):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.1
        inits.append(_project_to_sphere(pts))
        
    # 5 pre-relaxed Fibonacci (using simple GD)
    for i in range(5):
        base = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.05
        base = _project_to_sphere(base)
        relaxed = _simple_gradient_descent(base, steps=50, step_size=0.01, deadline=start_time + 10)
        inits.append(relaxed)
        
    # 5 medium jitter Fibonacci
    for i in range(5):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.05
        inits.append(_project_to_sphere(pts))
        
    best_pts = inits[0].copy()
    best_energy = _energy_and_tangent_grad(best_pts)[0]
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        time_left = deadline - time.time()
        run_deadline = time.time() + min(4.0, time_left - 5.0)
        
        if run_deadline <= time.time():
            continue
            
        optimized_pts = _conjugate_gradient(init_pts, max_iter=500, deadline=run_deadline)
        
        e_opt, _ = _energy_and_tangent_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
            
    # L-BFGS-B polish on the best candidate
    if time.time() < deadline - 5:
        lbfgs_pts = _lbfgs_manifold_polish(best_pts, deadline=deadline - 2)
        e_lbfgs, _ = _energy_and_tangent_grad(lbfgs_pts)
        if e_lbfgs < best_energy:
            best_energy = e_lbfgs
            best_pts = lbfgs_pts.copy()
    
    # Final CG polish with tight tolerance
    if time.time() < deadline - 2:
        final_pts = _conjugate_gradient(best_pts, max_iter=3000, tol=1e-13, deadline=deadline - 1)
        e_final, _ = _energy_and_tangent_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
    # Another L-BFGS-B polish if time permits
    if time.time() < deadline - 1:
        lbfgs_pts2 = _lbfgs_manifold_polish(best_pts, deadline=deadline - 0.1)
        e_lbfgs2, _ = _energy_and_tangent_grad(lbfgs_pts2)
        if e_lbfgs2 < best_energy:
            best_energy = e_lbfgs2
            best_pts = lbfgs_pts2.copy()

    best_pts = _project_to_sphere(best_pts)
    
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END

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
    
    term1 = inv_dist_sq @ pts
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

def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _simple_gradient_descent(pts, steps=50, step_size=0.01, deadline=None):
    """
    Simple gradient descent to relax the configuration.
    """
    x = pts.copy()
    for _ in range(steps):
        if deadline and time.time() > deadline:
            break
        energy, grad = _energy_and_tangent_grad(x)
        # Move in direction of negative gradient
        x_new = x - step_size * grad
        x_new = _project_to_sphere(x_new)
        x = x_new
    return x

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
    
    # 5 pure random
    for i in range(5):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    # 5 small jitter Fibonacci
    for i in range(5):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.01
        inits.append(_project_to_sphere(pts))
        
    # 5 large jitter Fibonacci
    for i in range(5):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.1
        inits.append(_project_to_sphere(pts))
        
    # 5 pre-relaxed Fibonacci (using simple GD)
    for i in range(5):
        base = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.05
        base = _project_to_sphere(base)
        # Quick relax
        relaxed = _simple_gradient_descent(base, steps=50, step_size=0.01, deadline=start_time + 10)
        inits.append(relaxed)
        
    best_pts = inits[0].copy()
    best_energy = _energy_and_tangent_grad(best_pts)[0]
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        time_left = deadline - time.time()
        run_deadline = time.time() + min(5.0, time_left - 5.0)
        
        if run_deadline <= time.time():
            continue
            
        optimized_pts = _conjugate_gradient(init_pts, max_iter=500, deadline=run_deadline)
        
        e_opt, _ = _energy_and_tangent_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
            
    if time.time() < deadline - 5:
        # Use a tighter tolerance and more iterations for the final polish
        # Increase max_iter significantly for the final best candidate
        final_pts = _conjugate_gradient(best_pts, max_iter=2000, tol=1e-12, deadline=deadline - 1)
        e_final, _ = _energy_and_tangent_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
            
    # One more ultra-tight polish if time permits
    if time.time() < deadline - 1:
        final_pts2 = _conjugate_gradient(best_pts, max_iter=5000, tol=1e-14, deadline=deadline - 0.1)
        e_final2, _ = _energy_and_tangent_grad(final_pts2)
        if e_final2 < best_energy:
            best_energy = e_final2
            best_pts = final_pts2.copy()

    best_pts = _project_to_sphere(best_pts)
    
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
