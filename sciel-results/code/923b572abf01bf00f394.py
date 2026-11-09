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

def _lbfgs_manifold_polish(pts, deadline=None, maxiter=10000, ftol=1e-16, gtol=1e-15):
    """
    Use L-BFGS-B in a retraction-based manifold optimization framework.
    """
    x = pts.copy()
    x = _project_to_sphere(x)
    
    def obj_and_grad(y_flat):
        y = y_flat.reshape(N, 3)
        x_local = _project_to_sphere(y)
        energy, grad_tan = _energy_and_tangent_grad(x_local)
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
            'maxiter': maxiter,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50
        }
    )
    
    e_before = _energy_and_tangent_grad(x)[0]
    if result.fun < e_before:
        y_opt = result.x.reshape(N, 3)
        x_opt = _project_to_sphere(y_opt)
        return x_opt
    
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    
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
        if time.time() > deadline - 20:
            break
            
        time_left = deadline - time.time()
        run_deadline = time.time() + min(4.0, time_left - 10.0)
        
        if run_deadline <= time.time():
            continue
            
        optimized_pts = _conjugate_gradient(init_pts, max_iter=500, deadline=run_deadline)
        
        e_opt, _ = _energy_and_tangent_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
            
    # L-BFGS-B polish on the best candidate
    if time.time() < deadline - 10:
        lbfgs_pts = _lbfgs_manifold_polish(best_pts, deadline=deadline - 8, maxiter=5000, ftol=1e-15, gtol=1e-14)
        e_lbfgs, _ = _energy_and_tangent_grad(lbfgs_pts)
        if e_lbfgs < best_energy:
            best_energy = e_lbfgs
            best_pts = lbfgs_pts.copy()
    
    # Perturbation + re-optimization to escape potential local minima
    if time.time() < deadline - 15:
        rng2 = np.random.default_rng(123)
        for perturb_trial in range(3):
            if time.time() > deadline - 12:
                break
            # Small random perturbation in tangent space
            noise = rng2.normal(size=(N, 3)) * 0.005
            perturbed = best_pts + noise
            perturbed = _project_to_sphere(perturbed)
            
            # Quick CG to see if it improves
            quick_deadline = time.time() + min(3.0, deadline - time.time() - 8)
            if quick_deadline <= time.time():
                break
            opt_perturbed = _conjugate_gradient(perturbed, max_iter=300, deadline=quick_deadline)
            e_pert, _ = _energy_and_tangent_grad(opt_perturbed)
            
            if e_pert < best_energy:
                best_energy = e_pert
                best_pts = opt_perturbed.copy()
    
    # Final CG polish with tight tolerance
    if time.time() < deadline - 8:
        final_pts = _conjugate_gradient(best_pts, max_iter=3000, tol=1e-13, deadline=deadline - 6)
        e_final, _ = _energy_and_tangent_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
    # High-precision L-BFGS-B polish
    if time.time() < deadline - 6:
        lbfgs_pts2 = _lbfgs_manifold_polish(best_pts, deadline=deadline - 4, maxiter=10000, ftol=1e-16, gtol=1e-15)
        e_lbfgs2, _ = _energy_and_tangent_grad(lbfgs_pts2)
        if e_lbfgs2 < best_energy:
            best_energy = e_lbfgs2
            best_pts = lbfgs_pts2.copy()
    
    # Final ultra-tight CG
    if time.time() < deadline - 4:
        final_pts2 = _conjugate_gradient(best_pts, max_iter=5000, tol=1e-14, deadline=deadline - 2)
        e_final2, _ = _energy_and_tangent_grad(final_pts2)
        if e_final2 < best_energy:
            best_energy = e_final2
            best_pts = final_pts2.copy()
    
    # Last L-BFGS polish
    if time.time() < deadline - 2:
        lbfgs_pts3 = _lbfgs_manifold_polish(best_pts, deadline=deadline - 0.5, maxiter=20000, ftol=1e-16, gtol=1e-15)
        e_lbfgs3, _ = _energy_and_tangent_grad(lbfgs_pts3)
        if e_lbfgs3 < best_energy:
            best_energy = e_lbfgs3
            best_pts = lbfgs_pts3.copy()

    best_pts = _project_to_sphere(best_pts)
    
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
