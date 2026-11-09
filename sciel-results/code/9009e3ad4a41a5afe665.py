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
    E = sum_{i<j} log(1/|xi-xj|) = -0.5 * sum_{i<j} log(d_ij^2)
      = -0.25 * sum_{i!=j} log(d_ij^2)
    grad_raw_i = sum_{j!=i} (xi - xj) / d_ij^2
    (projected to tangent space for manifold optimization)
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
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

    dot = np.einsum('ij,ij->i', grad_raw, pts)
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

def _newton_polish(pts, max_iter=30, tol=1e-14, deadline=None):
    """
    Newton's method on the tangent space for a final high-precision polish.
    Builds the full Hessian of the log-energy on the sphere and solves
    for the tangent-space step.  Only used near convergence where the
    Hessian is well-conditioned.
    """
    x = _project_to_sphere(pts)
    energy, grad = _energy_and_tangent_grad(x)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break

        # ---- Build Hessian in ambient R^3 ----
        # H_amb[i,j] = sum_{k != i,j} (delta_ij * d_ij^2 - 2*(xi-xj)_i*(xi-xj)_j) / d_ij^4
        # For i == j:  H_amb[i,i] = 2 * sum_{k != i} (xi - xk) (xi - xk)^T / d_ik^4
        diff = x[:, None, :] - x[None, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-12)
        inv_d4 = 1.0 / (dist_sq * dist_sq)
        np.fill_diagonal(inv_d4, 0.0)

        # Off-diagonal Hessian blocks
        # H_amb[i,j] = inv_d4[i,j] * (delta_ij * d_ij^2 - 2 * diff[i,j])  for i != j
        H_amb = np.zeros((N, N, 3, 3))
        for i in range(N):
            for j in range(i + 1, N):
                w = inv_d4[i, j]
                d2 = dist_sq[i, j]
                dvec = diff[i, j]
                # delta_ij * d^2 term: 0 for i != j
                # -2 * dvec_i * dvec_j
                block = -2.0 * w * np.outer(dvec, dvec)
                H_amb[i, j] = block
                H_amb[j, i] = block

        # Diagonal Hessian blocks
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            w_i = inv_d4[i, mask]
            dvec_i = diff[i, mask]  # (N-1, 3)
            # H_amb[i,i] = 2 * sum_k w_i[k] * outer(dvec_i[k], dvec_i[k])
            H_amb[i, i] = 2.0 * np.einsum('k,ka,kb->ab', w_i, dvec_i, dvec_i)

        # ---- Project to tangent space ----
        # H_tan = P_i @ H_amb @ P_j  where P_i = I - x_i x_i^T
        # Step: H_tan @ step = -grad_tan
        # In flattened form: (I_N - xx^T) kron H_amb (I_N - xx^T) @ step_flat = -grad_flat
        # We solve: M @ s = -g  where M is the tangent Hessian
        # Build M as (N*3, N*3) matrix
        # M = P @ H_amb_flat @ P  where P = I - kron(xx^T)
        # More efficiently: M[i,j] = P_i @ H_amb[i,j] @ P_j
        # We'll build M explicitly (N*3 = 300, so 90000 elements, fine)

        M = np.zeros((N * 3, N * 3))
        for i in range(N):
            Pi = np.eye(3) - np.outer(x[i], x[i])
            for j in range(N):
                Pj = np.eye(3) - np.outer(x[j], x[j])
                M[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = Pi @ H_amb[i, j] @ Pj

        g_flat = grad.flatten()

        # Solve M @ s = -g_flat with damped least squares if needed
        try:
            s_flat, residuals, rank, sv = np.linalg.lstsq(M, -g_flat, rcond=None)
        except np.linalg.LinAlgError:
            break

        # Damping: if step too large, shrink
        s_norm = np.linalg.norm(s_flat)
        if s_norm > 0.1:
            s_flat *= 0.1 / s_norm

        s = s_flat.reshape(N, 3)

        # Retraction: move and project
        x_new = x + s
        x_new = _project_to_sphere(x_new)

        e_new, g_new = _energy_and_tangent_grad(x_new)

        if e_new < energy:
            x = x_new
            energy = e_new
            grad = g_new
        else:
            # Try half step
            x_new = x + 0.5 * s
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)
            if e_new < energy:
                x = x_new
                energy = e_new
                grad = g_new
            else:
                break

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

    # Newton polish for final high-precision convergence
    if time.time() < deadline - 2:
        newton_pts = _newton_polish(best_pts, max_iter=50, tol=1e-15, deadline=deadline - 0.5)
        e_newton, _ = _energy_and_tangent_grad(newton_pts)
        if e_newton < best_energy:
            best_energy = e_newton
            best_pts = newton_pts.copy()

    # Last L-BFGS polish
    if time.time() < deadline - 2:
        lbfgs_pts3 = _lbfgs_manifold_polish(best_pts, deadline=deadline - 0.5, maxiter=20000, ftol=1e-16, gtol=1e-15)
        e_lbfgs3, _ = _energy_and_tangent_grad(lbfgs_pts3)
        if e_lbfgs3 < best_energy:
            best_energy = e_lbfgs3
            best_pts = lbfgs_pts3.copy()

    # One more Newton pass after L-BFGS
    if time.time() < deadline - 1:
        newton_pts2 = _newton_polish(best_pts, max_iter=30, tol=1e-15, deadline=deadline - 0.2)
        e_newton2, _ = _energy_and_tangent_grad(newton_pts2)
        if e_newton2 < best_energy:
            best_energy = e_newton2
            best_pts = newton_pts2.copy()

    best_pts = _project_to_sphere(best_pts)
    
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
