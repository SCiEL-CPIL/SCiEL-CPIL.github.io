"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _sphere_energy_and_grad(x_flat):
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, direction, step_size):
    new_pts = pts + step_size * direction
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    best_x = x0.copy()
    best_f, _ = _sphere_energy_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = _sphere_energy_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            _sphere_energy_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-16,
                'gtol': 1e-14,
                'maxfun': 15000
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=500, lr_init=0.05, lr_max=0.15):
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        new_pts = _retract(pts, -g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_proj = np.einsum('ij,ij->i', new_g, new_pts)
        new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.15, lr_max)
        else:
            lr *= 0.5
            if lr < 1e-14:
                break
    
    return pts, f

def _ultra_polish(pts, deadline, max_iter=2000):
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_t = g - proj[:, np.newaxis] * pts
    
    g_norm = np.linalg.norm(g_t)
    if g_norm < 1e-15:
        return pts, f
    
    alpha = 0.01
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        sufficient = 1e-18 * alpha * np.sum(g_t * (-g_t))
        cand = _retract(pts, -g_t, alpha)
        cand_f, cand_g = _energy_and_grad(cand)
        cand_proj = np.einsum('ij,ij->i', cand_g, cand)
        cand_g_t = cand_g - cand_proj[:, np.newaxis] * cand
        
        if cand_f <= f - sufficient:
            pts = cand
            f = cand_f
            g = cand_g
            g_t = cand_g_t
            if it % 10 == 0:
                alpha = min(alpha * 1.05, 0.05)
        else:
            alpha *= 0.5
            if alpha < 1e-18:
                break
        
        if np.linalg.norm(g_t) < 1e-15:
            break
    
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-12, deadline=None, m=25):
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    s_list = []
    y_list = []
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        if len(s_list) > 0:
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if abs(sy) > 1e-15 else 0.0)
            
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0
            r = -gamma * q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                beta_i = rho_i * np.sum(y_i * r)
                r = r + (alphas[i] - beta_i) * s_i
            
            direction = r
        else:
            direction = -g_t
        
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(50):
            candidate = _retract(x, direction, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            if new_f <= f + 1e-5 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-12:
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = new_x
        f = new_f
        g = new_g
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
    
    return x, f

def _rcg_polish(pts, deadline, max_iter=100):
    if time.time() > deadline - 3.0:
        return pts, _energy_and_grad(pts)[0]
    
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    d = g - proj[:, np.newaxis] * pts
    d = -d
    
    proj_d = np.einsum('ij,ij->i', d, pts)
    d = d - proj_d[:, np.newaxis] * pts
    
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-14:
        return pts, f
    
    d_prev = None
    g_prev = None
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        alpha = 0.1
        best_alpha = None
        best_f_new = f
        best_new_pts = None
        best_new_g = None
        best_new_g_t = None
        
        for ls in range(20):
            new_pts = _retract(pts, d, alpha)
            new_f, new_g = _energy_and_grad(new_pts)
            new_proj = np.einsum('ij,ij->i', new_g, new_pts)
            new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
            
            if new_f < best_f_new:
                best_f_new = new_f
                best_alpha = alpha
                best_new_pts = new_pts
                best_new_g = new_g
                best_new_g_t = new_g_tangent
            
            if new_f < f - 1e-4 * alpha * np.sum(d * d):
                break
            
            alpha *= 0.5
        
        if best_alpha is None:
            break
        
        pts = best_new_pts
        g = best_new_g
        f = best_f_new
        g_tangent = best_new_g_t
        
        if d_prev is not None:
            g_dot_g = np.sum(g_tangent * g_tangent)
            g_prev_dot_g_prev = np.sum(g_prev * g_prev)
            if g_prev_dot_g_prev > 1e-30:
                beta = g_dot_g / g_prev_dot_g_prev
            else:
                beta = 0.0
            d = -g_tangent + beta * d
        else:
            d = -g_tangent
        
        proj_d = np.einsum('ij,ij->i', d, pts)
        d = d - proj_d[:, np.newaxis] * pts
        
        d_prev = d.copy()
        g_prev = g_tangent.copy()
        
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-14:
            break
    
    return pts, f

def construct():
    start_time = time.time()
    deadline = start_time + 147.0
    
    initializations = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)
    
    # 2. Random points
    for seed_offset in range(4):
        rng_local = np.random.default_rng(42 + seed_offset)
        random_pts = rng_local.standard_normal((N, 3))
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / np.maximum(norms, 1e-15)
        initializations.append(random_pts)
    
    # 3. Perturbed Fibonacci
    rng_pert = np.random.default_rng(123)
    for pert_scale in [0.01, 0.05]:
        perturbed = fib_pts + rng_pert.standard_normal((N, 3)) * pert_scale
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        initializations.append(perturbed)
    
    # 4. Shifted Fibonacci
    for angle_shift in [0.1, 0.2]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + angle_shift
        r2 = np.sqrt(1.0 - z2 * z2)
        shifted_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initializations.append(shifted_pts)
    
    best_x = None
    best_f = float('inf')
    
    n_inits = len(initializations)
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        x0 = init_pts.flatten()
        
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
        
        time_per_init = max(15.0, (deadline - time.time() - 20.0) / max(1, n_inits - idx))
        local_deadline = time.time() + min(time_per_init, remaining - 20.0)
        
        x_opt, f_opt = _lbfgs_sphere(x0, local_deadline, maxiter=3000)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()
    
    if best_x is None:
        best_x = fib_pts.flatten()
        best_f, _ = _sphere_energy_and_grad(best_x)
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        # Pass 1: standard gradient polish
        pts, f = _polish(pts, deadline, max_iter=300)
        best_x = pts.flatten()
        best_f = f
        
        # Pass 2: Custom L-BFGS on sphere for deeper optimization
        if time.time() < deadline - 4.0:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _lbfgs_sphere_custom(pts, max_iter=200, tol=1e-13, deadline=deadline - 2.0)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Pass 3: RCG polish
        if time.time() < deadline - 3.0:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _rcg_polish(pts, deadline, max_iter=150)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Pass 4: another gradient polish
        if time.time() < deadline - 1.5:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _polish(pts, deadline, max_iter=200)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Pass 5: ultra-fine polish
        if time.time() < deadline - 0.5:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _ultra_polish(pts, deadline, max_iter=1000)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Pass 6: final ultra-fine
        if time.time() < deadline - 0.2:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _ultra_polish(pts, deadline, max_iter=500)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
    
    pts_opt = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / np.maximum(norms, 1e-15)
    
    return pts_opt
# EVOLVE-BLOCK-END
