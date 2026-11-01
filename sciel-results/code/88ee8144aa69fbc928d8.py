# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
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
    """
    Objective and gradient for scipy.optimize.minimize.
    x_flat is a 3*N array of coordinates.
    We normalize each point to the sphere inside this function.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    # Project gradient onto tangent space for the effective gradient
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, grad_tangent, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts - step_size * grad_tangent
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    """
    Run L-BFGS-B on the sphere with time guard.
    """
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
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 10000
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=500):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space. Uses adaptive learning rate.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = 0.05
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        new_pts = _retract(pts, g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_proj = np.einsum('ij,ij->i', new_g, new_pts)
        new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.15, 0.15)
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=300, tol=1e-10, deadline=None, m=20):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    # Project gradient to tangent space
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    # L-BFGS memory
    s_list = []  # steps
    y_list = []  # gradient differences
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Two-loop recursion to compute direction
        if len(s_list) > 0:
            # Compute rho
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if sy != 0 else 0.0)
            
            # Two-loop recursion
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0  # initial step length
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
        
        # Line search with backtracking
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(40):
            candidate = _retract(x, direction, -step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            # Armijo condition
            if new_f <= f + 1e-4 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        # Update L-BFGS memory
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-10:
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

def _simulated_annealing(pts, deadline, max_iter=100000, rng=None):
    """
    Simulated Annealing with periodic local refinement.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    current_pts = pts.copy()
    current_f, _ = _energy_and_grad(current_pts)
    best_pts = current_pts.copy()
    best_f = current_f
    
    # Initial temperature
    T = 0.5 
    T_min = 1e-6
    cooling_rate = 0.9999
    
    # Number of points to perturb
    n_perturb = max(2, N // 10)
    
    for i in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Periodic local refinement
        if i % 1000 == 0 and i > 0:
            # Run a quick polish to snap to local min
            refined_pts, refined_f = _polish(current_pts, deadline - 10.0, max_iter=10)
            if refined_f < current_f:
                current_pts = refined_pts
                current_f = refined_f
                if current_f < best_f:
                    best_f = current_f
                    best_pts = current_pts.copy()
            
        # Generate perturbation
        indices = rng.choice(N, n_perturb, replace=False)
        noise = rng.randn(n_perturb, 3) * 0.05
        new_pts = current_pts.copy()
        new_pts[indices] += noise
        
        # Normalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        
        # Calculate energy
        new_f, _ = _energy_and_grad(new_pts)
        
        # Acceptance criterion
        delta_f = new_f - current_f
        if delta_f < 0:
            # Always accept if better
            current_pts = new_pts
            current_f = new_f
            if current_f < best_f:
                best_f = current_f
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(-delta_f / T)
            if T > T_min:
                prob = np.exp(-delta_f / T)
                if rng.rand() < prob:
                    current_pts = new_pts
                    current_f = new_f
        
        # Cool down
        T *= cooling_rate
        
    return best_pts, best_f

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci lattice) - good global structure
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(pts_fib)
    
    # 2. A few random starts
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        pts_rand = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        pts_rand = pts_rand / norms
        inits.append(pts_rand)
        
    # 3. Perturbed Fibonacci
    np.random.seed(11)
    noise = np.random.randn(N, 3) * 0.05
    pts_pert = pts_fib + noise
    norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
    pts_pert = pts_pert / norms
    inits.append(pts_pert)

    # Optimize each initialization with a quick local method to get a good starting point for SA
    n_inits = len(inits)
    time_per_init = 10.0  # Quick start
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        init_deadline = min(deadline - 20.0, time.time() + time_per_init)
        
        # Use custom L-BFGS for quick convergence
        pts_opt, energy = _lbfgs_sphere_custom(init_pts, max_iter=100, tol=1e-8, deadline=init_deadline, m=10)
        
        # Ensure points are on the sphere
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        pts_opt = pts_opt / np.maximum(norms, 1e-15)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
            
    if best_pts is None:
        best_pts = pts_fib
        best_energy, _ = _energy_and_grad(best_pts)

    # Run Simulated Annealing from the best initial point
    if time.time() < deadline - 15.0:
        sa_deadline = deadline - 15.0
        rng = np.random.RandomState(int(time.time() * 1000) % 2**32)
        best_pts, best_energy = _simulated_annealing(best_pts, sa_deadline, max_iter=200000, rng=rng)
        
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _polish(best_pts, deadline, max_iter=500)
    
    # Second round of polish if time permits
    if time.time() < deadline - 3.0:
        best_pts, best_energy = _polish(best_pts, deadline, max_iter=300)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)
    
    return best_pts
# EVOLVE-BLOCK-END
