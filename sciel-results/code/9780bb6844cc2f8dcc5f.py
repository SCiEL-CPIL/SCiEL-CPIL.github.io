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
    """
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
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts + step_size * direction
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    """
    Run L-BFGS-B on the sphere with time guard and tight tolerances.
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
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space. Uses adaptive learning rate.
    """
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
    """
    Ultra-precise Newton-style polish with very small steps and tight Armijo.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_t = g - proj[:, np.newaxis] * pts
    
    # Estimate local curvature scale from gradient magnitude
    g_norm = np.linalg.norm(g_t)
    if g_norm < 1e-15:
        return pts, f
    
    # Start with a small step relative to gradient scale
    alpha = 0.01
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        # Armijo backtracking
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
            # Slightly increase step if we made good progress
            if it % 10 == 0:
                alpha = min(alpha * 1.05, 0.05)
        else:
            alpha *= 0.5
            if alpha < 1e-18:
                break
        
        # Check convergence
        if np.linalg.norm(g_t) < 1e-15:
            break
    
    return pts, f

def _hessian_and_rhessian(pts):
    """
    Compute the Riemannian Hessian (approximate) for the logarithmic energy on the sphere.
    For E = -sum_{i<j} log(|x_i - x_j|), the Hessian in ambient space is:
    H_ii = sum_{j!=i} [I/d_ij^2 - 3*(x_i-x_j)(x_i-x_j)^T/d_ij^4]
    H_ij = -[I/d_ij^2 - 3*(x_i-x_j)(x_i-x_j)^T/d_ij^4] for i!=j
    
    The Riemannian Hessian projects this onto the tangent space.
    We use a simplified version that computes the key components.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Build the ambient Hessian block structure
    # H is (n*3, n*3), but we work with per-point blocks
    # H_ii = sum_{j!=i} [I/|d_ij|^2 - 3*d_ij*d_ij^T/|d_ij|^4]
    # H_ij = -[I/|d_ij|^2 - 3*d_ij*d_ij^T/|d_ij|^4]
    
    # Compute weight matrices
    W = np.zeros((n, n))  # W[i,j] = 1/d_ij^2
    V = np.zeros((n, n))  # V[i,j] = 3/d_ij^4
    
    iu = np.triu_indices(n, k=1)
    W[iu] = 1.0 / dist2[iu]
    W[(iu[1], iu[0])] = W[iu]
    V[iu] = 3.0 / (dist2[iu] ** 2)
    V[(iu[1], iu[0])] = V[iu]
    
    # For each point i, the diagonal block is:
    # H_ii = sum_{j!=i} [W[i,j]*I - V[i,j]*d_ij*d_ij^T]
    # Off-diagonal block H_ij = -[W[i,j]*I - V[i,j]*d_ij*d_ij^T]
    
    return W, V, diff, dist2

def _newton_polish(pts, deadline, max_iter=100):
    """
    Newton-style polish using the Riemannian Hessian.
    Solves H * delta = -g_tangent in the tangent space.
    Uses CG to solve the linear system for efficiency.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    W, V, diff, dist2 = _hessian_and_rhessian(x)
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        g_norm = np.linalg.norm(g_t)
        if g_norm < 1e-14:
            break
        
        # Build the Hessian-vector product operator
        # For a tangent vector v (n x 3), the Hessian applied to v is:
        # (Hv)_i = sum_j [W[i,j]*(v_i - v_j) - V[i,j]*d_ij*(d_ij . (v_i - v_j))]
        # This is the ambient Hessian projected appropriately.
        
        def hvp(v):
            """Compute Hessian-vector product (ambient, then project to tangent)."""
            # v is (n, 3)
            # (Hv)_i = sum_{j!=i} [W[i,j]*v_i - W[i,j]*v_j - V[i,j]*d_ij*(d_ij . (v_i - v_j))]
            # = W[i,:].dot(v) - sum_j W[i,j]*v_j - sum_j V[i,j]*d_ij*(d_ij . (v_i - v_j))
            
            # First term: sum_j W[i,j] * v_i = (sum_j W[i,j]) * v_i
            w_row_sum = np.sum(W, axis=1)  # (n,)
            term1 = w_row_sum[:, np.newaxis] * v  # (n, 3)
            
            # Second term: sum_j W[i,j] * v_j
            term2 = W @ v  # (n, 3)
            
            # Third term: sum_j V[i,j] * d_ij * (d_ij . (v_i - v_j))
            # d_ij = x_i - x_j
            # d_ij . (v_i - v_j) = sum_k d_ijk * (v_ik - v_jk)
            dv = v[:, np.newaxis, :] - v[np.newaxis, :, :]  # (n, n, 3)
            dot_dv = np.sum(diff * dv, axis=2)  # (n, n)
            term3 = np.einsum('ij,ij,ijk->ik', V, dot_dv, diff)  # (n, 3)
            
            hv = term1 - term2 - term3
            return hv
        
        # Solve H * delta = -g_t using conjugate gradient
        # Project CG onto tangent space
        b = -g_t
        r = b.copy()
        p = r.copy()
        rs = np.sum(r * r)
        
        delta = np.zeros_like(x)
        converged = False
        
        for cg_iter in range(60):
            Ap = hvp(p)
            # Project Ap to tangent space
            ap_proj = np.einsum('ij,ij->i', Ap, x)
            Ap_t = Ap - ap_proj[:, np.newaxis] * x
            
            pAp = np.sum(p * Ap_t)
            if abs(pAp) < 1e-30:
                break
            
            alpha_cg = rs / pAp
            delta = delta + alpha_cg * p
            r_new = r - alpha_cg * Ap_t
            
            # Project r_new to tangent
            rn_proj = np.einsum('ij,ij->i', r_new, x)
            r_new = r_new - rn_proj[:, np.newaxis] * x
            
            rs_new = np.sum(r_new * r_new)
            if np.sqrt(rs_new) < 1e-14 * np.sqrt(rs):
                converged = True
                break
            
            beta_cg = rs_new / rs
            p_new = r_new + beta_cg * p
            # Project p_new to tangent
            pn_proj = np.einsum('ij,ij->i', p_new, x)
            p_new = p_new - pn_proj[:, np.newaxis] * x
            
            r = r_new
            p = p_new
            rs = rs_new
        
        if not converged and np.linalg.norm(delta) < 1e-15:
            # Fall back to steepest descent
            delta = -g_t
        
        # Line search with backtracking
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(30):
            candidate = _retract(x, delta, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            if new_f <= f + 1e-8 * step_size * np.sum(g_t * delta):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        x = new_x
        f = new_f
        g = new_g
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
        
        # Recompute Hessian data
        W, V, diff, dist2 = _hessian_and_rhessian(x)
    
    return x, f

def _rcg_polish(pts, deadline, max_iter=500):
    """
    Riemannian conjugate gradient polish with Polak-Ribiere coefficient.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    p = -g_t
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        g_norm = np.linalg.norm(g_t)
        if g_norm < 1e-14:
            break
        
        # Line search
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(50):
            candidate = _retract(x, p, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            if new_f <= f + 1e-5 * step_size * np.sum(g_t * p):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        # Polak-Ribiere beta
        g_dot_new_g = np.sum(g_t * new_g_t)
        g_dot_g = np.sum(g_t * g_t)
        if g_dot_g > 1e-30:
            beta = max(0.0, g_dot_new_g / g_dot_g)
        else:
            beta = 0.0
        
        p_new = -new_g_t + beta * p
        # Project to tangent space
        pn_proj = np.einsum('ij,ij->i', p_new, new_x)
        p_new = p_new - pn_proj[:, np.newaxis] * new_x
        
        # Restart if direction not descent
        if np.sum(new_g_t * p_new) > 0:
            p_new = -new_g_t
        
        x = new_x
        f = new_f
        g = new_g
        g_t = new_g_t
        p = p_new
    
    return x, f

def _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-12, deadline=None, m=25):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
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
        
        # Line search with backtracking
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

def _deduplicate(pts_list, threshold=0.3):
    """Remove initializations that are too similar to already-accepted ones."""
    accepted = []
    for pts in pts_list:
        is_dup = False
        for acc in accepted:
            # Compute mean pairwise distance similarity
            diffs = pts[:, np.newaxis, :] - acc[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=2))
            # If the mean distance between corresponding points is small, they're similar
            mean_dist = np.mean(dists)
            if mean_dist < threshold:
                is_dup = True
                break
        if not is_dup:
            accepted.append(pts)
    return accepted

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci lattice)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(pts_fib)
    
    # 2. Multiple random starts
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]:
        np.random.seed(seed)
        pts_rand = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        pts_rand = pts_rand / norms
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci with various scales
    for seed in [11, 22, 33, 44, 55]:
        np.random.seed(seed)
        noise_scale = 0.02 + 0.03 * (seed // 11)
        noise = np.random.randn(N, 3) * noise_scale
        pts_pert = pts_fib + noise
        norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
        pts_pert = pts_pert / norms
        inits.append(pts_pert)
    
    # 4. Icosahedral base
    def icosahedron():
        t = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        return verts / norms
    
    ico = icosahedron()
    
    # 5. Band-based structured starts
    np.random.seed(777)
    for n_bands in [8, 10, 12, 14, 16]:
        pts_bands = []
        for band in range(n_bands):
            z_b = 1.0 - 2.0 * (band + 0.5) / n_bands
            r_b = np.sqrt(max(1.0 - z_b*z_b, 0.0))
            n_pts_in_band = N // n_bands
            if band < N % n_bands:
                n_pts_in_band += 1
            phis = 2.0 * np.pi * np.arange(n_pts_in_band) / n_pts_in_band + band * 0.3
            for ph in phis:
                pts_bands.append([r_b * np.cos(ph), r_b * np.sin(ph), z_b])
        pts_bands = np.array(pts_bands[:N])
        if len(pts_bands) == N:
            inits.append(pts_bands)
    
    # 6. Concentric rings with random offsets
    for seed in [100, 200, 300]:
        np.random.seed(seed)
        n_rings = 8
        pts_rings = []
        for ring in range(n_rings):
            z_r = 1.0 - 2.0 * (ring + 0.5) / n_rings
            r_r = np.sqrt(max(1.0 - z_r*z_r, 0.0))
            n_in_ring = N // n_rings
            if ring < N % n_rings:
                n_in_ring += 1
            offset = np.random.uniform(0, 2*np.pi)
            phis = 2.0 * np.pi * np.arange(n_in_ring) / n_in_ring + offset
            for ph in phis:
                pts_rings.append([r_r * np.cos(ph), r_r * np.sin(ph), z_r])
        pts_rings = np.array(pts_rings[:N])
        if len(pts_rings) == N:
            inits.append(pts_rings)
    
    # Deduplicate similar initializations
    inits = _deduplicate(inits, threshold=0.3)
    
    # Optimize each initialization
    n_inits = len(inits)
    # Reserve time for final polish stages
    polish_time = 30.0
    time_per_init = (145.0 - polish_time) / max(n_inits, 1)
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - polish_time:
            break
        
        init_deadline = min(deadline - polish_time, time.time() + time_per_init)
        
        # Alternate between optimizers for diversity
        if idx % 4 == 0:
            pts_opt, energy = _lbfgs_sphere_custom(init_pts, max_iter=500, tol=1e-12, deadline=init_deadline, m=25)
        elif idx % 4 == 1:
            x0 = init_pts.flatten()
            x_opt, energy = _lbfgs_sphere(x0, init_deadline, maxiter=3000)
            pts_opt = x_opt.reshape(N, 3)
        elif idx % 4 == 2:
            pts_opt, energy = _polish(init_pts, init_deadline, max_iter=200)
        else:
            pts_opt, energy = _rcg_polish(init_pts, init_deadline, max_iter=300)
        
        # Ensure points are on the sphere
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        pts_opt = pts_opt / np.maximum(norms, 1e-15)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    if best_pts is None:
        best_pts = pts_fib
        best_energy, _ = _energy_and_grad(best_pts)
    
    # Stage 1: Standard polish
    if time.time() < deadline - 15.0:
        best_pts, best_energy = _polish(best_pts, deadline, max_iter=500)
    
    # Stage 2: Custom L-BFGS from the polished result
    if time.time() < deadline - 12.0:
        stage2_deadline = min(deadline - 12.0, time.time() + 10.0)
        best_pts, best_energy = _lbfgs_sphere_custom(best_pts, max_iter=300, tol=1e-13, deadline=stage2_deadline, m=25)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / np.maximum(norms, 1e-15)
    
    # Stage 3: Riemannian CG for further refinement
    if time.time() < deadline - 8.0:
        stage3_deadline = min(deadline - 8.0, time.time() + 8.0)
        best_pts, best_energy = _rcg_polish(best_pts, stage3_deadline, max_iter=300)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / np.maximum(norms, 1e-15)
    
    # Stage 4: Newton polish for high-precision convergence
    if time.time() < deadline - 4.0:
        stage4_deadline = min(deadline - 4.0, time.time() + 10.0)
        best_pts, best_energy = _newton_polish(best_pts, stage4_deadline, max_iter=80)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / np.maximum(norms, 1e-15)
    
    # Stage 5: Ultra-precise final polish
    if time.time() < deadline - 2.0:
        best_pts, best_energy = _ultra_polish(best_pts, deadline, max_iter=2000)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)
    
    return best_pts
# EVOLVE-BLOCK-END
