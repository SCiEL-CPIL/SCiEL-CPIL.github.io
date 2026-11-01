"""Optimized logarithmic energy minimization for 100 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _energy_and_grad(pts):
    """Compute energy and Euclidean gradient."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(np.log(dist[iu]))
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    grad = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2
    return energy, grad

def _tangent_grad(pts):
    """Project gradient to tangent space."""
    grad = _energy_and_grad(pts)[1]
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - grad_dot_xi * pts

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _icosahedron_subdivide(n, rng):
    """Generate points by subdividing an icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    pts = []
    for v in verts:
        pts.append(v)
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a,b), max(a,b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1,3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1,3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_sphere(n - len(pts), rng)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)

def _riemannian_gradient_descent(pts0, max_iters=3000, tol=1e-13, time_limit=25.0):
    """Riemannian Gradient Descent with backtracking line search."""
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    g_t = _tangent_grad(pts)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.02
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        g_norm = np.linalg.norm(g_t)
        if g_norm < tol:
            break
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(40):
            new_pts = _project_to_sphere(pts + alpha * g_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * g_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-15:
                break
            continue
        
        _, _ = _energy_and_grad(pts)
        g_t = _tangent_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        lr = max(0.001, lr * 0.9998)
    
    return best_pts, best_energy

def _polish_lbfgs(pts, max_iter=800, gtol=1e-16, ftol=1e-18):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e = _log_energy(x_pts)
        g_t = _tangent_grad(x_pts)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)

def _polish_bfgs(pts, max_iter=300, gtol=1e-16):
    """Polish using BFGS with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e = _log_energy(x_pts)
        g_t = _tangent_grad(x_pts)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 50})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)

def _optimize_from_start(pts0, deadline):
    """Optimize from a starting configuration with multiple stages."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    # Stage 1: Riemannian Gradient Descent
    if time.time() < deadline - 20.0:
        rgd_pts, rgd_energy = _riemannian_gradient_descent(best_pts, max_iters=2500, time_limit=18.0)
        if rgd_energy < best_energy:
            best_pts = rgd_pts
            best_energy = rgd_energy
    
    # Stage 2: L-BFGS-B polish
    if time.time() < deadline - 10.0:
        lbfgs_pts, lbfgs_energy = _polish_lbfgs(best_pts, max_iter=500, gtol=1e-15)
        if lbfgs_energy < best_energy:
            best_pts = lbfgs_pts
            best_energy = lbfgs_energy
    
    # Stage 3: BFGS polish
    if time.time() < deadline - 5.0:
        bfgs_pts, bfgs_energy = _polish_bfgs(best_pts, max_iter=200, gtol=1e-15)
        if bfgs_energy < best_energy:
            best_pts = bfgs_pts
            best_energy = bfgs_energy
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron-based
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Random starts
    for seed_offset in range(8):
        rng_sub = np.random.default_rng(42 + seed_offset)
        initializations.append(_random_sphere(N, rng_sub))
    
    # Perturbed Fibonacci
    for sigma in [0.01, 0.03, 0.08, 0.15, 0.25, 0.4]:
        rng_sub = np.random.default_rng(1000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.02, 0.05, 0.1, 0.2]:
        rng_sub = np.random.default_rng(2000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng_sub), rng_sub, sigma))
    
    # Spherical grid (lat/lon)
    n_lat = 10
    n_lon = 10
    grid_pts = []
    for i in range(n_lat):
        theta = np.pi * i / (n_lat - 1)
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:N])
    initializations.append(_project_to_sphere(grid_pts))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    # Sort by initial energy to try good starts first
    init_energies = [_log_energy(p) for p in initializations]
    sorted_indices = np.argsort(init_energies)
    
    # Try initializations in order of initial energy
    for idx in sorted_indices:
        if time.time() > deadline - 20.0:
            break
        
        init_pts = initializations[idx]
        pts, energy = _optimize_from_start(init_pts, deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        final_pts, final_energy = _polish_lbfgs(best_pts, max_iter=800, gtol=1e-16, ftol=1e-18)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
        
        if time.time() < deadline - 2.0:
            final_pts2, final_energy2 = _polish_bfgs(best_pts, max_iter=300, gtol=1e-16)
            if final_energy2 < best_energy:
                best_pts = final_pts2
                best_energy = final_energy2
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
