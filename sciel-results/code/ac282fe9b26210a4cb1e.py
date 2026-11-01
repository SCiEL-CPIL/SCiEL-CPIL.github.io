import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill
EPS = 1e-15


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, EPS)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)


def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, EPS)
    
    dist = np.sqrt(d2)
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    energy = -0.5 * np.sum(log_dist)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    weight_sums = np.sum(inv_d2, axis=1, keepdims=True)
    weighted_pts = inv_d2 @ pts
    
    grad = weighted_pts - weight_sums * pts
    
    return energy, grad


def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return pts / norms


def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    return _project_to_sphere(new_pts)


def _fibonacci_spiral(n, offset=0.5):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _random_points(n, seed=42):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return _project_to_sphere(pts)


def _icosahedron_subdivide(n, seed=42):
    """Generate points by subdividing an icosahedron."""
    rng = np.random.RandomState(seed)
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
    
    pts = [v.copy() for v in verts]
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            edge = (min(a, b), max(a, b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1, 3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1, 3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_points(n - len(pts), seed=seed + 1000)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)


def _riemannian_descent(pts, max_iter=800, tol=1e-12, deadline=None):
    """
    Riemannian Gradient Descent with Backtracking Line Search and momentum.
    """
    pts = _project_to_sphere(pts.copy())
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    momentum = np.zeros_like(pts)
    beta = 0.9
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Momentum update
        momentum = beta * momentum + (1 - beta) * g_t
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(30):
            new_pts = _retract(pts, -alpha * momentum)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(momentum * momentum):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-14:
                break
            continue
        
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        lr = max(0.001, lr * 0.998)
    
    return best_pts, best_energy


def _polish_lbfgs(pts, max_iter=500, gtol=1e-14, deadline=None):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        if deadline is not None and time.time() > deadline:
            return pts, _log_energy(pts)
        
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)


def _polish_bfgs(pts, max_iter=300, gtol=1e-14, deadline=None):
    """Polish using BFGS with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        if deadline is not None and time.time() > deadline:
            return pts, _log_energy(pts)
        
        res = minimize(fun_and_grad, x0, jac=True, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)


def _final_polish(pts, deadline=None):
    """Final high-precision polish with adaptive Newton-style refinement."""
    pts = _project_to_sphere(pts.copy())
    energy = _log_energy(pts)
    
    for _ in range(200):
        if deadline is not None and time.time() > deadline:
            break
        e, g = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, g)
        gnorm = np.linalg.norm(g_t)
        if gnorm < 1e-15:
            break
        
        # Adaptive step: use gradient norm to set step size
        alpha = min(0.05, 0.5 / (gnorm + 1e-10))
        new_pts = _retract(pts, -alpha * g_t)
        new_e = _log_energy(new_pts)
        if new_e < e:
            pts = new_pts
            energy = new_e
        else:
            alpha *= 0.5
            new_pts = _retract(pts, -alpha * g_t)
            new_e = _log_energy(new_pts)
            if new_e < e:
                pts = new_pts
                energy = new_e
    
    return pts, energy


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral with different offsets
    for offset in [0.3, 0.5, 0.7]:
        inits.append((f'fib_{offset}', _fibonacci_spiral(N, offset)))
    
    # 2. Golden spiral variants
    for offset in [0.0, 0.25, 0.5, 0.75]:
        inits.append((f'golden_{offset}', _golden_spiral_v2(N, offset)))
    
    # 3. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 4. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.05, 10), (0.1, 11), (0.15, 12), (0.2, 13), (0.3, 14)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    # 5. Icosahedron subdivision
    inits.append(('ico', _icosahedron_subdivide(N, seed=42)))
    inits.append(('ico_pert', _perturbed_points(_icosahedron_subdivide(N, seed=42), 0.1, seed=50)))
    
    # 6. Perturbed random
    rand_base = _random_points(N, seed=100)
    for sigma, seed in [(0.1, 20), (0.2, 21)]:
        inits.append((f'randpert{sigma}_{seed}', _perturbed_points(rand_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE_SECONDS - 25:
            break
        
        # Stage 1: Riemannian Gradient Descent with momentum
        rd_deadline = min(deadline - 15, t_start + elapsed + 10)
        pts_opt, energy_opt = _riemannian_descent(init_pts, max_iter=800, tol=1e-11,
                                                  deadline=rd_deadline)
        
        # Stage 2: L-BFGS-B Polish
        if time.time() < deadline - 12:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=300, gtol=1e-13,
                                                deadline=deadline - 10)
        
        # Stage 3: BFGS Polish for additional refinement
        if time.time() < deadline - 8:
            pts_opt, energy_opt = _polish_bfgs(pts_opt, max_iter=200, gtol=1e-14,
                                               deadline=deadline - 6)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=1000, gtol=1e-15,
                                              deadline=deadline - 3)
        
        # Additional Newton-style refinement
        if time.time() < deadline - 2:
            best_pts, best_energy = _final_polish(best_pts, deadline=deadline - 1)
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = _project_to_sphere(best_pts)
    return best_pts
