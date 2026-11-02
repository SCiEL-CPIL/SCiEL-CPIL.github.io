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

def _gradient_log_energy(pts):
    """Tangent-space gradient of logarithmic energy, optimized with matrix ops."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, EPS)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # grad of E = -sum_{i<j} log(d_ij) is: -sum_{j!=i} (x_i - x_j)/d^2
    grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
    
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
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
    return pts / np.maximum(norms, EPS)

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
    
    pts = list(verts)
    
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

def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _riemannian_descent(pts, max_iter=300, tol=1e-10, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_log_energy(pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        step = 1.0 / (1.0 + 0.5 * grad_norm)
        
        accepted = False
        for ls_iter in range(30):
            new_pts = pts - step * grad
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-4 * step * grad_norm * grad_norm:
                    accepted = True
                    break
                if new_f < f - 1e-8:
                    accepted = True
                    break
            step *= 0.7
        
        if accepted:
            pts = new_pts
            f = new_f
        else:
            break
    
    return pts, f

def _optimize_from_start(pts0, deadline):
    """Optimize from a starting configuration using L-BFGS-B first, then Riemannian descent."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_log_energy(pts)
        return g.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    # First pass: L-BFGS-B with moderate tolerance
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 1000, 'ftol': 1e-10}
        )
        if res.fun < best_energy:
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Second pass: Riemannian gradient descent for robustness
    if time.time() < deadline - 8.0:
        try:
            pts_rd, e_rd = _riemannian_descent(best_pts, max_iter=300, tol=1e-12, deadline=deadline - 8.0)
            if e_rd < best_energy:
                best_pts = pts_rd
                best_energy = e_rd
        except Exception:
            pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Third pass: L-BFGS-B with tighter tolerance
    try:
        x1 = best_pts.reshape(-1)
        res2 = minimize(
            objective,
            x1,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 800, 'gtol': 1e-12, 'maxfun': 1500, 'ftol': 1e-14}
        )
        if res2.fun < best_energy:
            pts_new = _project_to_sphere(res2.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(('fib', _fibonacci_sphere(N)))
    
    # 2. Golden spiral v2
    inits.append(('gold_v2', _golden_spiral_v2(N)))
    
    # 3. Icosahedron-based
    rng0 = np.random.RandomState(42)
    inits.append(('icosa', _icosahedron_subdivide(N, rng0)))
    
    # 4. Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        inits.append((f'rand{seed}', _random_sphere(N, rng)))
    
    # 5. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_sphere(N)
    for sigma, seed in [(0.1, 10), (0.05, 11), (0.2, 12), (0.03, 13)]:
        rng = np.random.RandomState(seed)
        inits.append((f'pert{sigma}_{seed}', _perturb_sphere(fib_base, rng, sigma)))
    
    # 6. Perturbed golden spiral
    gold_base = _golden_spiral_v2(N)
    for sigma, seed in [(0.1, 20), (0.05, 21)]:
        rng = np.random.RandomState(seed)
        inits.append((f'pert_gold{sigma}_{seed}', _perturb_sphere(gold_base, rng, sigma)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization with adaptive time allocation
    n_inits = len(inits)
    time_per_init = (DEADLINE_SECONDS - 30) / n_inits  # Reserve 30s for final polish
    
    for idx, (name, init_pts) in enumerate(inits):
        elapsed = time.time() - t_start
        remaining = deadline - time.time()
        
        if remaining < 10:
            break
        
        # Allocate more time to earlier, more promising initializations
        if idx < 3:
            time_budget = min(remaining - 10, time_per_init * 1.5)
        elif idx < 6:
            time_budget = min(remaining - 10, time_per_init)
        else:
            time_budget = min(remaining - 10, time_per_init * 0.7)
        
        init_deadline = time.time() + time_budget
        if init_deadline > deadline - 15:
            init_deadline = deadline - 15
        
        pts_opt, energy_opt = _optimize_from_start(init_pts, init_deadline)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        # Multiple polish passes with tightening tolerances
        for gtol in [1e-12, 1e-14, 1e-15]:
            if time.time() > deadline - 3:
                break
            try:
                def objective(x):
                    pts = _project_to_sphere(x.reshape(N, 3))
                    return _log_energy(pts)
                
                def gradient(x):
                    pts = _project_to_sphere(x.reshape(N, 3))
                    g = _gradient_log_energy(pts)
                    return g.reshape(-1)
                
                x0 = best_pts.reshape(-1)
                res = minimize(
                    objective,
                    x0,
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'gtol': gtol, 'maxfun': 2000, 'ftol': 1e-16}
                )
                if res.fun < best_energy:
                    pts_new = _project_to_sphere(res.x.reshape(N, 3))
                    e_new = _log_energy(pts_new)
                    if e_new < best_energy:
                        best_pts = pts_new
                        best_energy = e_new
            except Exception:
                pass
        
        # Final Riemannian descent pass for very tight convergence
        if time.time() < deadline - 3:
            try:
                pts_rd, e_rd = _riemannian_descent(best_pts, max_iter=500, tol=1e-14, deadline=deadline - 3)
                if e_rd < best_energy:
                    best_pts = pts_rd
                    best_energy = e_rd
            except Exception:
                pass
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = _project_to_sphere(best_pts)
    return best_pts
