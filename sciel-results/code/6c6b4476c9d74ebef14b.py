"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Compute energy with math.fsum
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    return energy, grads_tangent.flatten()


def _riemannian_cg(points, max_iter=5000, time_limit=20.0):
    """Riemannian conjugate gradient with exponential map updates."""
    start = time.time()
    pts = points.copy()
    n = len(pts)
    
    def _compute_grad(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(d_sq, 1.0)
        w = np.zeros_like(d_sq)
        mask = ~np.eye(n, dtype=bool)
        w[mask] = 1.0 / d_sq[mask]
        sw = np.sum(w, axis=1)
        ws = w @ p
        g = ws - p * sw[:, np.newaxis]
        dot = np.sum(g * p, axis=1, keepdims=True)
        return g - dot * p
    
    g = _compute_grad(pts)
    d = -g.copy()
    beta = 0.0
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Line search with backtracking
        step = 1.0
        e_curr = _log_energy(pts)
        
        for ls in range(30):
            # Exponential map: move along tangent vector d
            norm_d = np.linalg.norm(d, axis=1, keepdims=True)
            norm_d = np.maximum(norm_d, 1e-15)
            theta = step * norm_d
            sin_theta = np.sin(theta) / norm_d
            cos_theta = np.cos(theta)
            new_pts = pts * cos_theta + d * sin_theta
            new_pts = _project_to_sphere(new_pts)
            
            e_new = _log_energy(new_pts)
            if e_new < e_curr - 1e-4 * step * np.linalg.norm(d):
                break
            step *= 0.5
        else:
            break
        
        pts = new_pts
        e_curr = e_new
        
        if e_curr < best_e:
            best_e = e_curr
            best_pts = pts.copy()
        
        g_new = _compute_grad(pts)
        grad_norm = np.linalg.norm(g_new)
        if grad_norm < 1e-12:
            break
        
        # Polak-Ribiere beta
        beta = max(0.0, np.sum(g_new * (g_new - g)) / max(np.sum(g * g), 1e-15))
        d = -g_new + beta * d
        g = g_new
    
    return best_pts


def _polish(points, max_iter=5000, ftol=1e-14, gtol=1e-10, time_limit=30.0):
    """Polish a point configuration using L-BFGS-B with tight tolerances."""
    start = time.time()
    x0 = points.flatten()
    e0 = _log_energy(points)
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if res.success or res.fun < e0:
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < e0 and time.time() - start < time_limit:
                return p_res
    except Exception:
        pass
    return points


def _gradient_descent_sphere(points, lr=0.01, max_iter=10000, time_limit=20.0):
    """Gradient descent on the sphere with adaptive step size."""
    start = time.time()
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Compute gradient
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ pts
        grads = weighted_sum - pts * sum_weights[:, np.newaxis]
        
        # Project to tangent
        dot_products = np.sum(grads * pts, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * pts
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-12:
            break
        
        step = lr / (1.0 + grad_norm * 0.1)
        new_pts = pts - step * grads_tangent
        new_pts = _project_to_sphere(new_pts)
        
        e_new = _log_energy(new_pts)
        if e_new < best_e:
            best_e = e_new
            best_pts = new_pts.copy()
            pts = new_pts
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.9
    
    return best_pts


def _subdivided_icosahedron(n_target=150):
    """Generate points from a subdivided icosahedron for better initial distribution."""
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of icosahedron
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges (faces)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each face
    def midpoint_on_sphere(p1, p2):
        m = (p1 + p2) / 2.0
        return m / np.linalg.norm(m)
    
    # Store midpoints to avoid duplicates
    mid_cache = {}
    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key not in mid_cache:
            mid_cache[key] = midpoint_on_sphere(verts[i], verts[j])
        return mid_cache[key]
    
    # Subdivide once: each triangle becomes 4
    new_verts = list(verts)
    new_faces = []
    
    for face in faces:
        a, b, c = face
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        
        idx_ab = len(new_verts)
        new_verts.append(ab)
        idx_bc = len(new_verts)
        new_verts.append(bc)
        idx_ca = len(new_verts)
        new_verts.append(ca)
        
        new_faces.append([a, idx_ab, idx_ca])
        new_faces.append([idx_ab, b, idx_bc])
        new_faces.append([idx_ca, idx_bc, c])
        new_faces.append([idx_ab, idx_bc, idx_ca])
    
    new_verts = np.array(new_verts)
    new_verts = new_verts / np.linalg.norm(new_verts, axis=1, keepdims=True)
    
    # If we have more points than needed, randomly sample
    if len(new_verts) >= n_target:
        return new_verts[:n_target]
    
    # If too few, add random points
    while len(new_verts) < n_target:
        rng = np.random.RandomState(len(new_verts))
        p = rng.randn(3)
        new_verts = np.vstack([new_verts, p / np.linalg.norm(p)])
    
    return new_verts


def _simulated_annealing(points, T0=0.1, T_end=0.001, max_iter=5000, time_limit=25.0):
    """Simulated annealing to escape local minima."""
    start = time.time()
    rng = np.random.RandomState(12345)
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    current_e = best_e
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Cooling schedule
        alpha = it / max_iter
        T = T0 * (T_end / T0) ** alpha
        
        # Random perturbation - move 1-3 points
        n_move = rng.randint(1, 4)
        new_pts = pts.copy()
        for _ in range(n_move):
            idx = rng.randint(0, N)
            noise = rng.randn(3) * 0.08
            new_pts[idx] = pts[idx] + noise
        new_pts = _project_to_sphere(new_pts)
        
        e_new = _log_energy(new_pts)
        delta = e_new - current_e
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_e = e_new
            if e_new < best_e:
                best_e = e_new
                best_pts = new_pts.copy()
    
    return best_pts


def _multi_start_optimize(points, time_limit=30.0):
    """Try multiple perturbation strategies and pick the best."""
    start = time.time()
    best_pts = points.copy()
    best_e = _log_energy(best_pts)
    
    # Try different perturbation scales with CG refinement
    for scale in [0.05, 0.1, 0.15, 0.25]:
        remaining = time_limit - (time.time() - start)
        if remaining < 8.0:
            break
        
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        perturbed = points + rng.randn(N, 3) * scale
        perturbed = _project_to_sphere(perturbed)
        
        # Use Riemannian CG for faster convergence
        cg_result = _riemannian_cg(perturbed, max_iter=3000, 
                                    time_limit=remaining - 5.0)
        e_cg = _log_energy(cg_result)
        
        if e_cg < best_e:
            best_e = e_cg
            best_pts = cg_result.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations - diverse starting points
    inits = [
        _fibonacci_sphere(N),
        _subdivided_icosahedron(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=7),
        _random_uniform_sphere(N, seed=13),
        _random_uniform_sphere(N, seed=42),
        _random_uniform_sphere(N, seed=99),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # First pass: moderate optimization on all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 55.0:
            break
        
        try:
            res = minimize(
                _energy_and_grad,
                init.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'maxfun': 20000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Second pass: simulated annealing to escape local minima
    if time.time() < deadline - 40.0:
        sa_result = _simulated_annealing(best_points, T0=0.1, T_end=0.001, 
                                          max_iter=5000, 
                                          time_limit=deadline - time.time() - 35.0)
        e_sa = _log_energy(sa_result)
        if e_sa < best_energy:
            best_energy = e_sa
            best_points = sa_result.copy()
    
    # Third pass: Riemannian CG refinement
    if time.time() < deadline - 25.0:
        cg_result = _riemannian_cg(best_points, max_iter=5000,
                                    time_limit=deadline - time.time() - 20.0)
        e_cg = _log_energy(cg_result)
        if e_cg < best_energy:
            best_energy = e_cg
            best_points = cg_result.copy()
    
    # Fourth pass: aggressive L-BFGS-B polishing
    if time.time() < deadline - 15.0:
        polished = _polish(best_points, max_iter=8000, ftol=1e-14, gtol=1e-10,
                           time_limit=deadline - time.time() - 10.0)
        e_polish = _log_energy(polished)
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = polished.copy()
    
    # Fifth pass: multi-start with perturbations if time remains
    if time.time() < deadline - 10.0:
        multi_result = _multi_start_optimize(best_points, 
                                              time_limit=deadline - time.time() - 5.0)
        e_multi = _log_energy(multi_result)
        if e_multi < best_energy:
            best_energy = e_multi
            best_points = multi_result.copy()
    
    # Final ultra-tight polish with CG + L-BFGS-B
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 10.0:
            cg_final = _riemannian_cg(best_points, max_iter=5000, time_limit=remaining / 2.0)
            e_cg_f = _log_energy(cg_final)
            if e_cg_f < best_energy:
                best_energy = e_cg_f
                best_points = cg_final.copy()
        
        if time.time() < deadline - 3.0:
            final_polished = _polish(best_points, max_iter=10000, ftol=1e-15, gtol=1e-12,
                                     time_limit=deadline - time.time() - 2.0)
            e_final = _log_energy(final_polished)
            if e_final < best_energy:
                best_points = final_polished.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < 1e-18:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
