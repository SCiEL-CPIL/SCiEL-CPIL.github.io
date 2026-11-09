"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    d_sq = np.maximum(d_sq, 1e-30)
    return math.fsum((-0.5 * np.log(d_sq)).tolist())

def _energy_and_grad(pts):
    """Compute energy and gradient of logarithmic energy."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = math.fsum((-0.5 * np.log(d_sq)).tolist())
    
    grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere."""
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts

def _fibonacci_initialization():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)

def _random_initialization(seed):
    """Generate random points on sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (N, 3))
    return _project_to_sphere(pts)

def _perturbed_fibonacci(seed, scale=0.05):
    """Fibonacci with random perturbation."""
    pts0 = _fibonacci_initialization()
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, scale, (N, 3))
    return _project_to_sphere(pts0 + jitter)

def _antipodal_pairs_initialization(seed):
    """Initialize with pairs of nearly antipodal points."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N // 2):
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p)
        # Add small perturbation to avoid exact antipodal
        q = -p + rng.normal(0, 0.01, 3)
        q = q / np.linalg.norm(q)
        pts.append(p)
        pts.append(q)
    if N % 2 == 1:
        p = rng.normal(0, 1, 3)
        pts.append(p / np.linalg.norm(p))
    return np.array(pts)

def _lattice_initialization():
    """Generate points using a lattice-like structure on sphere."""
    pts = []
    # Use multiple rings with hexagonal-like spacing
    n_rings = 13
    for ring in range(n_rings):
        z = 1.0 - 2.0 * (ring + 0.5) / n_rings
        r = np.sqrt(max(0, 1.0 - z * z))
        if r < 1e-10:
            pts.append([0, 0, z])
            continue
        n_points = max(1, int(2 * np.pi * r / 0.14))
        for k in range(n_points):
            phi = 2 * np.pi * k / n_points + ring * 0.15
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            pts.append([x, y, z])
    
    while len(pts) < N:
        rng = np.random.default_rng(len(pts) + 1000)
        p = rng.normal(0, 1, 3)
        pts.append(p / np.linalg.norm(p))
    pts = np.array(pts[:N])
    return _project_to_sphere(pts)

def _lgbfgs_optimize(pts0, max_iter=500, ftol=1e-13, gtol=1e-11):
    """Optimize using L-BFGS-B with sphere projection."""
    def obj_and_grad(x):
        pts = _project_to_sphere(x)
        energy, grad = _energy_and_grad(pts)
        grad_t = _tangent_project(pts, grad)
        return energy, grad_t
    
    result = minimize(
        obj_and_grad,
        pts0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    
    if result.fun < 1e10:
        pts_opt = _project_to_sphere(result.x)
        energy, _ = _energy_and_grad(pts_opt)
        return pts_opt, energy
    return pts0, _log_energy(pts0)

def _newton_polish(pts0, max_iter=200, tol=1e-12):
    """Newton's method with line search on the sphere."""
    pts = pts0.copy()
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        grad_t = _tangent_project(pts, grad)
        
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < tol:
            break
        
        direction = -grad_t
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-15:
            break
        
        # Backtracking line search with Armijo condition
        c1 = 1e-4
        rho = 0.5
        alpha = 1.0
        for _ in range(30):
            new_pts = pts + alpha * direction
            new_pts = _project_to_sphere(new_pts)
            e_new, _ = _energy_and_grad(new_pts)
            if e_new <= energy + c1 * alpha * np.dot(grad_t.ravel(), direction.ravel()):
                pts = new_pts
                break
            alpha *= rho
        
        if alpha < 1e-10:
            break
    
    return pts, _log_energy(pts)

def _conjugate_gradient_polish(pts0, max_iter=100, tol=1e-12):
    """Conjugate gradient method on the sphere."""
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    grad_t = _tangent_project(pts, grad)
    
    direction = -grad_t
    for it in range(max_iter):
        if np.linalg.norm(grad_t) < tol:
            break
        
        # Line search
        c1 = 1e-4
        rho = 0.5
        alpha = 1.0
        for _ in range(30):
            new_pts = pts + alpha * direction
            new_pts = _project_to_sphere(new_pts)
            e_new, grad_new = _energy_and_grad(new_pts)
            if e_new <= energy + c1 * alpha * np.dot(grad_t.ravel(), direction.ravel()):
                break
            alpha *= rho
        else:
            break
        
        grad_new_t = _tangent_project(new_pts, grad_new)
        
        # Fletcher-Reeves beta
        grad_norm_sq = np.dot(grad_t.ravel(), grad_t.ravel())
        if grad_norm_sq < 1e-30:
            break
        beta = np.dot(grad_new_t.ravel(), grad_new_t.ravel()) / grad_norm_sq
        direction = -grad_new_t + beta * direction
        
        pts = new_pts
        energy = e_new
        grad = grad_new
        grad_t = grad_new_t
    
    return pts, _log_energy(pts)

def _basin_hopping(pts0, n_iter=10, rng_seed=42, deadline=None):
    """Basin hopping with time check."""
    rng = np.random.default_rng(rng_seed)
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    for _ in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        scale = 0.08
        jitter = rng.normal(0, scale, (N, 3))
        perturbed = _project_to_sphere(best_pts + jitter)
        
        pts_opt, energy = _lgbfgs_optimize(perturbed, max_iter=150, ftol=1e-11, gtol=1e-9)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci
    initializations.append(_fibonacci_initialization())
    
    # Perturbed Fibonacci with various scales
    for seed in range(1, 9):
        if time.time() > deadline - 40:
            break
        scale = 0.02 + 0.015 * seed
        initializations.append(_perturbed_fibonacci(seed, scale))
    
    # Random initializations
    for seed in range(100, 108):
        if time.time() > deadline - 40:
            break
        initializations.append(_random_initialization(seed))
    
    # Antipodal pairs
    if time.time() < deadline - 40:
        initializations.append(_antipodal_pairs_initialization(42))
    
    # Lattice
    if time.time() < deadline - 40:
        initializations.append(_lattice_initialization())
    
    # Optimize each initialization
    best_pts = initializations[0].copy()
    best_energy = _log_energy(best_pts)
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        try:
            # L-BFGS optimization
            pts_opt, energy = _lgbfgs_optimize(init_pts, max_iter=400, ftol=1e-12, gtol=1e-10)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts_opt
            
            # Newton polish if we have time
            if time.time() < deadline - 10:
                pts_polished, energy_polished = _newton_polish(pts_opt, max_iter=100, tol=1e-11)
                if energy_polished < best_energy:
                    best_energy = energy_polished
                    best_pts = pts_polished
                    
        except Exception:
            continue
    
    # Basin hopping from best point
    if time.time() < deadline - 25:
        try:
            pts_bh, energy_bh = _basin_hopping(best_pts, n_iter=8, rng_seed=123, deadline=deadline - 15)
            if energy_bh < best_energy:
                best_energy = energy_bh
                best_pts = pts_bh
        except Exception:
            pass
    
    # Final polish with Newton
    if time.time() < deadline - 8:
        try:
            pts_final, energy_final = _newton_polish(best_pts, max_iter=150, tol=1e-12)
            if energy_final < best_energy:
                best_energy = energy_final
                best_pts = pts_final
        except Exception:
            pass
    
    # CG polish if time permits
    if time.time() < deadline - 4:
        try:
            pts_cg, energy_cg = _conjugate_gradient_polish(best_pts, max_iter=50, tol=1e-12)
            if energy_cg < best_energy:
                best_energy = energy_cg
                best_pts = pts_cg
        except Exception:
            pass
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        best_pts = _fibonacci_initialization()
    
    return best_pts
# EVOLVE-BLOCK-END
