"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    d_sq = np.maximum(d_sq, 1e-30)
    return math.fsum((-0.5 * np.log(d_sq)).tolist())

def _energy_and_grad(pts):
    """Compute energy and gradient of logarithmic energy.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = math.fsum((-0.5 * np.log(d_sq)).tolist())
    
    # Gradient
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
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (N, 3))
    return _project_to_sphere(pts)

def _perturbed_fibonacci(seed, scale=0.05):
    pts0 = _fibonacci_initialization()
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, scale, (N, 3))
    pts = _project_to_sphere(pts0 + jitter)
    return pts

def _lgbfgs_optimize(pts0, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Optimize using L-BFGS-B in Cartesian coordinates with sphere constraint."""
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

def _newton_polish(pts0, max_iter=200):
    """Newton's method polish on the sphere with adaptive line search."""
    pts = pts0.copy()
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        grad_t = _tangent_project(pts, grad)
        
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-12:
            break
        
        # Compute step direction: negative gradient in tangent space
        direction = -grad_t
        
        # Line search: find alpha that minimizes energy
        best_alpha = 1.0
        best_e = energy
        alphas = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
        for alpha in alphas:
            new_pts = pts + alpha * direction
            new_pts = _project_to_sphere(new_pts)
            e_new, _ = _energy_and_grad(new_pts)
            if e_new < best_e:
                best_e = e_new
                best_alpha = alpha
                pts = new_pts
            else:
                break
        
        if best_alpha < 0.001:
            break
    
    return pts, _log_energy(pts)

def _basin_hopping(pts0, n_iter=10, rng_seed=42, time_limit=30.0):
    """Basin hopping: perturb, optimize, keep best."""
    rng = np.random.default_rng(rng_seed)
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    start_time = time.time()
    
    for _ in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        scale = 0.08
        jitter = rng.normal(0, scale, (N, 3))
        perturbed = _project_to_sphere(best_pts + jitter)
        
        # Optimize from perturbed point
        pts_opt, energy = _lgbfgs_optimize(perturbed, max_iter=200, ftol=1e-10, gtol=1e-8)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    return best_pts, best_energy

def _differential_evolution_search(time_limit=45.0):
    """Global search using differential evolution in spherical coordinates."""
    def sphere_energy(theta_phi):
        theta = theta_phi[:N]
        phi = theta_phi[N:]
        pts = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        pts = _project_to_sphere(pts)
        return _log_energy(pts)
    
    # Define bounds for theta (0 to pi) and phi (0 to 2*pi)
    bounds = [(0, math.pi)] * N + [(0, 2 * math.pi)] * N
    
    start_time = time.time()
    
    result = differential_evolution(
        sphere_energy,
        bounds,
        maxiter=50,
        popsize=10,
        tol=1e-8,
        seed=42,
        polish=False,
        init='sobol'
    )
    
    if result.fun < 1e10:
        theta_phi = result.x
        theta = theta_phi[:N]
        phi = theta_phi[N:]
        pts = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        pts = _project_to_sphere(pts)
        # Polish with L-BFGS
        pts_opt, energy = _lgbfgs_optimize(pts, max_iter=300, ftol=1e-11, gtol=1e-9)
        return pts_opt, energy
    
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_initialization()
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Generate diverse initializations
    initializations = [pts0]
    
    # Perturbed Fibonacci with different scales
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 40:
            break
        scale = 0.03 + 0.02 * seed
        pts_init = _perturbed_fibonacci(seed, scale)
        initializations.append(pts_init)
    
    # Random initializations
    for seed in [100, 101, 102]:
        if time.time() > deadline - 40:
            break
        pts_init = _random_initialization(seed)
        initializations.append(pts_init)
    
    # Optimize each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        try:
            # L-BFGS-B
            pts_opt, energy = _lgbfgs_optimize(init_pts, max_iter=300, ftol=1e-11, gtol=1e-9)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts_opt
            
            # Newton polish
            if time.time() < deadline - 15:
                pts_polished, energy_polished = _newton_polish(pts_opt, max_iter=100)
                if energy_polished < best_energy:
                    best_energy = energy_polished
                    best_pts = pts_polished
                    
        except Exception:
            continue
    
    # Basin hopping from best point if we have time
    if time.time() < deadline - 30:
        try:
            time_left = deadline - 30 - time.time()
            pts_bh, energy_bh = _basin_hopping(best_pts, n_iter=8, rng_seed=123, time_limit=time_left)
            if energy_bh < best_energy:
                best_energy = energy_bh
                best_pts = pts_bh
        except Exception:
            pass
    
    # Differential evolution global search if we have time
    if time.time() < deadline - 20:
        try:
            time_left = deadline - 20 - time.time()
            pts_de, energy_de = _differential_evolution_search(time_limit=time_left)
            if pts_de is not None and energy_de < best_energy:
                best_energy = energy_de
                best_pts = pts_de
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            pts_final, energy_final = _newton_polish(best_pts, max_iter=200)
            if energy_final < best_energy:
                best_energy = energy_final
                best_pts = pts_final
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere
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
