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
    # log(1/d) = -0.5 * log(d_sq)
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
    
    # Energy: -0.5 * sum_{i != j} log(d_ij) = -sum_{i<j} log(d_ij)
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
    """Project gradient onto tangent space of sphere.
    For point p on unit sphere, tangent vector v must satisfy p . v = 0.
    Projected gradient = grad - (p . grad) * p
    """
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts

def _fibonacci_initialization():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)

def _sobol_like_initialization(seed):
    """Generate diverse quasi-random points using different seeds."""
    rng = np.random.default_rng(seed)
    # Use a mix of uniform and structured sampling
    pts = rng.normal(0, 1, (N, 3))
    pts = _project_to_sphere(pts)
    return pts

def _perturbed_fibonacci(seed, scale=0.05):
    """Fibonacci with random perturbation."""
    pts0 = _fibonacci_initialization()
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, scale, (N, 3))
    pts = _project_to_sphere(pts0 + jitter)
    return pts

def _hexagonal_grid_like():
    """Approximate hexagonal packing on sphere using rings."""
    pts = []
    # Multiple rings at different latitudes
    n_rings = 12
    for ring in range(n_rings):
        z = 1.0 - 2.0 * (ring + 0.5) / n_rings
        r = np.sqrt(max(0, 1.0 - z * z))
        n_points_ring = int(2 * np.pi * r / 0.15)  # spacing ~0.15
        n_points_ring = max(1, n_points_ring)
        for k in range(n_points_ring):
            phi = 2 * np.pi * k / n_points_ring + ring * 0.1  # offset
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            pts.append([x, y, z])
    
    # Pad or truncate to N
    while len(pts) < N:
        rng = np.random.default_rng(len(pts))
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p)
        pts.append(p)
    pts = np.array(pts[:N])
    return _project_to_sphere(pts)

def _lgbfgs_optimize(pts0, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Optimize using L-BFGS-B in Cartesian coordinates with sphere constraint.
    We use the trick of optimizing in R^3 and projecting to sphere.
    """
    def obj_and_grad(x):
        pts = _project_to_sphere(x)
        energy, grad = _energy_and_grad(pts)
        # Project gradient to tangent space
        grad_t = _tangent_project(pts, grad)
        return energy, grad_t
    
    result = minimize(
        obj_and_grad,
        pts0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    
    if result.fun < 1e10:  # valid result
        pts_opt = _project_to_sphere(result.x)
        energy, _ = _energy_and_grad(pts_opt)
        return pts_opt, energy
    return pts0, _log_energy(pts0)

def _newton_polish(pts0, max_iter=100):
    """Newton's method polish on the sphere.
    Solve the Hessian system in tangent space.
    """
    pts = pts0.copy()
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        grad_t = _tangent_project(pts, grad)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-10:
            break
        
        # Approximate Hessian via finite differences in tangent space
        # For efficiency, use a simplified approach: line search + gradient descent
        # with adaptive step size
        
        # Compute step direction: negative gradient in tangent space
        direction = -grad_t
        
        # Line search: find alpha that minimizes energy
        best_alpha = 1.0
        best_e = energy
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            new_pts = pts + alpha * direction
            new_pts = _project_to_sphere(new_pts)
            e_new, _ = _energy_and_grad(new_pts)
            if e_new < best_e:
                best_e = e_new
                best_alpha = alpha
                pts = new_pts
            else:
                break
        
        if best_alpha < 0.01:
            break
    
    return pts, _log_energy(pts)

def _basin_hopping(pts0, n_iter=10, rng_seed=42):
    """Simple basin hopping: perturb, optimize, keep best."""
    rng = np.random.default_rng(rng_seed)
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    for _ in range(n_iter):
        # Random perturbation
        scale = 0.1
        jitter = rng.normal(0, scale, (N, 3))
        perturbed = _project_to_sphere(best_pts + jitter)
        
        # Optimize from perturbed point
        pts_opt, energy = _lgbfgs_optimize(perturbed, max_iter=200, ftol=1e-10, gtol=1e-8)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_initialization()
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Generate diverse initializations
    initializations = [pts0]
    
    # Perturbed Fibonacci with different scales
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 30:
            break
        scale = 0.03 + 0.02 * seed
        pts_init = _perturbed_fibonacci(seed, scale)
        initializations.append(pts_init)
    
    # Sobol-like random initializations
    for seed in [100, 101, 102, 103, 104, 105]:
        if time.time() > deadline - 30:
            break
        pts_init = _sobol_like_initialization(seed)
        initializations.append(pts_init)
    
    # Hexagonal grid
    if time.time() < deadline - 30:
        pts_init = _hexagonal_grid_like()
        initializations.append(pts_init)
    
    # Optimize each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        try:
            # First pass: L-BFGS-B
            pts_opt, energy = _lgbfgs_optimize(init_pts, max_iter=300, ftol=1e-11, gtol=1e-9)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts_opt
            
            # Second pass: Newton polish if we have time
            if time.time() < deadline - 10:
                pts_polished, energy_polished = _newton_polish(pts_opt, max_iter=50)
                if energy_polished < best_energy:
                    best_energy = energy_polished
                    best_pts = pts_polished
                    
        except Exception:
            continue
    
    # Final basin hopping from best point if we have time
    if time.time() < deadline - 20:
        try:
            pts_bh, energy_bh = _basin_hopping(best_pts, n_iter=5, rng_seed=123)
            if energy_bh < best_energy:
                best_energy = energy_bh
                best_pts = pts_bh
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            pts_final, energy_final = _newton_polish(best_pts, max_iter=100)
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
        # Fallback to Fibonacci
        best_pts = _fibonacci_initialization()
    
    return best_pts
# EVOLVE-BLOCK-END
