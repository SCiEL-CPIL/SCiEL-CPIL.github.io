"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on unit sphere."""
    n = len(pts)
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask out self-pairs (diagonal)
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use math.fsum for high precision matching evaluator
    triu_i, triu_j = np.triu_indices(n, k=1)
    d_sq = dist_sq[triu_i, triu_j]
    # Avoid log(0) by clamping
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    log_terms = -np.log(d)
    energy = math.fsum(log_terms.tolist())
    
    # Gradient: for point i, grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    inv_dist_sq = np.where(mask, 1.0 / dist_sq_safe, 0.0)
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _tangent_projection(pts, grad):
    """Project gradient to tangent space of the sphere at each point."""
    # For each point i, the tangent space is orthogonal to pts[i]
    # proj = grad - (grad . pts_i) * pts_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - dot * pts


def _lbfgs_on_sphere(pts, maxiter=2000, gtol=1e-12):
    """Run L-BFGS-B on the sphere using tangent space gradients."""
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _energy_and_grad(best_pts)[0]
    
    def objective(x):
        pts_x = _project_to_sphere(x.reshape(-1, 3))
        return _energy_and_grad(pts_x)[0]
    
    def gradient(x):
        pts_x = _project_to_sphere(x.reshape(-1, 3))
        e, g = _energy_and_grad(pts_x)
        g = _tangent_projection(pts_x, g)
        return g.flatten()
    
    x0 = best_pts.flatten()
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'gtol': gtol, 'maxfun': 50000})
        if res.fun < best_energy:
            new_pts = _project_to_sphere(res.x.reshape(-1, 3))
            new_energy = _energy_and_grad(new_pts)[0]
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except Exception:
        pass
    
    return best_pts, best_energy


def _conjugate_gradient_sphere(pts, maxiter=5000, tol=1e-12):
    """Conjugate gradient method on the sphere."""
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _energy_and_grad(best_pts)[0]
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    n = len(current_pts)
    
    # Initial gradient
    _, grad = _energy_and_grad(current_pts)
    grad = _tangent_projection(current_pts, grad)
    
    # Initial direction
    p = -grad.copy()
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-15:
        return best_pts, best_energy
    
    for it in range(maxiter):
        # Line search (backtracking)
        alpha = 0.01
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-15:
            break
        
        # Normalize step direction for line search
        p_dir = p / p_norm
        
        for ls in range(30):
            new_pts = _project_to_sphere(current_pts + alpha * p_dir)
            new_energy = _energy_and_grad(new_pts)[0]
            if new_energy < current_energy:
                break
            alpha *= 0.5
        else:
            break
        
        # Update
        current_pts = new_pts
        current_energy = new_energy
        
        if current_energy < best_energy:
            best_pts = current_pts.copy()
            best_energy = current_energy
        
        # Check convergence
        if alpha * p_norm < 1e-14:
            break
        
        # New gradient
        _, new_grad = _energy_and_grad(current_pts)
        new_grad = _tangent_projection(current_pts, new_grad)
        
        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < tol:
            break
        
        # Conjugate gradient update (Polak-Ribiere)
        old_grad = grad
        beta = np.sum(new_grad * (new_grad - old_grad)) / (np.sum(old_grad * old_grad) + 1e-30)
        beta = max(0, beta)
        p = -new_grad + beta * p
        
        grad = new_grad
    
    return best_pts, best_energy


def _simulated_annealing(pts, time_budget=15.0, T0=0.05, T1=1e-8, alpha=0.9995):
    """Simulated annealing on the sphere."""
    start_time = time.time()
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _energy_and_grad(best_pts)[0]
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    T = T0
    n = len(current_pts)
    rng = np.random.RandomState(12345)
    
    while time.time() - start_time < time_budget:
        # Generate a random perturbation
        idx = rng.randint(n)
        # Random direction in tangent space
        rand_dir = rng.randn(3)
        rand_dir -= np.dot(rand_dir, current_pts[idx]) * current_pts[idx]
        norm = np.linalg.norm(rand_dir)
        if norm > 1e-12:
            rand_dir /= norm
        
        # Step size proportional to temperature
        step_size = T * 0.05
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + step_size * rand_dir
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy_and_grad(new_pts)[0]
        delta_e = new_energy - current_energy
        
        # Accept or reject
        if delta_e < 0 or rng.rand() < np.exp(-delta_e / T):
            current_pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_pts = current_pts.copy()
                best_energy = current_energy
        
        T *= alpha
        if T < T1:
            break
    
    return best_pts, best_energy


def _local_search_restarts(pts, n_restarts=20, time_budget=10.0):
    """Multiple local search restarts with multi-point perturbations."""
    start_time = time.time()
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _energy_and_grad(best_pts)[0]
    
    n = len(pts)
    rng = np.random.RandomState(67890)
    
    for r in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Multi-point random perturbation: move 2-4 points simultaneously
        perturbed = pts.copy()
        n_moves = rng.randint(2, 5)  # 2 to 4 points
        move_scale = 0.005 + 0.015 * rng.rand()  # scale between 0.005 and 0.02
        
        for _ in range(n_moves):
            idx = rng.randint(n)
            rand_dir = rng.randn(3)
            rand_dir -= np.dot(rand_dir, perturbed[idx]) * perturbed[idx]
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-12:
                rand_dir /= norm
            perturbed[idx] += move_scale * rand_dir
        perturbed = _project_to_sphere(perturbed)
        
        # Optimize with tighter tolerance
        opt_pts, opt_energy = _lbfgs_on_sphere(perturbed, maxiter=800, gtol=1e-11)
        
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Phase 1: Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    for seed in range(1, 15):
        if time.time() > deadline - 40.0:
            break
        starts.append(_random_sphere(N, seed=seed))
    
    # Phase 2: Optimize each starting point with limited time
    best_pts = starts[0]
    best_energy = _energy_and_grad(best_pts)[0]
    
    time_per_start = max(4.0, (deadline - start_time - 30.0) / max(1, len(starts)))
    
    for i, s in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - 30.0 - time.time()
        if remaining < 3.0:
            break
        
        pts, energy = _lbfgs_on_sphere(s, maxiter=1000, gtol=1e-11)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Phase 3: Conjugate gradient polish
    if time.time() < deadline - 15.0:
        cg_budget = min(10.0, deadline - 15.0 - time.time())
        if cg_budget > 2.0:
            cg_pts, cg_energy = _conjugate_gradient_sphere(best_pts, maxiter=3000, tol=1e-13)
            if cg_energy < best_energy:
                best_pts = cg_pts
                best_energy = cg_energy
    
    # Phase 4: Simulated annealing from best point
    if time.time() < deadline - 12.0:
        sa_budget = min(6.0, deadline - 12.0 - time.time())
        if sa_budget > 2.0:
            sa_pts, sa_energy = _simulated_annealing(best_pts, time_budget=sa_budget)
            if sa_energy < best_energy:
                best_pts = sa_pts
                best_energy = sa_energy
    
    # Phase 5: Local search restarts
    if time.time() < deadline - 10.0:
        ls_budget = min(8.0, deadline - 10.0 - time.time())
        if ls_budget > 1.0:
            ls_pts, ls_energy = _local_search_restarts(best_pts, n_restarts=25, time_budget=ls_budget)
            if ls_energy < best_energy:
                best_pts = ls_pts
                best_energy = ls_energy
    
    # Phase 6: Final high-precision polish with CG + L-BFGS
    if time.time() < deadline - 1.0:
        polish_budget = deadline - 1.0 - time.time()
        if polish_budget > 1.0:
            # First a CG pass for very tight convergence
            if polish_budget > 3.0:
                cg_pts, cg_energy = _conjugate_gradient_sphere(best_pts, maxiter=5000, tol=1e-14)
                if cg_energy < best_energy:
                    best_pts = cg_pts
                    best_energy = cg_energy
            
            # Then final L-BFGS with very tight tolerance
            final_pts, final_energy = _lbfgs_on_sphere(best_pts, maxiter=10000, gtol=1e-15)
            if final_energy < best_energy:
                best_pts = final_pts
                best_energy = final_energy
    
    # Final validation: ensure all points are on unit sphere and non-degenerate
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
