"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())


def _energy_and_gradient(pts):
    """Compute energy and gradient of log energy w.r.t. points.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    energy = -math.fsum(np.log(dist[iu]).tolist())
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _lgbfgs_optimize(pts, deadline, maxiter=500):
    """Optimize using L-BFGS-B with spherical constraint via projection."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if time.time() > deadline:
            return pts, _log_energy(pts)
        opt_pts = _project_to_sphere(result.x.reshape(N, 3))
        opt_energy = _log_energy(opt_pts)
        return opt_pts, opt_energy
    except Exception:
        return pts, _log_energy(pts)


def _gradient_descent(pts, max_iter=5000, lr_init=0.01, deadline=None):
    """Gradient descent with adaptive learning rate and sphere projection."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    
    lr = lr_init
    patience = 0
    best_pts = pts.copy()
    best_energy = current_energy
    min_improvement = 1e-10
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, grad = _energy_and_gradient(pts)
        
        # Project gradient to tangent space for sphere constraint
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        new_pts = pts - lr * grad_tangent
        new_pts = _project_to_sphere(new_pts)
        
        min_dist = np.min(np.linalg.norm(new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :], axis=2) * (1 - np.eye(N)))
        
        if min_dist < 1e-8:
            lr *= 0.5
            patience += 1
            if patience > 50:
                break
            continue
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < current_energy - min_improvement:
            pts = new_pts
            current_energy = new_energy
            lr *= 1.05
            patience = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            lr *= 0.7
            patience += 1
            if patience > 100:
                break
    
    return best_pts, best_energy


def _simulated_annealing(pts, max_iter=3000, t_init=0.01, t_min=1e-6, deadline=None, rng=None):
    """Simulated annealing for escaping local minima."""
    if rng is None:
        rng = np.random.default_rng()
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        noise_scale = 0.01 * (1 + it / max_iter)
        new_pts = pts + noise_scale * rng.standard_normal((N, 3))
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_energy


def _high_precision_polish(pts, deadline):
    """Final high-precision polish with very tight tolerances."""
    pts = _project_to_sphere(pts.copy())
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
        )
        if time.time() > deadline:
            return pts, _log_energy(pts)
        opt_pts = _project_to_sphere(result.x.reshape(N, 3))
        opt_energy = _log_energy(opt_pts)
        return opt_pts, opt_energy
    except Exception:
        return pts, _log_energy(pts)


def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    fib_pts = _fibonacci_points(N)
    
    # Multi-restart with diverse initializations
    initializations = [fib_pts]
    
    for _ in range(4):
        initializations.append(_random_points(N, rng))
    
    for scale in [0.05, 0.1, 0.2]:
        perturbed = fib_pts + scale * rng.standard_normal((N, 3))
        initializations.append(_project_to_sphere(perturbed))
    
    for init_pts in initializations:
        if time.time() > deadline - 20.0:
            break
        
        # Phase 1: L-BFGS-B optimization (fast convergence)
        lgbfgs_pts, lgbfgs_energy = _lgbfgs_optimize(init_pts, deadline - 5.0, maxiter=300)
        
        if lgbfgs_energy < best_energy:
            best_energy = lgbfgs_energy
            best_pts = lgbfgs_pts.copy()
        
        # Phase 2: Simulated annealing to escape local minima
        if time.time() < deadline - 15.0:
            sa_pts, sa_energy = _simulated_annealing(lgbfgs_pts, max_iter=1500, t_init=0.005, t_min=1e-7, deadline=deadline - 10.0, rng=rng)
            
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
            
            # Phase 3: L-BFGS-B from SA result
            if time.time() < deadline - 10.0:
                lgbfgs2_pts, lgbfgs2_energy = _lgbfgs_optimize(sa_pts, deadline - 5.0, maxiter=200)
                
                if lgbfgs2_energy < best_energy:
                    best_energy = lgbfgs2_energy
                    best_pts = lgbfgs2_pts.copy()
    
    if best_pts is None:
        best_pts = fib_pts
    
    # Final high-precision polish
    if time.time() < deadline - 3.0:
        polished_pts, polished_energy = _high_precision_polish(best_pts, deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: check for very close points
    dists = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
