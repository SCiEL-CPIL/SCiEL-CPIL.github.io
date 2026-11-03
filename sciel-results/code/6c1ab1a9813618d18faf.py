"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=None):
    """Generate n uniformly random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _energy_and_grad(x):
    """Compute energy and gradient.
    x is a 2D array of shape (N, 3) of points on the unit sphere.
    Returns (energy, grad) where grad is a 2D array of shape (N, 3).
    """
    x = np.asarray(x, dtype=float)
    
    # Compute pairwise squared distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(sq_dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(N, k=1)
    log_dists = 0.5 * np.log(sq_dists[iu[0], iu[1]])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_sq_dists = 1.0 / sq_dists
    np.fill_diagonal(inv_sq_dists, 0.0)
    grad = np.sum(diff * inv_sq_dists[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _tangent_project(grad, x):
    """Project gradient to tangent space of the sphere."""
    return grad - np.sum(grad * x, axis=1, keepdims=True) * x


def _lbfgs_optimize(x0, maxiter=500, ftol=1e-13, gtol=1e-11, deadline=None):
    """Run L-BFGS-B optimization on the sphere using reparameterization."""
    
    def objective_and_grad(flat_x):
        x = flat_x.reshape(N, 3)
        x = _project_to_sphere(x)
        e, g = _energy_and_grad(x)
        grad_t = _tangent_project(g, x)
        return e, grad_t.flatten()
    
    result = minimize(
        objective_and_grad,
        x0.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 100}
    )
    
    x_opt = result.x.reshape(N, 3)
    x_opt = _project_to_sphere(x_opt)
    e_opt, _ = _energy_and_grad(x_opt)
    
    return x_opt, e_opt


def _gradient_descent_polish(x, max_iter=5000, tol=1e-14, deadline=None):
    """Polish with gradient descent on the sphere using adaptive step size and backtracking."""
    x = x.copy()
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e, _ = _energy_and_grad(x)
    
    # Compute initial gradient
    _, g = _energy_and_grad(x)
    grad_t = _tangent_project(g, x)
    
    # Adaptive step size with backtracking line search
    alpha = 0.05
    min_alpha = 1e-16
    c1 = 1e-4  # Armijo condition
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Backtracking line search
        accepted = False
        trial_alpha = alpha
        while trial_alpha >= min_alpha:
            x_trial = _project_to_sphere(x - trial_alpha * grad_t)
            e_trial, g_trial = _energy_and_grad(x_trial)
            grad_t_trial = _tangent_project(g_trial, x_trial)
            
            # Armijo condition
            if e_trial <= best_e - c1 * trial_alpha * np.sum(grad_t * grad_t):
                # Accept step
                x = x_trial
                best_x = x.copy()
                best_e = e_trial
                grad_t = grad_t_trial
                alpha = min(trial_alpha * 1.5, 0.2)  # Increase step size
                accepted = True
                break
            else:
                trial_alpha *= 0.5
        
        if not accepted:
            break
        
        # Check convergence
        if np.linalg.norm(grad_t) < tol:
            break
    
    return best_x, best_e


def _newton_step(x, max_iter=100, tol=1e-12, deadline=None):
    """Newton's method with BFGS Hessian approximation for faster convergence."""
    x = x.copy()
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e, g = _energy_and_grad(x)
    grad_t = _tangent_project(g, x)
    
    # BFGS Hessian approximation
    H_inv = np.eye(3 * N)  # Inverse Hessian approximation
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Newton direction: -H^{-1} * grad
        direction = -H_inv.dot(grad_t.flatten()).reshape(N, 3)
        
        # Project direction to tangent space
        direction = _tangent_project(direction, x)
        
        # Line search
        alpha = 1.0
        for _ in range(30):
            x_trial = _project_to_sphere(x + alpha * direction)
            e_trial, g_trial = _energy_and_grad(x_trial)
            grad_t_trial = _tangent_project(g_trial, x_trial)
            
            if e_trial < best_e:
                # Update BFGS
                s = alpha * direction.flatten()
                y = (grad_t_trial - grad_t).flatten()
                sy = s.dot(y)
                if sy > 1e-10:
                    H_inv = H_inv - np.outer(s, s) * (H_inv.dot(y).dot(y) / sy) / sy + (s + y).dot(s + y) / (sy * sy)
                
                x = x_trial
                best_x = x.copy()
                best_e = e_trial
                grad_t = grad_t_trial
                break
            else:
                alpha *= 0.5
        
        if np.linalg.norm(grad_t) < tol:
            break
    
    return best_x, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    x_fib = _fibonacci_sphere(N)
    x_fib = _project_to_sphere(x_fib)
    initializations.append(x_fib)
    
    # 2. Random with multiple seeds for diversity
    for seed in [42, 123, 456, 789, 1024]:
        x_rand = _random_uniform_sphere(N, seed=seed)
        initializations.append(x_rand)
    
    best_x = initializations[0].copy()
    best_e, _ = _energy_and_grad(best_x)
    
    # Optimize each initialization with L-BFGS-B
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # L-BFGS-B optimization
        try:
            x_opt, e_opt = _lbfgs_optimize(x_init, maxiter=400, ftol=1e-13, gtol=1e-11, deadline=deadline - 20)
            if e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
    
    # Polish with gradient descent
    if time.time() < deadline - 10:
        try:
            x_pol, e_pol = _gradient_descent_polish(best_x, max_iter=3000, tol=1e-14, deadline=deadline - 5)
            if e_pol < best_e:
                best_x = x_pol
                best_e = e_pol
        except Exception:
            pass
    
    # Final high-precision polish with Newton's method
    if time.time() < deadline - 3:
        try:
            x_final, e_final = _newton_step(best_x, max_iter=50, tol=1e-13, deadline=deadline - 1)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
        except Exception:
            pass
    
    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    min_dist = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                # Nudge point j slightly
                rng = np.random.RandomState(12345 + i * 100 + j)
                best_x[j] += 1e-10 * rng.randn(3)
                best_x = _project_to_sphere(best_x)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
