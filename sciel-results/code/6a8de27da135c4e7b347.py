"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = points.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return total

def _log_energy_fast(points):
    """Faster vectorized energy computation."""
    n = points.shape[0]
    # Compute all pairwise distances
    # Use broadcasting but be careful with memory for N=150 (150x150 is fine)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))

def _gradient(points):
    """Compute gradient of logarithmic energy.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = points.shape[0]
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            grad[i] += d / (dist * dist)
    return grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _line_search(points, direction, energy):
    """Backtracking line search with projection onto sphere."""
    n = points.shape[0]
    # Compute gradient for Wolfe conditions
    grad = _gradient(points)
    # Project gradient to tangent space
    grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    # Armijo condition
    alpha = 1.0
    c1 = 1e-4
    for _ in range(50):
        new_points = _project_to_sphere(points + alpha * direction)
        new_energy = _log_energy_fast(new_points)
        if new_energy < energy + c1 * alpha * np.sum(grad_tangent * direction):
            return new_points, new_energy, alpha
        alpha *= 0.5
    return _project_to_sphere(points + alpha * direction), _log_energy_fast(_project_to_sphere(points + alpha * direction)), alpha

def _lbfgs_step(points, history_p, history_g, m=10):
    """L-BFGS two-loop recursion to compute search direction."""
    n = len(history_p)
    if n == 0:
        return -history_g[-1] if history_g else None
    
    # Use stored history
    s = history_p
    y = history_g
    
    rho = [1.0 / np.dot(s_i, y_i) for s_i, y_i in zip(s, y) if np.dot(s_i, y_i) > 1e-12]
    if len(rho) == 0:
        return -history_g[-1]
    
    # Two-loop recursion
    q = history_g[-1].copy()
    r = [0.0] * len(s)
    for i in range(len(s) - 1, -1, -1):
        if np.dot(s[i], y[i]) > 1e-12:
            r[i] = np.dot(s[i], q) / np.dot(s[i], y[i])
            q = q - r[i] * y[i]
    
    # Initial Hessian approximation (identity)
    gamma = 1.0
    if len(s) > 0 and len(y) > 0:
        y_last = y[-1]
        s_last = s[-1]
        if np.dot(s_last, y_last) > 1e-12:
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
    
    r_vec = gamma * q
    
    for i in range(len(s)):
        if np.dot(s[i], y[i]) > 1e-12:
            alpha_i = np.dot(s[i], r_vec) / np.dot(s[i], y[i])
            r_vec = r_vec + (alpha_i - r[i]) * y[i]
    
    return -r_vec

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_energy = _log_energy_fast(points)
    
    # L-BFGS history
    max_history = 10
    history_p = []
    history_g = []
    
    # Gradient descent with L-BFGS
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    iteration = 0
    no_improve_count = 0
    patience = 200
    
    while time.time() < deadline:
        iteration += 1
        
        # Compute gradient
        grad = _gradient(points)
        
        # Project to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # L-BFGS direction
        if len(history_g) > 0:
            direction = _lbfgs_step(points, history_p[-max_history:], history_g[-max_history:], max_history)
        else:
            direction = -grad_tangent
        
        # Normalize direction to unit norm for consistent step size
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-12:
            direction = direction / dir_norm
        else:
            direction = -grad_tangent
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-12:
                direction = direction / dir_norm
        
        # Adaptive learning rate
        if no_improve_count > 50:
            lr *= 0.9
        elif no_improve_count > 20:
            lr *= 0.95
        
        # Try multiple step sizes
        best_trial_points = None
        best_trial_energy = best_energy
        for trial_lr in [lr, lr * 0.5, lr * 2.0, lr * 0.1]:
            if time.time() >= deadline:
                break
            trial_points = _project_to_sphere(points + trial_lr * direction)
            trial_energy = _log_energy_fast(trial_points)
            if trial_energy < best_trial_energy:
                best_trial_energy = trial_energy
                best_trial_points = trial_points
        
        if best_trial_points is not None and best_trial_energy < best_energy:
            # Update history for L-BFGS
            step = best_trial_points - points
            history_p.append(step)
            history_g.append(grad)
            if len(history_p) > max_history:
                history_p.pop(0)
                history_g.pop(0)
            
            points = best_trial_points
            best_energy = best_trial_energy
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                # Restart with small perturbation
                if time.time() >= deadline:
                    break
                perturb = np.random.randn(*points.shape) * 0.01
                points = _project_to_sphere(points + perturb)
                current_energy = _log_energy_fast(points)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = points.copy()
                no_improve_count = 0
                history_p.clear()
                history_g.clear()
                lr = 0.005
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
    
    # Final verification
    final_points = _project_to_sphere(best_points)
    final_energy = _log_energy_fast(final_points)
    
    # Ensure points are valid
    norms = np.linalg.norm(final_points, axis=1)
    if np.any(norms < 1e-15) or not np.all(np.isfinite(final_points)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        final_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return final_points
# EVOLVE-BLOCK-END
