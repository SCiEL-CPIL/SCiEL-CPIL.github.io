import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _energy(points):
    """Compute logarithmic energy E = sum_{i<j} log(1/||x_i - x_j||)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    E = sum_{i<j} log(1/||x_i - x_j||)
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    return energy, grad

def _objective_and_grad(x):
    """
    Objective function and gradient for scipy.minimize.
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points_sphere = points / norms
    
    e, g = _energy_and_grad(points_sphere)
    
    lambda_pen = 100.0
    diff_pen = points - points_sphere
    penalty = 0.5 * lambda_pen * np.sum(diff_pen**2)
    grad_pen = lambda_pen * diff_pen
    
    total_grad = (g + grad_pen).flatten()
    return e + penalty, total_grad

def _fibonacci_spiral(seed=0):
    rng = np.random.RandomState(seed)
    # Slightly perturb the spiral for diversity
    i = np.arange(N) + 0.5 + rng.uniform(-0.01, 0.01, N)
    i = np.clip(i, 0.5, N - 0.5)
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi, N)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_initial(seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _gradient_descent_tangent(points, lr=0.1, max_iter=500, tol=1e-10, time_budget=30.0):
    """
    Gradient descent on the sphere using tangent space projection.
    """
    start = time.time()
    p = points.copy()
    E, g = _energy_and_grad(p)
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Project gradient to tangent space
        # For each point i, tangent space is {v : v . p_i = 0}
        # g_tangent_i = g_i - (g_i . p_i) * p_i
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        
        # Check convergence
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
        
        # Adaptive learning rate
        step = lr * g_tan
        p_new = p + step
        p_new = _project_to_sphere(p_new)
        
        E_new, g_new = _energy_and_grad(p_new)
        
        if E_new < E:
            p = p_new
            E = E_new
            g = g_new
        else:
            # Halve learning rate
            lr *= 0.5
            if lr < 1e-15:
                break
    
    return p, E

def _sa_step(points, T, rng):
    """
    Perform one Simulated Annealing step.
    Moves a batch of points by a small random rotation.
    """
    batch_size = rng.randint(1, 4)
    indices = rng.choice(N, size=batch_size, replace=False)
    
    new_points = points.copy()
    
    angle = 0.1 * np.sqrt(T)
    if angle > 0.5:
        angle = 0.5
        
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    
    u = axis
    u_col = u[:, np.newaxis]
    K = np.array([[0, -u[2], u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]])
    c = np.cos(angle)
    s = np.sin(angle)
    R = c * np.eye(3) + (1 - c) * (u_col @ u_col.T) + s * K
    
    new_points[indices] = new_points[indices] @ R.T
    new_points[indices] = _project_to_sphere(new_points[indices])
    
    E_old = _energy(points)
    E_new = _energy(new_points)
    delta_E = E_new - E_old
    
    return new_points, delta_E, E_new

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_points = None
    best_energy = float('inf')
    
    def try_improve(points, tag=""):
        nonlocal best_points, best_energy
        e = _energy(points)
        if e < best_energy:
            best_energy = e
            best_points = points.copy()
    
    # Try multiple initializations
    num_starts = 5
    for s in range(num_starts):
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
        
        if s == 0:
            points = _fibonacci_spiral(0)
        elif s == 1:
            points = _fibonacci_spiral(123)
        else:
            points = _random_initial(1000 + s)
        
        # Gradient descent polish
        p_gd, e_gd = _gradient_descent_tangent(points, lr=0.1, max_iter=300, time_budget=20.0)
        try_improve(p_gd, f"gd_{s}")
        
        # L-BFGS polish
        try:
            x0 = p_gd.flatten()
            res = minimize(_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20})
            if hasattr(res, 'x'):
                new_points = _project_to_sphere(res.x.reshape(N, 3))
                e_new = _energy(new_points)
                try_improve(new_points, f"lbfgs_{s}")
        except Exception:
            pass
        
        # SA from the best so far
        if s >= 1:
            if time.time() - start_time > TIME_LIMIT * 0.6:
                break
            
            current_points = best_points.copy()
            current_energy = best_energy
            
            T = 2.0
            cooling_rate = 0.998
            sa_time_limit = min(30.0, max(0, TIME_LIMIT * 0.8 - (time.time() - start_time)))
            
            while time.time() - start_time < TIME_LIMIT * 0.8 and T > 0.01:
                new_points, delta_E, E_new = _sa_step(current_points, T, rng)
                
                if delta_E < 0 or (T > 1e-6 and rng.rand() < np.exp(-delta_E / T)):
                    current_points = new_points
                    current_energy = E_new
                    try_improve(current_points, f"sa_{s}")
                
                T *= cooling_rate
            
            # Final polish after SA
            if time.time() - start_time < TIME_LIMIT * 0.8:
                p_gd2, e_gd2 = _gradient_descent_tangent(best_points, lr=0.01, max_iter=500, time_budget=15.0)
                try_improve(p_gd2, f"gd_final_{s}")
                
                try:
                    x0 = p_gd2.flatten()
                    res = minimize(_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                                   options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                    if hasattr(res, 'x'):
                        new_points = _project_to_sphere(res.x.reshape(N, 3))
                        e_new = _energy(new_points)
                        try_improve(new_points, f"lbfgs_final_{s}")
                except Exception:
                    pass
    
    # Final validation and collision check
    if best_points is None:
        best_points = _fibonacci_spiral(0)
    
    best_points = _project_to_sphere(best_points)
    
    # Collision check
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng_coll = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng_coll.randn(3) * 1e-6
                best_points[i] = _project_to_sphere(best_points[i].reshape(1, 3))[0]
    
    return best_points
