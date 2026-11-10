"""
Strategy:
1. Start with Fibonacci sphere points (good global distribution).
2. Use L-BFGS-B on the tangent space to minimize logarithmic energy.
3. If L-BFGS-B converges or time is running out, use Simulated Annealing 
   with random perturbations to escape potential local minima.
4. Strict time management to ensure < 120s execution.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _energy_and_grad(points):
    """
    Calculate logarithmic energy and gradient for points on unit sphere.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    """
    # points: (N, 3)
    # Distances: (N, N)
    # diff[i, j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero (should not happen for valid configs, but safe)
    dists[dist == 0] = 1e-10
    # Actually, we need to be careful with the diagonal
    np.fill_diagonal(dists, 1.0) # Diagonal doesn't contribute to sum i<j
    
    # Energy calculation
    # We want sum_{i<j} -log(dists[i,j])
    # Create upper triangular mask
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    log_dists = np.log(dists)
    energy = -np.sum(log_dists[mask])
    
    # Gradient calculation
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2 * 1 ? 
    # Let's derive: E = - sum_{i<j} log(||x_i - x_j||)
    # dE/dx_i = - sum_{j != i} (1/2) * (1/||x_i - x_j||^2) * 2*(x_i - x_j) * (1/2)? No.
    # Let r_ij = ||x_i - x_j||.
    # d/dx_i log(r_ij) = (1/r_ij) * (x_i - x_j)/r_ij = (x_i - x_j)/r_ij^2
    # So dE/dx_i = - sum_{j != i} (x_i - x_j)/r_ij^2
    # Note: for j>i, term is -(x_i-x_j)/r^2. For j<i, term is -(x_i-x_j)/r^2 (since pair is j,i, deriv wrt x_i is same form).
    # Actually, let's just sum over all j != i.
    # grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    # Vectorized gradient
    # inv_dist_sq: (N, N)
    inv_dist_sq = 1.0 / (dists * dists)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = - sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = - sum_j inv_dist_sq[i,j] * x_i + sum_j inv_dist_sq[i,j] * x_j
    # = - x_i * sum_j inv_dist_sq[i,j] + sum_j (inv_dist_sq[i,j] * x_j)
    
    row_sums = np.sum(inv_dist_sq, axis=1)
    # grad = - points * row_sums[:, np.newaxis] + (inv_dist_sq @ points)
    grad = -points * row_sums[:, np.newaxis] + (inv_dist_sq @ points)
    
    return energy, grad

def _project_tangent(v, p):
    """Project vector v onto tangent space of sphere at p."""
    return v - np.dot(v, p) * p

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def construct():
    start_time = time.time()
    
    # Initial guess: Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # 1. L-BFGS-B Optimization on Tangent Space
    # We parameterize the update as a step in the tangent space.
    # However, standard scipy.optimize.minimize works on flat vector spaces.
    # We can use a wrapper that projects the gradient and re-normalizes.
    # A better approach for sphere: use spherical coordinates or just 
    # optimize in R^3 with a penalty or constraint? 
    # L-BFGS-B doesn't handle equality constraints well.
    # Common trick: optimize in R^3, normalize after each step.
    # Gradient in R^3 needs to be projected to tangent space.
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize to ensure we are on sphere
        x = _normalize(x)
        e, g = _energy_and_grad(x)
        # Project gradient to tangent space
        g_proj = np.array([_project_tangent(g[i], x[i]) for i in range(N)])
        return e, g_proj.flatten()
        
    # Run L-BFGS-B
    # We need to be careful with the time limit.
    # L-BFGS-B can be slow if it takes many iterations.
    # Let's run it for a limited number of iterations or until time runs out.
    
    best_points = points
    best_energy = _energy_and_grad(points)[0]
    
    try:
        # Use a callback to check time
        def callback(xk):
            nonlocal best_points, best_energy
            xk = _normalize(xk.reshape(N, 3))
            e, _ = _energy_and_grad(xk)
            if e < best_energy:
                best_energy = e
                best_points = xk
            if time.time() - start_time > TIME_LIMIT * 0.6:
                return True # Stop optimization
            
        res = minimize(
            objective_and_grad,
            points.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'callback': callback}
        )
        if res.success or res.status in [1, 2, 3, 4]: # 1: maxiter, 2: ftol, 3: gtol, 4: rounding
             final_points = _normalize(res.x.reshape(N, 3))
             e, _ = _energy_and_grad(final_points)
             if e < best_energy:
                 best_energy = e
                 best_points = final_points
                 
    except Exception:
        pass

    # 2. Simulated Annealing / Local Refinement
    # If we have time left, try to improve with small random perturbations
    current_points = best_points
    current_energy = best_energy
    T = 0.01  # Initial temperature
    cooling_rate = 0.99
    
    while time.time() - start_time < TIME_LIMIT - 2.0:
        # Generate a random perturbation
        # Move one or a few points randomly
        idx = np.random.randint(N)
        # Random direction on sphere
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        
        # Step size proportional to T
        step_size = 0.05 * T
        new_point = current_points[idx] + step_size * v
        # Normalize
        new_point /= np.linalg.norm(new_point)
        
        # Check if too close to others
        # (Simple check: min distance > 1e-9)
        
        candidate = current_points.copy()
        candidate[idx] = new_point
        candidate = _normalize(candidate)
        
        new_energy, _ = _energy_and_grad(candidate)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_energy - current_energy
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_points = candidate
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points
        
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6 # Floor temperature to allow fine tuning
            
    return best_points

# Ensure the function is accessible
if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
