import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: random points in a slightly smaller box to avoid immediate boundary issues
    # Box [-0.45, 0.45]
    x0 = np.random.uniform(-0.45, 0.45, (N, 3))
    
    # Current radius to try to achieve
    r_current = 0.08
    r_best = 0.0
    x_best = x0.copy()
    
    # Step size for r
    dr = 0.002
    
    # Bounds for the optimizer
    bounds = [(-0.5, 0.5)] * (N * 3)

    # Helper to compute energy and gradient for a given r
    def energy_and_grad(x_flat, r_val):
        x = x_flat.reshape(N, 3)
        
        # Calculate distances between all pairs
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask out self-pairs
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise energy: sum(max(0, 2r - d)^2)
        overlap = 2.0 * r_val - dists
        overlap[overlap < 0] = 0.0
        energy_pairs = np.sum(overlap ** 2)
        
        # Boundary energy: sum(max(0, r - b)^2)
        abs_x = np.abs(x)
        b = 0.5 - abs_x
        boundary_violation = r_val - b
        boundary_violation[boundary_violation < 0] = 0.0
        energy_boundaries = np.sum(boundary_violation ** 2)
        
        total_energy = energy_pairs + energy_boundaries
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        safe_dists = np.where(dists == 0, 1.0, dists)
        factor = 2.0 * overlap / safe_dists
        np.fill_diagonal(factor, 0.0)
        
        grad_pairs = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        grad += grad_pairs
        
        # Boundary gradient
        sign_x = np.sign(x)
        grad_boundaries = 2.0 * boundary_violation * sign_x
        grad += grad_boundaries
        
        return total_energy, grad.flatten()

    # Optimization loop
    stall_count = 0
    
    while time.time() < deadline:
        # Try to minimize energy for current r_current
        res = minimize(
            energy_and_grad, 
            x_best.flatten(), 
            args=(r_current,), 
            method='L-BFGS-B', 
            jac=True,
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success or res.fun < 1e-6:
            x_trial = res.x.reshape(N, 3)
            
            # Check actual feasibility
            diff = x_trial[:, np.newaxis, :] - x_trial[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            d_min = np.min(dists)
            
            abs_x = np.abs(x_trial)
            b = 0.5 - abs_x
            b_min = np.min(b)
            
            r_achieved = min(d_min / 2.0, b_min)
            
            if r_achieved > r_best:
                r_best = r_achieved
                x_best = x_trial.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # If we achieved the target radius, try to increase it
            if r_achieved >= r_current - 1e-4:
                r_current += dr
                # If we are getting close to the limit, reduce step
                if r_current > 0.11:
                    dr = 0.001
            else:
                # Failed to reach r_current
                if stall_count > 5:
                    # Restart
                    x_best = np.random.uniform(-0.45, 0.45, (N, 3))
                    r_current = max(0.08, r_best * 0.95)
                    stall_count = 0
                else:
                    r_current -= dr * 0.5
                    if r_current < 0.05:
                        break
        else:
            # Optimization failed
            stall_count += 1
            if stall_count > 5:
                x_best = np.random.uniform(-0.45, 0.45, (N, 3))
                r_current = max(0.08, r_best * 0.95)
                stall_count = 0
            else:
                r_current -= dr * 0.5
                if r_current < 0.05:
                    break

    return x_best
