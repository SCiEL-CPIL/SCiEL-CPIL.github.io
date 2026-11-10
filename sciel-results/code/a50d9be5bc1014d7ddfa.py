"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, LinearConstraint

DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s kill

def _compute_radius(centres):
    """Compute the valid common radius for a set of centres in [-0.5, 0.5]^3."""
    # Check validity
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(np.abs(centres) > 0.5 + 1e-12):
        return 0.0
    
    # Compute min distance to boundary
    min_boundary = np.min(0.5 - np.abs(centres))
    
    # Compute min pairwise distance / 2
    # For efficiency, compute all pairwise distances
    # Use broadcasting
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_boundary, min_pair)

def _initial_grid():
    """Create an initial grid of N points in the cube."""
    # Use a 5x5x4 grid = 100 points
    # Side length 1, so spacing is 1/5 = 0.2 in x,y and 1/4 = 0.25 in z
    # Centers at (i+0.5)/m - 0.5
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _objective_and_grad(params, n):
    """
    Objective: minimize -r where r is the radius.
    params: flat array of shape (3*n,)
    Returns (value, gradient)
    """
    centres = params.reshape(n, 3)
    
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Distance to boundaries
    bound_dist = 0.5 - np.abs(centres)
    min_bound = np.min(bound_dist)
    
    # Min pairwise distance
    min_pair = np.min(dist) / 2.0
    
    r = min(min_bound, min_pair)
    
    # Gradient: we want to maximize r, so minimize -r
    # The gradient of -r w.r.t. centres depends on which constraint is active
    grad = np.zeros_like(centres)
    
    # Find active constraints
    # For boundary: gradient of (0.5 - |c_ik|) w.r.t. c_i is -sign(c_ik)
    # For pairwise: gradient of |c_i - c_j|/2 w.r.t. c_i is (c_i - c_j) / (2 * |c_i - c_j|)
    
    if min_bound <= min_pair:
        # Boundary constraint is active
        # Find all points at minimum boundary distance
        for i in range(n):
            for k in range(3):
                if abs(bound_dist[i, k] - min_bound) < 1e-12:
                    grad[i, k] -= np.sign(centres[i, k]) if centres[i, k] != 0 else 1.0
    else:
        # Pairwise constraint is active
        for i in range(n):
            for j in range(i+1, n):
                d = dist[i, j]
                if abs(d - 2*min_pair) < 1e-12:
                    diff_ij = centres[i] - centres[j]
                    grad[i] += diff_ij / (2 * d)
                    grad[j] -= diff_ij / (2 * d)
    
    return -r, grad.flatten()

def _optimize_from_initial(centres, time_limit):
    """Run local optimization from given initial centres."""
    n = centres.shape[0]
    params = centres.flatten()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (3 * n)
    
    best_params = params.copy()
    best_r = _compute_radius(centres)
    
    start_time = time.time()
    
    # Try multiple optimization strategies
    try:
        # Strategy 1: L-BFGS-B
        res = minimize(
            _objective_and_grad,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        
        if time.time() - start_time > time_limit:
            return best_params.reshape(n, 3), best_r
        
        r_new = _compute_radius(res.x.reshape(n, 3))
        if r_new > best_r:
            best_params = res.x
            best_r = r_new
        
        # Strategy 2: Try from the result with different initial perturbation
        if time.time() - start_time < time_limit * 0.5:
            # Add small random perturbation
            perturbed = best_params.reshape(n, 3) + np.random.randn(n, 3) * 0.01
            # Clip to bounds
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            res2 = minimize(
                _objective_and_grad,
                perturbed.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            
            if time.time() - start_time > time_limit:
                return best_params.reshape(n, 3), best_r
            
            r_new2 = _compute_radius(res2.x.reshape(n, 3))
            if r_new2 > best_r:
                best_params = res2.x
                best_r = r_new2
        
        # Strategy 3: Nelder-Mead as a final polish
        if time.time() - start_time < time_limit * 0.7:
            res3 = minimize(
                lambda p: -_compute_radius(p.reshape(n, 3)),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if time.time() - start_time > time_limit:
                return best_params.reshape(n, 3), best_r
            
            r_new3 = _compute_radius(res3.x.reshape(n, 3))
            if r_new3 > best_r:
                best_params = res3.x
                best_r = r_new3
        
    except Exception:
        pass
    
    return best_params.reshape(n, 3), best_r

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial grid
    centres = _initial_grid()
    best_centres = centres.copy()
    best_r = _compute_radius(centres)
    
    # Try multiple random initializations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create a random initial configuration
        if restart == 0:
            init = _initial_grid()
        else:
            # Random points with some structure
            init = np.random.uniform(-0.45, 0.45, (N, 3))
        
        # Optimize
        opt_centres, opt_r = _optimize_from_initial(init, 10.0)
        
        if opt_r > best_r:
            best_centres = opt_centres
            best_r = opt_r
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 5.0:
        final_centres, final_r = _optimize_from_initial(best_centres, 20.0)
        if final_r > best_r:
            best_centres = final_centres
            best_r = final_r
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
