"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _objective(x):
    """Objective to minimize: negative of the radius."""
    c = x.reshape(N, 3)
    # Check if any point is outside the cube
    if np.any(c < -0.5 + 1e-12) or np.any(c > 0.5 - 1e-12):
        return 1e10
    r = _compute_radius(c)
    return -r

def _initial_configuration():
    """Create a better initial configuration using a structured grid."""
    # Use a 5x5x4 grid arrangement (50 points)
    # This gives a more uniform distribution than a simple 4x4x4 grid
    points = []
    # 5 x 5 x 4 grid
    for k in range(4):
        for j in range(5):
            for i in range(5):
                x = -0.5 + (i + 0.5) / 5.0
                y = -0.5 + (j + 0.5) / 5.0
                z = -0.5 + (k + 0.5) / 4.0
                points.append([x, y, z])
    points = np.array(points[:N])
    return points

def construct():
    """Construct the sphere centers using optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with a structured initial configuration
    c0 = _initial_configuration()
    x0 = c0.flatten()
    
    # Try multiple random restarts
    best_c = c0.copy()
    best_r = _compute_radius(best_c)
    
    rng = np.random.RandomState(42)
    
    # Try several random perturbations and optimize
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Create a perturbed initial configuration
        if restart == 0:
            x_start = x0.copy()
        else:
            # Add random perturbation
            perturbation = rng.randn(N, 3) * 0.05
            x_start = (c0 + perturbation).flatten()
            # Clip to stay within bounds
            x_start = np.clip(x_start, -0.499, 0.499)
        
        # Use Nelder-Mead optimization (robust for non-smooth objectives)
        try:
            result = minimize(
                _objective,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.success or result.fun < -best_r + 1e-12:
                c_new = result.x.reshape(N, 3)
                # Ensure points are within bounds
                if np.all(c_new >= -0.5) and np.all(c_new <= 0.5):
                    r_new = _compute_radius(c_new)
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c_new.copy()
        except Exception:
            pass
        
        # Also try L-BFGS-B with gradient-free approximation
        if time.time() > deadline:
            break
        
        try:
            result2 = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result2.fun < -best_r + 1e-12:
                c_new = result2.x.reshape(N, 3)
                if np.all(c_new >= -0.5) and np.all(c_new <= 0.5):
                    r_new = _compute_radius(c_new)
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c_new.copy()
        except Exception:
            pass
    
    # Final local refinement with many small perturbations
    if time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            # Small random perturbation
            perturbation = rng.randn(N, 3) * 0.01
            c_perturbed = best_c + perturbation
            c_perturbed = np.clip(c_perturbed, -0.499, 0.499)
            
            try:
                result = minimize(
                    _objective,
                    c_perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                if result.fun < -best_r + 1e-12:
                    c_new = result.x.reshape(N, 3)
                    if np.all(c_new >= -0.5) and np.all(c_new <= 0.5):
                        r_new = _compute_radius(c_new)
                        if r_new > best_r:
                            best_r = r_new
                            best_c = c_new.copy()
            except Exception:
                pass
    
    return best_c
# EVOLVE-BLOCK-END
