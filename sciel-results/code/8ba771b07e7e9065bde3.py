"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/d_ij) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10  # invalid
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _normalize(points):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _energy_from_spherical(theta, phi):
    """Convert spherical coordinates to cartesian and compute energy."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    return _log_energy(points)

def construct():
    # Start with a good initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates for optimization
    # theta = arccos(z), phi = arctan2(y, x)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi_sph = np.arctan2(points[:, 1], points[:, 0])
    
    # Use L-BFGS-B for local optimization in spherical coordinates
    # This is more efficient than Cartesian for constrained optimization
    
    def objective(params):
        t = params[:N]
        p = params[N:]
        # Wrap phi to [0, 2*pi)
        p = np.mod(p, 2 * np.pi)
        x = np.sin(t) * np.cos(p)
        y = np.sin(t) * np.sin(p)
        z = np.cos(t)
        pts = np.column_stack([x, y, z])
        return _log_energy(pts)
    
    initial_params = np.concatenate([theta, phi_sph])
    
    # Try L-BFGS-B
    result = minimize(objective, initial_params, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if result.success or result.fun < _log_energy(points):
        t = result.x[:N]
        p = result.x[N:]
        p = np.mod(p, 2 * np.pi)
        x = np.sin(t) * np.cos(p)
        y = np.sin(t) * np.sin(p)
        z = np.cos(t)
        points = np.column_stack([x, y, z])
    
    # Refine with a second pass using different initial perturbation
    np.random.seed(42)
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    for trial in range(3):
        # Perturb the current best
        perturbation = np.random.randn(N, 3) * 0.1
        perturbed = best_points + perturbation
        perturbed = _normalize(perturbed)
        
        # Convert to spherical
        t = np.arccos(np.clip(perturbed[:, 2], -1.0, 1.0))
        p = np.arctan2(perturbed[:, 1], perturbed[:, 0])
        initial_params = np.concatenate([t, p])
        
        result = minimize(objective, initial_params, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.success:
            t = result.x[:N]
            p = result.x[N:]
            p = np.mod(p, 2 * np.pi)
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            pts = np.column_stack([x, y, z])
            e = _log_energy(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts
    
    return best_points
# EVOLVE-BLOCK-END
