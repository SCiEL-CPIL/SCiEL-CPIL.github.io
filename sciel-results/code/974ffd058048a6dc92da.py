"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _min_pairwise_cos(points):
    """Compute the maximum dot product over all pairs (excluding self)."""
    dots = points @ points.T
    np.fill_diagonal(dots, -2.0)
    return np.max(dots)


def construct():
    # Objective: minimize the maximum pairwise dot product
    # (equivalent to maximizing the minimum pairwise angular distance)
    # This is a different approach from Coulomb energy minimization.
    
    def objective(params):
        theta = params[:N]
        phi = params[N:]
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.column_stack([x, y, z])
        
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        
        max_dot = _min_pairwise_cos(points)
        # We want to minimize max_dot (which means maximizing min angle)
        return max_dot
    
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0.0, np.pi)] * N + [(0.0, 2.0 * np.pi)] * N
    
    # Use differential evolution - a population-based stochastic optimizer
    # This is fundamentally different from gradient-based L-BFGS-B
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,
        popsize=10,
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=True,
        init='latinhypercube'
    )
    
    theta_opt = result.x[:N]
    phi_opt = result.x[N:]
    
    x = np.sin(theta_opt) * np.cos(phi_opt)
    y = np.sin(theta_opt) * np.sin(phi_opt)
    z = np.cos(theta_opt)
    p_opt = np.column_stack([x, y, z])
    
    norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    p_opt = p_opt / norms
    
    return p_opt
# EVOLVE-BLOCK-END
