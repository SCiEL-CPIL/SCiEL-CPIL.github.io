"""Optimized program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute (max_dist / min_dist)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return float('inf')
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Exclude diagonal (zero distances)
    mask = ~np.eye(n, dtype=bool)
    pairwise_dists = dists[mask]
    if len(pairwise_dists) == 0:
        return float('inf')
    d_min = np.min(pairwise_dists)
    d_max = np.max(pairwise_dists)
    if d_min < 1e-12:
        return float('inf')
    return (d_max / d_min) ** 2

def _construct_from_fourier(coeffs, n_harmonics, n_points=N):
    """
    Construct n_points from Fourier series coefficients.
    coeffs: array of length 2*n_harmonics+1 (a0, a1, b1, a2, b2, ...)
    Uses even harmonics only for central symmetry: r(theta) = a0 + sum_{k=1}^{n_harmonics} a_k * cos(2*k*theta)
    """
    a0 = coeffs[0]
    a_coeffs = coeffs[1:]  # a1, a2, ..., a_{n_harmonics}
    
    # Generate angles for n_points equally spaced
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Compute r(theta) = a0 + sum_{k=1}^{n_harmonics} a_k * cos(2*k*theta)
    r = np.full(n_points, a0, dtype=float)
    for k, ak in enumerate(a_coeffs, start=1):
        r += ak * np.cos(2*k*theta)
    
    # Ensure r > 0 for all points
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack([x, y])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    best_score = float('inf')
    best_points = None
    
    # Start with a regular 16-gon as initial guess
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    r0 = np.ones(N)
    init_points = np.column_stack([r0*np.cos(theta), r0*np.sin(theta)])
    
    # Fourier series with up to 8 harmonics
    n_harmonics = 8
    n_coeffs = 1 + n_harmonics  # a0 + a1..a_n
    
    # Initial coefficients: a0=1, others=0
    init_coeffs = np.zeros(n_coeffs)
    init_coeffs[0] = 1.0
    
    # Try multiple random restarts
    np.random.seed(42)
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = init_coeffs.copy()
        else:
            # Random perturbation
            x0 = init_coeffs.copy()
            for i in range(1, n_coeffs):
                x0[i] = np.random.normal(0, 0.1)
        
        def objective(coeffs):
            points = _construct_from_fourier(coeffs, n_harmonics)
            return _score(points)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success or result.fun < best_score:
                best_coeffs = result.x
                best_points = _construct_from_fourier(best_coeffs, n_harmonics)
                best_score = result.fun
        except Exception:
            pass
    
    # If no valid result, fall back to regular polygon
    if best_points is None:
        best_points = init_points
    
    return best_points
# EVOLVE-BLOCK-END
