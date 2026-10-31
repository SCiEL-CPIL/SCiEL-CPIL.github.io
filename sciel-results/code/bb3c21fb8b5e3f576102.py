import numpy as np
from scipy.optimize import differential_evolution
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective(theta):
    # theta: array of 17 angles (for degree 16 polynomial)
    # P(z) = sum_{k=0}^{16} exp(i * theta_k) * z^k
    # Coefficients for np.roots must be in descending order of power: [c_16, ..., c_0]
    coeffs = np.exp(1j * theta)
    # coeffs[0] is c_0 (power 0), coeffs[16] is c_16 (power 16)
    # np.roots expects [c_16, c_15, ..., c_0]
    poly_coeffs = coeffs[::-1]
    
    try:
        roots = np.roots(poly_coeffs)
    except Exception:
        return 1e10
        
    # Check for NaNs or Infs
    if not np.all(np.isfinite(roots)):
        return 1e10

    points = np.column_stack((roots.real, roots.imag))
    
    # Check for distinct points
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals == 0) or np.any(d_sq_vals < 1e-12):
        return 1e10
        
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    if d_min_sq == 0:
        return 1e10
        
    return d_max_sq / d_min_sq

def construct():
    # We use a degree 16 polynomial to get 16 roots
    # Variables: 17 angles
    bounds = [(0, 2 * np.pi) for _ in range(17)]
    
    # Differential Evolution
    # Seed for reproducibility
    np.random.seed(42)
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            atol=1e-4,
            polish=True
        )
        
        if res.fun < 1e9:
            best_theta = res.x
            # Reconstruct points
            coeffs = np.exp(1j * best_theta)
            poly_coeffs = coeffs[::-1]
            roots = np.roots(poly_coeffs)
            points = np.column_stack((roots.real, roots.imag))
            
            # Verify validity
            d_sq = pairwise_distances_sq(points)
            mask = ~np.eye(N, dtype=bool)
            d_sq_vals = d_sq[mask]
            if np.any(d_sq_vals < 1e-12):
                raise ValueError("Invalid points")
            
            return points
    except Exception:
        pass

    # Fallback: Return a known good configuration if DE fails
    # 4x4 grid
    grid = list(itertools.product(range(4), range(4)))
    points = np.array(grid, dtype=float)
    points -= np.mean(points, axis=0)
    return points
