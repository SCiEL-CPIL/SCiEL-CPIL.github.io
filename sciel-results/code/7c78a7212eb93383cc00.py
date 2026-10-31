import numpy as np
from scipy.optimize import differential_evolution
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective(x):
    points = x.reshape(-1, D)
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals == 0):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def construct():
    # Approach: Polynomial with unit coefficients
    # P(z) = sum_{k=0}^{15} exp(i*theta_k) * z^k
    # Optimize theta_k to minimize distance ratio of roots
    
    def poly_objective(thetas):
        coeffs = np.exp(1j * thetas)
        try:
            roots = np.roots(coeffs)
        except Exception:
            return 1e10
        
        points = np.column_stack([roots.real, roots.imag])
        
        d_sq = pairwise_distances_sq(points)
        mask = ~np.eye(N, dtype=bool)
        d_sq_vals = d_sq[mask]
        
        if np.any(d_sq_vals < 1e-12):
            return 1e10
        
        d_min_sq = np.min(d_sq_vals)
        d_max_sq = np.max(d_sq_vals)
        
        return d_max_sq / d_min_sq
    
    bounds = [(0, 2 * np.pi)] * N
    
    try:
        result = differential_evolution(
            poly_objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-5,
            popsize=12,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True
        )
        best_thetas = result.x
    except Exception:
        best_thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    coeffs = np.exp(1j * best_thetas)
    roots = np.roots(coeffs)
    points = np.column_stack([roots.real, roots.imag])
    points -= np.mean(points, axis=0)
    
    final_score = objective(points.flatten())
    
    if final_score > 1e6:
        grid = list(itertools.product(range(4), repeat=2))[:N]
        points = np.array(grid, dtype=float)
        points -= np.mean(points, axis=0)
    
    return points
