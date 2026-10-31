"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def construct():
    """
    Place 16 distinct points in 2D Euclidean space to minimize the squared ratio
    of maximum to minimum pairwise distance.
    
    Approach: Represent 16 points as vertices of a convex polygon defined by a radial
    function r(theta) parameterized by a low-order Fourier series with only even harmonics
    to enforce central symmetry. This restricts the search to a low-dimensional manifold
    of star-shaped convex bodies.
    
    We use 16 points evenly spaced in angle, with radii given by:
    r(theta) = 1 + sum_{k=1}^{K} a_k * cos(2k*theta)
    
    The parameters (a_1, ..., a_K) are optimized to minimize (max_dist / min_dist)^2.
    """
    
    K = 7  # Number of even harmonics: cos(2θ), cos(4θ), ..., cos(14θ)
    
    def points_from_params(params):
        """Convert Fourier parameters to 16 points in 2D."""
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for k in range(K):
            radii += params[k] * np.cos(2 * (k + 1) * thetas)
        
        # Ensure radii are positive (star-shaped)
        radii = np.maximum(radii, 0.1)
        
        x = radii * np.cos(thetas)
        y = radii * np.sin(thetas)
        return np.column_stack([x, y])
    
    def objective(params):
        """Compute the squared ratio of max to min pairwise distance."""
        pts = points_from_params(params)
        
        # Compute all pairwise distances
        n = len(pts)
        dx = pts[:, np.newaxis, 0] - pts[np.newaxis, :, 0]
        dy = pts[:, np.newaxis, 1] - pts[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Get upper triangle (excluding diagonal)
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        min_d = np.min(pair_dists)
        max_d = np.max(pair_dists)
        
        if min_d < 1e-10:
            return 1e6  # Penalty for degenerate configurations
        
        ratio = max_d / min_d
        return ratio ** 2
    
    # Initial guess: circle (all params = 0)
    x0 = np.zeros(K)
    
    # Try multiple starting points to avoid local minima
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Random perturbations with increasing scale
    np.random.seed(42)
    for trial in range(50):
        if trial == 0:
            x_try = x0.copy()
        else:
            # Use a mix of small and large perturbations
            if trial < 30:
                scale = 0.3
            else:
                scale = 0.15
            x_try = np.random.uniform(-scale, scale, size=K)
        
        res = minimize(objective, x_try, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    
    # Final refinement with tighter tolerances
    res = minimize(objective, best_x, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    if res.fun < best_val:
        best_val = res.fun
        best_x = res.x
    
    pts = points_from_params(best_x)
    
    # Verify points are distinct
    # If any points are too close, add small perturbation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-8:
                pts[j] += np.array([1e-6, 1e-6])
    
    return pts

# EVOLVE-BLOCK-END
