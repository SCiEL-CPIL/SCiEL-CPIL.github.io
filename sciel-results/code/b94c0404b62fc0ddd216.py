"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares to minimize the sum of squared inverse distances,
    which effectively maximizes the minimum distance between circles and from the boundary.
    """
    n = N
    
    # Initial guess: Hexagonal lattice
    # We want to fit ~300 points in radius 1.
    # Area per point ~ pi / 300 ~ 0.0104.
    # Side length of hex cell s: area = s^2 * sqrt(3)/2 ~ 0.0104 => s ~ 0.12.
    # Let's generate a hex grid and select points inside radius 1 - margin.
    
    def get_hex_grid(radius_limit):
        # Spacing
        # Approximate number of points in hex grid of radius R with spacing s:
        # N ~ pi * R^2 / (s^2 * sqrt(3)/2)
        # s ~ sqrt(2 * pi * R^2 / (N * sqrt(3)))
        s = np.sqrt(2.0 * np.pi * radius_limit**2 / (n * np.sqrt(3.0)))
        
        # Generate points
        # k is the number of "rings"
        k = int(radius_limit / (s * np.sqrt(3.0) / 2.0)) + 2
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * s + 0.5 * s * (j % 2)
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= radius_limit:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n:
            # If we don't have enough, decrease spacing
            return None
        # Select n points closest to origin to keep them centered? 
        # Or just take the first n. Taking closest to origin might bias.
        # Better: take all, and if more than n, we need to remove some.
        # For initial guess, just take the first n is fine, but randomizing might help.
        # Let's take the n points closest to the center to ensure they are well inside.
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        return points[indices[:n]]

    x0 = get_hex_grid(0.95)
    if x0 is None:
        # Fallback to simple grid
        xs = np.linspace(-0.9, 0.9, 20)
        ys = np.linspace(-0.9, 0.9, 20)
        grid = np.array([[x, y] for x in xs for y in ys])
        x0 = grid[:n]

    def residuals(x):
        centres = x.reshape(n, 2)
        
        # Pairwise distances
        # We need to avoid division by zero.
        # pdist returns condensed distance matrix.
        dists = pdist(centres)
        
        # Clip small distances to avoid inf
        dists = np.maximum(dists, 1e-6)
        
        # Boundary distances
        norms = np.linalg.norm(centres, axis=1)
        # Clip to avoid division by zero if point is at center? No, 1 - 0 = 1.
        # Clip to avoid division by zero if point is on boundary.
        boundary_dists = 1.0 - norms
        boundary_dists = np.maximum(boundary_dists, 1e-6)
        
        # Residuals: inverse distances
        # We want to minimize sum of (1/d)^2. 
        # To balance pair and boundary terms, we might scale them.
        # Typically, there are ~45000 pairs and 300 boundary terms.
        # The pair terms will dominate.
        # Let's weight the boundary terms to be comparable.
        # Or just return them and let least_squares handle it.
        # Since we minimize sum of squares, the magnitude matters.
        # 1/d for d~0.1 is 10. 10^2 = 100.
        # 1/b for b~0.1 is 10. 10^2 = 100.
        # There are 45k pairs. Sum ~ 4.5M.
        # There are 300 boundary. Sum ~ 30k.
        # The pair terms dominate. This is good, we want to spread them out.
        # But we also need to keep them in.
        
        r_pairs = 1.0 / dists
        r_boundary = 1.0 / boundary_dists
        
        # Scale boundary to be more influential?
        # Let's multiply boundary by sqrt(150) to balance counts roughly?
        # sqrt(45000/300) = sqrt(150) ~ 12.
        r_boundary *= 12.0
        
        return np.concatenate([r_pairs, r_boundary])

    # Run optimization
    # Use 'trf' (Trust Region Reflective) which is robust for non-smooth or large problems.
    # max_nfev should be high enough to converge.
    result = least_squares(residuals, x0.flatten(), method='trf', max_nfev=20000, 
                           xtol=1e-8, ftol=1e-8, gtol=1e-8)
    
    centres = result.x.reshape(n, 2)
    
    # Calculate score
    dists = pdist(centres)
    min_pair_dist = np.min(dists)
    norms = np.linalg.norm(centres, axis=1)
    min_boundary_dist = 1.0 - np.max(norms)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # If the result is invalid or very bad, try a restart with a different initial guess
    if r < 0.04 or not np.isfinite(r):
        # Try again with a slightly perturbed initial guess
        x0_perturbed = x0 + np.random.normal(0, 0.01, x0.shape)
        # Ensure perturbed points are still inside
        norms_p = np.linalg.norm(x0_perturbed, axis=1)
        mask = norms_p < 0.98
        if np.sum(mask) < n:
             # Re-normalize or clip
             x0_perturbed = x0_perturbed * (0.98 / np.max(norms_p))
        
        result2 = least_squares(residuals, x0_perturbed.flatten(), method='trf', max_nfev=20000, 
                               xtol=1e-8, ftol=1e-8, gtol=1e-8)
        centres2 = result2.x.reshape(n, 2)
        
        dists2 = pdist(centres2)
        min_pair_dist2 = np.min(dists2)
        norms2 = np.linalg.norm(centres2, axis=1)
        min_boundary_dist2 = 1.0 - np.max(norms2)
        r2 = min(min_pair_dist2 / 2.0, min_boundary_dist2)
        
        if r2 > r:
            centres = centres2
            r = r2

    return centres
# EVOLVE-BLOCK-END
