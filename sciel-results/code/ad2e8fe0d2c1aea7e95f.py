"""Rewritten program to find LJ38 global minimum using FCC structure."""
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy(positions):
    """Calculate the total Lennard-Jones energy."""
    # positions: (N, 3)
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    # Mask diagonal (self-interaction)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Avoid division by zero if any two atoms overlap (should not happen with LJ repulsion)
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    r6 = dist_sq ** -3
    r12 = r6 ** 2
    
    # Sum over all pairs. Since the matrix is symmetric, we sum upper triangle and multiply by 2?
    # Or just sum all and divide by 2?
    # The formula is 4 * sum(r^-12 - r^-6) over all pairs i<j.
    # Summing the whole matrix sums each pair twice.
    # So, 4 * 0.5 * sum_matrix = 2 * sum_matrix.
    energy = 2.0 * np.sum(r12 - r6)
    return energy

def _fcc_points(n_points, cutoff=4.0):
    """Generate the first n_points of an FCC lattice centered at the origin."""
    points = []
    # Generate candidate points in a box
    # FCC condition: x+y+z is integer, coordinates are multiples of 0.5
    # Range needs to be large enough to get n_points
    coords = np.arange(-cutoff, cutoff + 0.5, 0.5)
    for x in coords:
        for y in coords:
            for z in coords:
                # Check FCC condition
                s = x + y + z
                if np.isclose(s, round(s)):
                    points.append((x, y, z))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    
    # Take the first n_points
    selected = points[:n_points]
    return np.array(selected, dtype=float)

def construct():
    # 1. Generate initial FCC structure
    # The nearest neighbor distance in LJ is 2^(1/6) ~ 1.122
    # The FCC lattice constant a relates to nearest neighbor distance r_nn = a / sqrt(2)
    # So a = r_nn * sqrt(2) ~ 1.587
    # My generator uses a=1 effectively (points like 0.5, 0.5, 0 are distance 0.707)
    # So I need to scale the generated points by a factor.
    # Let's generate with unit lattice constant (nearest neighbor 1/sqrt(2) ~ 0.707)
    # Then scale by 1.122 / 0.707 ~ 1.587 to get NN distance ~ 1.122
    
    init_points = _fcc_points(N, cutoff=3.0)
    
    # Scale to approximate LJ minimum distance
    # Distance of first shell in my generator: sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ~ 0.707
    # Target NN distance: 1.122
    scale = 1.122 / np.sqrt(0.5)
    init_points = init_points * scale
    
    # 2. Relax using L-BFGS-B
    # Define objective function
    def objective(x):
        pos = x.reshape(N, 3)
        # Optional: add penalty for atoms too close (< 0.5)
        # LJ repulsion is strong, so this might be unnecessary, but safe.
        dist_sq = np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :]) ** 2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        penalty = 0.0
        if min_dist_sq < 0.5**2:
            penalty = 1000.0 * (0.5**2 - min_dist_sq)
        return _lj_energy(pos) + penalty

    # Flatten positions for optimizer
    x0 = init_points.flatten()
    
    # Run optimization
    # Use L-BFGS-B as it handles bounds well (though we don't have explicit bounds, 
    # the potential handles them)
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-6})
        best_pos = res.x.reshape(N, 3)
        best_energy = res.fun
    except Exception:
        # Fallback to initial guess if optimization fails
        best_pos = init_points
        best_energy = _lj_energy(init_points)
        
    # Check if we found a valid solution
    # Ensure atoms are at least 0.5 apart
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    if np.min(dist_sq) < 0.5**2:
        # If invalid, return the initial FCC guess which is definitely valid
        # (FCC points are well separated)
        return init_points
        
    return best_pos
