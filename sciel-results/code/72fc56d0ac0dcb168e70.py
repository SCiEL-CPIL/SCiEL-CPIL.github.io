"""Improved packing: hexagonal grid init + simulated annealing + local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # seconds, leaving margin before 150s kill

def _compute_radius(centers):
    """Compute the valid radius for a set of centers in [-0.5, 0.5]^2."""
    # Distance to boundary
    x, y = centers[:, 0], centers[:, 1]
    boundary_dist = np.minimum(0.5 - np.abs(x), 0.5 - np.abs(y))
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return min_boundary
    # Use broadcasting for pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)


def _hex_init(n):
    """Generate hexagonal packing initial configuration in [-0.5, 0.5]^2."""
    # Hexagonal lattice: rows offset by half spacing
    # Try to fit n circles in a hex pattern
    # Start with a reasonable number of columns
    best_centers = None
    best_r = 0
    
    for cols in range(4, 12):
        rows = int(np.ceil(n / cols))
        # Hex spacing
        # In a hex lattice, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
        # But we need to fit in [-0.5, 0.5]^2
        # Let's parameterize by the hex unit
        # Horizontal: cols * 2*r <= 1 => r <= 0.5/cols
        # Vertical: (rows-1)*sqrt(3)*r + 2*r <= 1 => r <= 1/(2 + (rows-1)*sqrt(3))
        
        r_h = 0.5 / cols
        r_v = 1.0 / (2.0 + (rows - 1) * np.sqrt(3))
        r = min(r_h, r_v)
        
        # Build hex grid
        centers = []
        for row in range(rows):
            for col in range(cols):
                x = -0.5 + r + col * 2 * r
                y_offset = 0 if row % 2 == 0 else r
                y = -0.5 + r + row * np.sqrt(3) * r + y_offset
                if len(centers) < n:
                    centers.append([x, y])
        
        if len(centers) < n:
            continue
        
        centers = np.array(centers[:n])
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers


def _simulated_annealing(centers, t_start=1.0, t_end=0.001, n_iter=50000, seed=42):
    """Simulated annealing to improve packing."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    n = len(current)
    
    for i in range(n_iter):
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Random perturbation
        delta = rng.randn(n, 2) * 0.05
        proposed = current + delta
        
        # Check if any point went out of bounds significantly
        if np.any(np.abs(proposed) > 0.6):
            continue
        
        proposed_r = _compute_radius(proposed)
        
        # Accept if better or with probability exp((new-old)/t)
        if proposed_r >= current_r:
            current = proposed
            current_r = proposed_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
        else:
            delta_r = proposed_r - current_r
            if delta_r > -0.1:  # Only accept if not too bad
                prob = np.exp(delta_r / t)
                if rng.random() < prob:
                    current = proposed
                    current_r = proposed_r
    
    return best, best_r


def _local_polish(centers):
    """Local optimization using scipy to maximize radius."""
    n = len(centers)
    
    def neg_radius(flat):
        c = flat.reshape(n, 2)
        # Penalize out-of-bounds
        if np.any(np.abs(c) > 0.5):
            return -0.001
        r = _compute_radius(c)
        return -r
    
    x0 = centers.flatten()
    result = minimize(neg_radius, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    if result.success or -result.fun > _compute_radius(centers):
        return result.x.reshape(n, 2)
    return centers


def construct():
    start_time = time.time()
    
    # Initialize with hexagonal pattern
    centers = _hex_init(N)
    current_r = _compute_radius(centers)
    best = centers.copy()
    best_r = current_r
    
    # Simulated annealing
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        anneal_centers, anneal_r = _simulated_annealing(centers, t_start=0.5, t_end=0.001, n_iter=30000)
        if anneal_r > best_r:
            best = anneal_centers
            best_r = anneal_r
    
    # Local polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15:
        polished = _local_polish(best)
        polished_r = _compute_radius(polished)
        if polished_r > best_r:
            best = polished
            best_r = polished_r
    
    # Final check
    final_r = _compute_radius(best)
    if final_r > best_r:
        best_r = final_r
    
    # Ensure all points are within bounds
    best = np.clip(best, -0.5, 0.5)
    
    return best
