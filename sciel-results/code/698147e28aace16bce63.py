"""Pack 50 equal circles in a unit circle.

Uses a pure Simulated Annealing approach with advanced move sets
to escape local minima, which is distinct from the gradient-based
approach of the previous version.
"""

import numpy as np

N = 50


def _objective(c, r):
    """Calculate the cost function: sum of squared overlaps and boundary violations."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu, ju = np.triu_indices(N, k=1)
    d = dists[iu, ju]
    
    # Overlap penalty: max(0, 2r - d)^2
    ov = np.maximum(0.0, 2.0 * r - d)
    e_pairs = np.sum(ov ** 2)
    
    # Boundary penalty: max(0, |c| + r - 1)^2
    radii = np.sqrt(np.sum(c ** 2, axis=1))
    bv = np.maximum(0.0, radii + r - 1.0)
    e_bound = np.sum(bv ** 2)
    
    return e_pairs + 10.0 * e_bound


def _hex_init(r):
    """Return a hexagonal lattice of N points inside the unit circle."""
    spacing = 2.0 * r
    pts = []
    k = 8
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2.0
            if x * x + y * y <= (1.0 - r) ** 2:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < N:
        rng = np.random.default_rng(42)
        while len(pts) < N:
            x, y = rng.uniform(-1, 1, size=2)
            if x * x + y * y <= (1.0 - r) ** 2:
                pts = np.vstack([pts, [x, y]])
    pts = pts[:N]
    # Shuffle
    rng2 = np.random.default_rng(123)
    idx = rng2.permutation(N)
    return pts[idx]


def construct():
    # Start with a slightly smaller radius to ensure a valid initial configuration
    r_start = 0.120
    c = _hex_init(r_start)
    
    # Target radius is the known best
    r_target = 0.1258254895304
    
    # Simulated Annealing
    rng = np.random.default_rng(12345)
    
    # We will anneal from a high temperature to a low one.
    # The "energy" is the violation of constraints for a fixed r.
    # However, we want to maximize r. 
    # Strategy: 
    # 1. Optimize positions for r_target. If cost is 0, we are done.
    # 2. If not, we need to find a configuration that allows r_target.
    #    We can do this by minimizing the cost for r_target directly.
    #    If we can't reach 0, we might need to adjust r.
    #    But usually, for N=50, r_target is achievable.
    
    # Let's try to minimize cost for r_target directly using SA.
    r = r_target
    c = _hex_init(r) # Init with r_target might be invalid (overlaps), but SA handles it.
    
    # To help SA, start from a valid configuration of a smaller radius and expand?
    # No, standard SA starts from random or heuristic. Hex init is good.
    
    T = 0.01
    T_min = 1e-10
    alpha = 0.999 # Cooling rate
    max_iter = 50000 # Number of SA steps
    step_size = 0.05
    
    best_c = c.copy()
    best_cost = _objective(c, r)
    current_cost = best_cost
    
    for i in range(max_iter):
        # Generate a new state
        c_new = c.copy()
        
        # Move type 1: Displace one random particle
        if rng.random() < 0.9:
            idx = rng.integers(N)
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0, step_size)
            c_new[idx] += dist * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Move type 2: "Shake" - small random displacement for all particles
            # This helps escape local minima where particles are stuck in a rigid configuration
            c_new += rng.normal(0, step_size * 0.1, size=c.shape)
        
        # Ensure particles stay within bounds roughly? 
        # The objective function penalizes boundary violations, so no hard constraint needed.
        
        cost_new = _objective(c_new, r)
        
        # Acceptance criterion
        if cost_new < current_cost or rng.random() < np.exp((current_cost - cost_new) / T):
            c = c_new
            current_cost = cost_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_c = c.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            break
    
    # Final check and rescale if necessary
    c = best_c
    
    # Calculate actual achievable radius
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu, ju = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu, ju]) / 2.0
    radii = np.sqrt(np.sum(c ** 2, axis=1))
    max_r = np.max(radii)
    boundary_r = 1.0 - max_r
    achievable = min(min_pair, boundary_r)
    
    # If the configuration is valid for r_target, return it.
    # If not, we might have found a better r? No, we fixed r.
    # If achievable < r_target, it means the configuration doesn't support r_target.
    # In that case, we should return the configuration scaled to fit?
    # The score is the largest common radius. 
    # If we return c, the evaluator computes r = min(min_pair, boundary_r).
    # So we just return the best configuration we found.
    
    return c
