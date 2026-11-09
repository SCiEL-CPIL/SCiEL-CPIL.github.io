"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(points):
    """Compute the packing radius for a set of points."""
    if len(points) < 2:
        return np.inf
    # Use cdist for efficiency in double precision
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = np.min(dist_matrix)
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    return min(min_dist / 2.0, min_boundary)

def _hexagonal_lattice(n_target):
    """Generate a hexagonal lattice packing inside the unit disk."""
    # Estimate spacing: area per point ~ pi*r^2, total area pi. 
    # For hex packing, density ~ 0.9069.
    # r ~ 1 / sqrt(n * 2 / sqrt(3)) approx
    # Let's use a standard heuristic: d = 2 / sqrt(n) * adjustment
    d = 2.0 / (np.sqrt(n_target) * 0.95) 
    max_range = int(2.0 / d) + 3
    points = []
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * d + (j % 2) * d / 2.0
            y = j * d * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= n_target:
        # Select the n_target points closest to the center? 
        # Actually, for a lattice, it's better to select a contiguous block or just the first N if generated in order.
        # However, the lattice generation above generates a square block of hex points.
        # To ensure we have a good initial packing, we should probably just take the first N points 
        # but ensure they are well-distributed. 
        # A better initialization for the optimizer is a hex lattice clipped to the disk.
        # If we have more than N, we should remove the ones closest to the boundary?
        # Or just take the first N generated? The generation order is row by row.
        # Let's sort by distance to center and take the innermost N? 
        # No, that creates a hole in the middle? No, it fills the center.
        # But we need exactly N points.
        # If the lattice has more points, we need to remove some.
        # Removing points closest to the boundary is a good heuristic to keep the "core" tight?
        # Actually, for the optimizer, a good starting point is a valid packing.
        # If we just take the first N, they might not be balanced.
        # Let's try to generate exactly N or close to it.
        
        # Alternative: If we have too many, remove the ones with smallest distance to boundary?
        # That maximizes the minimum boundary distance initially.
        dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
        # Sort by distance to boundary, keep the largest ones?
        # This keeps points away from the edge.
        indices = np.argsort(dists_to_boundary)[::-1][:n_target]
        return points[indices]
    else:
        # If we don't have enough, we need to add random points or adjust d.
        # This shouldn't happen with the heuristic d.
        return points

def _soft_min_objective(flat_points, temperature=0.01):
    """
    Objective function to minimize.
    We want to maximize r = min(min_dist/2, 1 - max_norm).
    Equivalent to minimizing -r.
    We approximate min(a, b) using soft min: -log(sum(exp(-a/T), exp(-b/T))) * T.
    Here a = min_pair_dist / 2, b = 1 - max_norm.
    
    However, min_pair_dist is itself a min over pairs.
    Let D_ij = ||p_i - p_j||.
    Let B_i = 1 - ||p_i||.
    
    We want to maximize min( min_ij D_ij / 2, min_i B_i ).
    
    Let's define:
    E_rep = soft_min over pairs of D_ij (using log-sum-exp with negative sign? No, soft min is -T * log(sum(exp(-x/T))))
    E_bound = soft_min over points of B_i
    
    Total E = soft_min(E_rep / 2, E_bound)
    Objective = -Total E
    
    Note: Soft min is an approximation. As T -> 0, it approaches true min.
    For optimization, a small T (e.g., 0.01) is good, but can be stiff.
    """
    points = flat_points.reshape(-1, 2)
    n = len(points)
    
    # Boundary term
    norms = np.linalg.norm(points, axis=1)
    B = 1.0 - norms
    
    # Pairwise distances
    # Using cdist is O(N^2) memory and time. For N=300, 90k pairs. Manageable.
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # We need soft min of all D_ij / 2 and all B_i.
    # Let's combine them into one array of "constraints"
    # Constraints: D_ij / 2 for all i<j, and B_i for all i.
    
    # To save memory, we can compute the log-sum-exp terms separately?
    # log(sum(exp(-x/T))) = log(sum(exp(-D_ij/(2T))) * exp(-B_i/T)) ? No, it's a sum of all terms.
    
    # Let's collect the values.
    # Number of pairs: N*(N-1)/2 ~ 45000.
    # Number of boundary terms: N = 300.
    # Total terms ~ 45300.
    
    # Extract upper triangle for distances
    iu = np.triu_indices(n, k=1)
    D_vals = dist_matrix[iu] / 2.0
    
    # Combine
    # We want soft_min = -T * log( sum(exp(-val/T)) )
    # To avoid underflow/overflow in exp, we can use the log-sum-exp trick:
    # log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    # Here x_i = -val_i / T.
    
    all_vals = np.concatenate([D_vals, B])
    
    # x = -all_vals / T
    x = -all_vals / temperature
    
    # Numerical stability
    x_max = np.max(x)
    # If x_max is very large, exp(x - x_max) is fine.
    # If x_max is -inf (all vals inf?), handle edge case.
    
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
    
    soft_min_val = -temperature * log_sum_exp
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _objective_wrapper(flat_points, temperature):
    return _soft_min_objective(flat_points, temperature)

def construct():
    """Construct centers of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # 1. Initialization
    # Try hexagonal lattice
    init_points = _hexagonal_lattice(N)
    
    if init_points is None or len(init_points) != N:
        # Fallback: random points in disk
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        init_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Flatten for L-BFGS-B
    x0 = init_points.flatten()
    
    # Check initial score
    best_points = init_points.copy()
    best_radius = _compute_radius(best_points)
    
    # 2. Optimization using L-BFGS-B
    # We can't use bounds directly for the disk constraint in L-BFGS-B easily 
    # without a custom penalty or by mapping. 
    # However, the soft-min objective includes the boundary term, 
    # so the gradient will push points inward if they go too close to the edge.
    # We should still provide loose bounds to prevent points from flying off to infinity.
    # Bounds: [-1.5, 1.5] for each coordinate.
    bounds = [(-1.5, 1.5)] * (N * 2)
    
    # Temperature schedule?
    # Start with a higher temperature to allow large moves, then decrease?
    # L-BFGS-B is a local optimizer. It works best on smooth landscapes.
    # A fixed small temperature (e.g., 0.05) might be good.
    # Let's try a few restarts with different temperatures if time permits.
    
    temperatures = [0.1, 0.05, 0.02, 0.01]
    
    current_x = x0
    for T in temperatures:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                _objective_wrapper,
                current_x,
                args=(T,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < -best_radius: # res.fun is -soft_min, so lower is better
                new_points = res.x.reshape(-1, 2)
                # Project back to disk? The soft min penalizes going outside, 
                # but it might not strictly enforce it if T is large.
                # We must ensure validity.
                norms = np.linalg.norm(new_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    new_points[mask] = new_points[mask] / norms[mask, None] * 0.999999
                
                new_radius = _compute_radius(new_points)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
                    current_x = new_points.flatten()
                    
        except Exception as e:
            # If optimization fails, continue with next temperature
            pass
            
    # 3. Final local refinement?
    # L-BFGS-B usually converges well. 
    # We can do a few steps of coordinate descent or simple local search if time remains.
    
    if time.time() < deadline - 5.0:
        # Simple local search: perturb points slightly and see if radius improves
        # This is expensive if done naively. 
        # Let's just do a final projection and check.
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            best_points[mask] = best_points[mask] / norms[mask, None] * 0.999999
        
    return best_points
# EVOLVE-BLOCK-END
