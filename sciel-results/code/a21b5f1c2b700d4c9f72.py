"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _energy(params, temperature=0.02):
    """
    Smooth energy function to maximize minimum distance.
    Uses soft-min approximation.
    """
    c = params.reshape(-1, 2)
    # Clip to stay inside box, slightly inside to avoid boundary issues in gradient
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    
    # Pairwise distances (half distance is the radius constraint)
    dists = pdist(c, metric='euclidean')
    pair_dists = dists / 2.0
    
    # Combine all constraints
    # We want to maximize min(boundary_dists, pair_dists)
    # Equivalent to minimizing -min(...)
    # Soft min: -T * log(sum(exp(-x/T)))
    # We use a combined array for efficiency if possible, but shapes differ.
    # Let's just take the soft min of the combined set.
    
    # To avoid overflow in exp, we can subtract the max (or min)
    # Actually, softmin(x) = -T * log(sum(exp(-x/T)))
    # Let y = -x/T. sum(exp(y)).
    # If x is large, y is large negative, exp is small.
    # If x is small (close to 0), y is close to 0, exp is ~1.
    
    all_dists = np.concatenate([boundary_dists.flatten(), pair_dists])
    
    # Safety check for NaNs or Infs
    if not np.all(np.isfinite(all_dists)):
        return 1e10
        
    # Soft min calculation
    # We want to approximate min(all_dists)
    # softmin = -T * log(sum(exp(-all_dists / T)))
    
    # Optimization: subtract max to prevent underflow/overflow?
    # exp(-x/T) where x >= 0.
    # If x is very large, exp is 0.
    # The term with smallest x dominates.
    
    scaled = -all_dists / temperature
    # max_scaled = np.max(scaled) # This corresponds to min(all_dists)
    # sum_exp = np.sum(np.exp(scaled - max_scaled))
    # softmin = max_scaled * temperature - temperature * np.log(sum_exp)
    # Wait, softmin(x) = -T log(sum exp(-x/T))
    # Let M = min(x). exp(-(x-M)/T) is stable.
    # sum exp(-(x-M)/T) = exp(M/T) sum exp(-x/T)
    # log(sum exp(-x/T)) = M/T + log(sum exp(-(x-M)/T))
    # softmin = -T (M/T + log(sum exp(-(x-M)/T))) = -M - T log(sum exp(-(x-M)/T))
    # As T->0, sum exp(-(x-M)/T) -> 1 (only term with x=M survives, exp(0)=1, others exp(-inf)=0)
    # log(1) = 0. softmin -> -M.
    # We want to MINIMIZE energy, which is -softmin? No.
    # We want to MAXIMIZE min_dist.
    # Let Energy = - min_dist.
    # Smooth Energy = - softmin(distances).
    
    m = np.min(all_dists)
    shifted = all_dists - m
    exp_terms = np.exp(-shifted / temperature)
    sum_exp = np.sum(exp_terms)
    
    # softmin = -m - temperature * np.log(sum_exp)
    # We want to minimize -softmin = m + temperature * np.log(sum_exp)
    # This is a smooth upper bound on the minimum.
    
    return m + temperature * np.log(sum_exp)

def _initialize_hexagonal(n_circles):
    """Generate initial positions using hexagonal lattice."""
    positions = []
    # Hexagonal lattice spacing
    # For n circles, we need roughly sqrt(n) rows
    # Adjusted for square aspect ratio
    rows = int(np.ceil(np.sqrt(n_circles * 2 / np.sqrt(3))))
    
    for row in range(rows + 1):
        for col in range(rows + 2):
            if len(positions) >= n_circles:
                break
            # Standard hex lattice coords
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            
            # Scale to fit in [-0.5, 0.5]
            # Approximate extent
            max_x = rows + 1
            max_y = rows * np.sqrt(3) / 2
            scale = 0.9 / max(max_x, max_y)
            
            x = x * scale - 0.45
            y = y * scale - 0.45
            
            positions.append([x, y])
    
    # If we don't have enough, add random points
    while len(positions) < n_circles:
        positions.append([np.random.uniform(-0.45, 0.45), 
                         np.random.uniform(-0.45, 0.45)])
    
    return np.array(positions[:n_circles])

def _initialize_grid(n_circles):
    """Generate initial positions using a square grid."""
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n_circles].copy()

def _local_optimize(initial_pos, max_iter=200, temperature=0.02):
    """Optimize positions starting from initial configuration using L-BFGS-B."""
    n = len(initial_pos)
    
    def objective(x):
        return _energy(x, temperature=temperature)
        
    # Use L-BFGS-B with box constraints
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    result = minimize(objective, initial_pos.flatten(), 
                     method='L-BFGS-B',
                     bounds=bounds,
                     options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    return result.x.reshape(n, 2)

def construct():
    """Construct optimal packing of 200 circles in unit square."""
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    best_pos = None
    best_score = -1.0
    
    np.random.seed(42)
    
    # Strategy 1: Hexagonal lattice with perturbations
    for trial in range(3):
        if time.time() > deadline:
            break
        
        pos = _initialize_hexagonal(N)
        # Add small random perturbations to break symmetry
        pos += np.random.normal(0, 0.01, size=pos.shape)
        
        try:
            # Coarse optimization
            optimized = _local_optimize(pos, max_iter=100, temperature=0.05)
            # Fine optimization
            optimized = _local_optimize(optimized, max_iter=200, temperature=0.01)
            
            score = _compute_radius(optimized)
            
            if score > best_score:
                best_score = score
                best_pos = optimized.copy()
        except Exception:
            continue

    # Strategy 2: Grid with perturbations
    if time.time() < deadline:
        pos = _initialize_grid(N)
        pos += np.random.normal(0, 0.02, size=pos.shape)
        
        try:
            optimized = _local_optimize(pos, max_iter=100, temperature=0.05)
            optimized = _local_optimize(optimized, max_iter=200, temperature=0.01)
            
            score = _compute_radius(optimized)
            
            if score > best_score:
                best_score = score
                best_pos = optimized.copy()
        except Exception:
            pass

    # Strategy 3: Pure random initialization
    if time.time() < deadline:
        pos = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        try:
            optimized = _local_optimize(pos, max_iter=100, temperature=0.05)
            optimized = _local_optimize(optimized, max_iter=200, temperature=0.01)
            
            score = _compute_radius(optimized)
            
            if score > best_score:
                best_score = score
                best_pos = optimized.copy()
        except Exception:
            pass
            
    # Final polish: try to improve the best solution found with very low temperature
    if best_pos is not None and time.time() < deadline - 20:
        try:
            polished = _local_optimize(best_pos, max_iter=500, temperature=0.001)
            score = _compute_radius(polished)
            
            if score > best_score:
                best_score = score
                best_pos = polished.copy()
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_pos is None:
        # Fallback to simple grid
        best_pos = _initialize_grid(N)
    
    # Verify and clip to valid range
    best_pos = np.clip(best_pos, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_pos
