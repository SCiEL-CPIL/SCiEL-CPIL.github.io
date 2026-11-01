import numpy as np
from scipy.optimize import minimize
import time

N = 200

def _initialize_hexagonal(n):
    """
    Initialize points in a hexagonal lattice pattern inside a square.
    This is a better starting point for packing than a square grid.
    """
    # Approximate number of rows
    # Hexagonal packing density is higher.
    # Let's just create a grid of hex points and pick the first n.
    # Or generate a dense hex grid and crop.
    
    # Side length of hexagon is related to spacing.
    # For n points, spacing approx 1/sqrt(n).
    # Let's generate a grid with slightly more points and crop.
    
    # Hex lattice vectors
    # a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
    # Spacing s.
    # Number of points in square of side 1 with spacing s is approx 1/s^2.
    # We want n points, so s ~ 1/sqrt(n).
    
    s = 1.0 / np.sqrt(n)
    
    # Generate grid
    # x from -0.5 to 0.5
    # y from -0.5 to 0.5
    # Hex rows: y = k * s * sqrt(3)/2
    # x offset for even/odd rows: 0 or s/2
    
    points = []
    y = -0.5
    row = 0
    while y <= 0.5:
        x_offset = 0 if row % 2 == 0 else s / 2
        x = -0.5 + x_offset
        while x <= 0.5:
            points.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
        
    points = np.array(points)
    
    # If we have too many, we need to select n.
    # Simple selection: take first n? No, that's biased to corner.
    # Better: random selection from the lattice points?
    # Or, if we have exactly n, great.
    # Usually we will have more.
    
    if len(points) >= n:
        # Select n points randomly from the lattice to break symmetry and cover area
        indices = np.random.choice(len(points), n, replace=False)
        selected = points[indices]
    else:
        # If not enough, pad with random points (should not happen with s=1/sqrt(n))
        # But let's ensure we have at least n
        while len(points) < n:
            points = np.vstack([points, np.random.uniform(-0.4, 0.4, 2)])
        indices = np.random.choice(len(points), n, replace=False)
        selected = points[indices]
        
    # Add small noise
    selected += np.random.normal(0, 0.001, selected.shape)
    selected = np.clip(selected, -0.49, 0.49)
    
    return selected

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    # Distance to boundaries
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    
    # Pairwise distances
    # Use scipy for efficiency if available, but manual is fine for N=200
    # N=200 -> ~20k pairs.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices_from(dists, k=1)
    min_pair = np.min(dists[iu]) / 2.0
    
    return min(min_boundary, min_pair)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    # 1. Initialization
    x0 = _initialize_hexagonal(N)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_x = x0.copy()
    best_radius = _compute_radius(best_x)
    
    # 2. Optimization Strategy: Repulsive Energy
    # Minimize E = sum(1/d^p) + sum(1/b^p)
    # As p increases, this focuses on the minimum distance.
    
    powers = [2, 4, 8, 16, 32]
    
    def energy(x, p):
        x = x.reshape(-1, 2)
        
        # Boundary distances
        b_dists = np.min(np.stack([
            x[:, 0] + 0.5,
            0.5 - x[:, 0],
            x[:, 1] + 0.5,
            0.5 - x[:, 1]
        ], axis=1), axis=1)
        
        # Clip to avoid division by zero or negative
        b_dists = np.clip(b_dists, 1e-6, None)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices_from(dists, k=1)
        p_dists = dists[iu]
        
        # Clip pairwise
        p_dists = np.clip(p_dists, 1e-6, None)
        
        # Energy
        e_b = np.sum(1.0 / (b_dists ** p))
        e_p = np.sum(1.0 / (p_dists ** p))
        
        return e_b + e_p
    
    current_x = x0.copy()
    
    for p in powers:
        if time.time() > deadline - 10:
            break
            
        # Optimize with current power
        try:
            res = minimize(
                lambda x: energy(x, p),
                current_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < energy(current_x, p):
                current_x = res.x
                r = _compute_radius(current_x)
                if r > best_radius:
                    best_radius = r
                    best_x = current_x.copy()
        except Exception:
            continue
            
    # 3. Local Polish with Nelder-Mead on the radius itself?
    # Nelder-Mead is derivative-free and can polish the result.
    # However, the radius function is non-smooth.
    # Let's try a few restarts of L-BFGS-B with very high power on the best found.
    
    if time.time() < deadline - 5:
        # Final polish with high power
        p_final = 64
        try:
            res = minimize(
                lambda x: energy(x, p_final),
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < energy(best_x, p_final):
                r = _compute_radius(res.x)
                if r > best_radius:
                    best_radius = r
                    best_x = res.x
        except Exception:
            pass
            
    return best_x.reshape(N, 2)
