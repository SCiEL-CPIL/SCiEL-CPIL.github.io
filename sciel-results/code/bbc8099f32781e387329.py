"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 300


def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice packing inside the unit disk."""
    # Start with a reasonable spacing and refine
    # For n points in unit disk, approximate spacing
    # Area per circle ~ pi/(n * (2r)^2) but let's just use empirical
    # For n=300, r~0.053, spacing ~ 0.106
    # Lattice spacing for hexagonal: s = 2r
    # Number of points in hex lattice with spacing s in radius 1:
    # ~ pi/(s^2 * sqrt(3)/2) = 2*pi/(s^2*sqrt(3))
    # For n=300: s^2 = 2*pi/(300*sqrt(3)) ~ 0.0121, s ~ 0.11
    
    # Let's use a parameter and search
    best_points = None
    best_score = -1
    
    for scale in np.linspace(0.09, 0.14, 50):
        # Hexagonal lattice
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        # Generate points
        points = []
        # Estimate how many rows/cols needed
        max_r = 1.0
        n_rows = int(2 * max_r / dy) + 2
        n_cols = int(2 * max_r / dx) + 2
        
        for i in range(-n_cols, n_cols + 1):
            for j in range(-n_rows, n_rows + 1):
                x = i * dx + (0.5 * dx if j % 2 != 0 else 0)
                y = j * dy
                if x * x + y * y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) < n_points:
            continue
        
        # Compute score for this configuration
        # Get distances between all pairs (approximate for large n)
        # For now, just check if we have enough points and estimate radius
        # The actual radius will be determined by optimization
        
        # Sort by distance from center, take closest n_points
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:n_points]
        pts = points[idx]
        
        # Estimate minimum distance
        if len(pts) > 1:
            # Compute pairwise distances (can be expensive for 300 points)
            # Use a simple approach: compute all pairwise distances
            from scipy.spatial.distance import cdist
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            min_dist = np.min(d)
            est_r = min_dist / 2
            # Also check boundary
            boundary = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
            est_r = min(est_r, boundary)
        else:
            est_r = 0
        
        if est_r > best_score:
            best_score = est_r
            best_points = pts
    
    if best_points is None:
        # Fallback: random
        best_points = np.random.randn(n_points, 2) * 0.5
        best_score = 0
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Get initial hex lattice configuration
    points, init_r = _hex_lattice_init(N)
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # We'll optimize positions and radius simultaneously
    # Variables: [x1, y1, x2, y2, ..., xN, yN, r]
    # Constraints:
    # 1. For each pair (i,j): ||c_i - c_j|| >= 2r
    # 2. For each i: ||c_i|| + r <= 1
    
    def objective(x):
        # Maximize r, so minimize -r
        r = x[-1]
        return -r
    
    def constraint_nonoverlap(x):
        # Return array of constraints: ||c_i - c_j||^2 - (2r)^2 >= 0
        # This is expensive for N=300 (44850 pairs)
        # We'll use a subset or approximate
        pts = x[:-1].reshape(-1, 2)
        r = x[-1]
        
        # Compute pairwise squared distances
        # For efficiency, we can use a grid-based approach or just compute all
        # For N=300, that's 44850 pairs, which is manageable
        n = len(pts)
        constraints = []
        
        # Use vectorized computation
        # d2[i,j] = ||c_i - c_j||^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Constraint: d2 >= (2r)^2 for all i != j
        # We only need the minimum, but for SLSQP we need all constraints
        # This would be 44850 constraints, which is too many for SLSQP
        
        # Alternative: use a smooth approximation
        # min_{i<j} (d2_ij - 4r^2) >= 0
        # Use soft-min: -log(sum(exp(-alpha*(d2_ij - 4r^2)))) / alpha
        # But this is also expensive
        
        # Better approach: only enforce constraints for nearby pairs
        # Or use a different optimization strategy
        
        return None  # Placeholder
    
    # Given the complexity of explicit constraints for 300 points,
    # let's use a different approach: iterative local optimization
    
    # Strategy: 
    # 1. Start with hex lattice
    # 2. Iteratively push points apart using gradient-based methods
    # 3. Use a penalty method or augmented Lagrangian
    
    # Simpler approach: use scipy.optimize.minimize with a penalty function
    def penalty_objective(x, mu=100.0):
        pts = x[:2*N].reshape(-1, 2)
        r = x[2*N]
        
        # Objective: maximize r
        obj = -r
        
        # Penalty for violations
        penalty = 0.0
        
        # Non-overlap penalty
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Violation: max(0, 2r - sqrt(d2))^2
        d = np.sqrt(d2)
        violation = np.maximum(0, 2*r - d)
        penalty += mu * np.sum(violation ** 2)
        
        # Boundary penalty
        norm_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary_violation = np.maximum(0, np.hypot(pts[:, 0], pts[:, 1]) + r - 1.0)
        penalty += mu * np.sum(boundary_violation ** 2)
        
        return obj + penalty
    
    # Initial guess
    x_init = np.concatenate([points.flatten(), [init_r]])
    
    # Bounds
    # Positions: [-1, 1] for each coordinate
    # Radius: [0, 0.1]
    bounds = []
    for _ in range(2 * N):
        bounds.append((-1.0, 1.0))
    bounds.append((0.0, 0.1))
    
    # Optimize
    best_x = x_init
    best_obj = penalty_objective(x_init)
    
    # Try multiple iterations with increasing penalty
    for mu in [10.0, 100.0, 1000.0, 10000.0]:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                penalty_objective,
                x_init,
                args=(mu,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
                x_init = result.x
        except Exception:
            pass
    
    # Extract final positions and radius
    pts = best_x[:2*N].reshape(-1, 2)
    r_opt = best_x[2*N]
    
    # Compute actual score
    # Check non-overlap
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_dist = np.sqrt(np.min(d2))
    
    # Check boundary
    norm_pts = np.hypot(pts[:, 0], pts[:, 1])
    max_norm = np.max(norm_pts)
    
    actual_r = min(min_dist / 2, 1.0 - max_norm)
    
    # If the optimized r is larger than actual, we need to adjust
    # Return the points that give the best actual score
    
    # Do a final polish: binary search on radius
    # Find the largest r such that we can place N circles of radius r
    
    # For now, return the current best
    return pts


# Test
if __name__ == "__main__":
    points = construct()
    print(f"Shape: {points.shape}")
    print(f"First few points: {points[:5]}")
    
    # Compute score
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_dist = np.sqrt(np.min(d2))
    
    norm_pts = np.hypot(points[:, 0], points[:, 1])
    max_norm = np.max(norm_pts)
    
    r = min(min_dist / 2, 1.0 - max_norm)
    print(f"Radius: {r}")
    print(f"Score: {r / 0.0531542991469}")
