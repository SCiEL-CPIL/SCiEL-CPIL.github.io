"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def _hexagonal_initial(n, r_target=0.05):
    """Generate a hexagonal lattice initial guess with n points inside unit disk."""
    # Estimate spacing from target radius
    spacing = 2.0 * r_target * 1.01  # slight buffer
    # Generate hexagonal lattice points
    k = int(1.0 / (spacing * np.sqrt(3) / 2)) + 3
    points = []
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3) / 2
        offset = 0.5 * spacing if (j % 2 != 0) else 0.0
        for i in range(-k, k + 1):
            x = i * spacing + offset
            points.append([x, y])
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    # Keep points that can fit a circle of radius r_target
    mask = norms <= 1.0 - r_target
    points = points[mask]
    
    # If we have too many, select the n closest to origin (better packing)
    if len(points) > n:
        order = np.argsort(norms)
        points = points[order[:n]]
    elif len(points) < n:
        # Try smaller spacing
        spacing *= 0.95
        k = int(1.0 / (spacing * np.sqrt(3) / 2)) + 3
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3) / 2
            offset = 0.5 * spacing if (j % 2 != 0) else 0.0
            for i in range(-k, k + 1):
                x = i * spacing + offset
                points.append([x, y])
        points = np.array(points)
        norms = np.linalg.norm(points, axis=1)
        mask = norms <= 1.0 - r_target
        points = points[mask]
        if len(points) > n:
            order = np.argsort(norms)
            points = points[order[:n]]
        elif len(points) < n:
            # Fallback: grid
            xs = np.linspace(-0.9, 0.9, 20)
            ys = np.linspace(-0.9, 0.9, 20)
            grid = np.array([[x, y] for x in xs for y in ys])
            points = grid[:n]
    
    return points[:n]


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses bisection on target radius with least_squares feasibility checks.
    """
    n = N
    # Start with a reasonable target radius
    # Best known is ~0.05315, so start slightly below
    r_low = 0.045
    r_high = 0.055
    
    # Get initial configuration
    init_centres = _hexagonal_initial(n, r_target=0.05)
    x0 = init_centres.flatten()
    
    def compute_radius(centres):
        """Compute the actual radius for a given configuration."""
        if len(centres) < 2:
            return 0.0
        dists = pdist(centres)
        min_pair = np.min(dists) / 2.0
        norms = np.linalg.norm(centres, axis=1)
        min_bound = np.min(1.0 - norms)
        return min(min_pair, min_bound)
    
    def residuals(x, r_target):
        """
        Residuals for least_squares. We want to find centres such that:
        1. |c_i - c_j| >= 2*r_target for all i < j
        2. |c_i| + r_target <= 1 for all i
        
        Residuals are the violations (negative when satisfied).
        We minimize sum of squares of violations.
        """
        centres = x.reshape(n, 2)
        
        # Pairwise distance violations
        dists = pdist(centres)
        pair_violations = np.maximum(0.0, 2.0 * r_target - dists)
        
        # Boundary violations
        norms = np.linalg.norm(centres, axis=1)
        bound_violations = np.maximum(0.0, r_target - (1.0 - norms))
        
        return np.concatenate([pair_violations, bound_violations])
    
    def check_feasibility(r_target, x_init, max_iter=50):
        """
        Try to find a feasible configuration for the given r_target.
        Returns (success, x_solution, cost).
        """
        x = x_init.copy()
        
        # Run least_squares to minimize violations
        result = least_squares(
            residuals,
            x,
            args=(r_target,),
            method='trf',
            max_nfev=2000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10
        )
        
        cost = result.cost
        x_sol = result.x
        
        # Check if we achieved feasibility (cost close to 0)
        if cost < 1e-6:
            return True, x_sol, cost
        
        # If not fully feasible, try a few more iterations with tighter tolerance
        if cost < 1e-4:
            result2 = least_squares(
                residuals,
                x_sol,
                args=(r_target,),
                method='trf',
                max_nfev=1000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12
            )
            if result2.cost < 1e-6:
                return True, result2.x, result2.cost
        
        return False, x_sol, cost
    
    # Bisection search for the maximum feasible radius
    best_x = x0.copy()
    best_r = compute_radius(init_centres)
    
    # First, check if our initial guess is feasible at r_low
    feasible, x_sol, cost = check_feasibility(r_low, x0)
    if feasible:
        best_x = x_sol
        best_r = r_low
    
    # Bisection
    for iteration in range(20):
        r_mid = (r_low + r_high) / 2.0
        
        # Use the best solution as warm start
        feasible, x_sol, cost = check_feasibility(r_mid, best_x)
        
        if feasible:
            r_low = r_mid
            best_x = x_sol
            best_r = r_mid
        else:
            r_high = r_mid
        
        # Early termination if interval is small enough
        if r_high - r_low < 1e-7:
            break
    
    # Final refinement at the best radius
    if best_r > 0:
        feasible, x_final, cost_final = check_feasibility(best_r, best_x)
        if feasible:
            best_x = x_final
    
    centres = best_x.reshape(n, 2)
    
    # Compute actual radius
    actual_r = compute_radius(centres)
    
    # Validate
    if actual_r <= 0 or not np.isfinite(actual_r):
        # Fallback: return the initial configuration
        return init_centres
    
    # Ensure all values are finite
    if not np.all(np.isfinite(centres)):
        return init_centres
    
    return centres
# EVOLVE-BLOCK-END
