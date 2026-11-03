"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def _make_hexagonal_init(n, scale=0.95):
    """Generate initial points from a hexagonal lattice inside a disk of radius `scale`."""
    # Estimate spacing so we get roughly n points
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            pts.append((x, y))
    pts = np.array(pts)
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r <= scale]
    if len(pts) < n:
        # Fallback: uniform random
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(rng.uniform(0, scale**2, n))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    else:
        # Sort by radius (center first) and take first n
        idx = np.argsort(r, kind='stable')
        pts = pts[idx[:n]]
    return pts


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    dists = pdist(centres)
    min_pair = np.min(dists) / 2.0
    boundary = 1.0 - np.linalg.norm(centres, axis=1)
    min_bound = np.min(boundary)
    return min(min_pair, min_bound)


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    
    Approach:
    1. Start with a hexagonal lattice initialization.
    2. Use a smooth, differentiable objective with least_squares:
       - Residuals for pairs: (2*r_target - d_ij) when d_ij < 2*r_target
       - Residuals for boundary: (r_target - (1 - |c_i|)) when 1-|c_i| < r_target
       where r_target is updated iteratively.
    3. Use a bisection-style approach: try increasing r_target until infeasible.
    """
    n = N
    t_start = time.time()
    
    # Step 1: Initial guess from hexagonal lattice
    x0 = _make_hexagonal_init(n, scale=0.95).flatten()
    
    # Step 2: Define smooth residuals for a given target radius r_t
    def make_residuals(r_t, smooth=50.0):
        """
        Return a residual function for target radius r_t.
        
        We use a smooth approximation of max(0, x):
        softplus(x) = log(1 + exp(x)) / smooth
        This is differentiable everywhere and approximates max(0, x) when smooth is large.
        """
        def residuals(x):
            centres = x.reshape(n, 2)
            
            # Pairwise distances
            dists = pdist(centres)
            
            # Boundary distances
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            
            # Pair residuals: want d_ij >= 2*r_t
            # residual = softplus(2*r_t - d_ij)
            pair_deficits = 2 * r_t - dists
            pair_res = np.log1p(np.exp(smooth * pair_deficits)) / smooth
            
            # Boundary residuals: want 1 - |c_i| >= r_t
            # residual = softplus(r_t - (1 - |c_i|))
            bound_deficits = r_t - boundary_dists
            bound_res = np.log1p(np.exp(smooth * bound_deficits)) / smooth
            
            return np.concatenate([pair_res, bound_res])
        
        return residuals
    
    # Step 3: Iteratively increase r_t and optimize
    # Start with current radius
    centres = x0.reshape(n, 2)
    r_current = _compute_radius(centres)
    
    # We want to push r_t up from r_current
    r_low = r_current
    r_high = r_current * 1.15  # Try 15% higher
    
    best_centres = centres.copy()
    best_r = r_current
    
    # Number of optimization attempts
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        if time.time() - t_start > 110:
            break
        
        # Try to optimize for r_high
        residuals_fn = make_residuals(r_high, smooth=30.0)
        
        try:
            result = least_squares(
                residuals_fn,
                x0,
                method='trf',
                max_nfev=8000,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
                verbose=0
            )
            
            new_centres = result.x.reshape(n, 2)
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r and np.all(np.isfinite(new_centres)):
                best_r = new_r
                best_centres = new_centres.copy()
                x0 = new_centres.flatten()
                
                # If we achieved r_high or better, try higher
                if new_r >= r_high * 0.99:
                    r_low = r_high
                    r_high = r_high * 1.08
                else:
                    # Didn't reach r_high, bisect
                    r_high = (r_low + r_high) / 2
            else:
                # Failed to improve, reduce target
                r_high = (r_low + r_high) / 2
                
        except Exception:
            r_high = (r_low + r_high) / 2
    
    # Final polish: one more optimization with the current best
    if time.time() - t_start < 115:
        r_target = best_r * 1.01
        residuals_fn = make_residuals(r_target, smooth=50.0)
        try:
            result = least_squares(
                residuals_fn,
                best_centres.flatten(),
                method='trf',
                max_nfev=5000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                verbose=0
            )
            final_centres = result.x.reshape(n, 2)
            final_r = _compute_radius(final_centres)
            
            if final_r > best_r and np.all(np.isfinite(final_centres)):
                best_centres = final_centres
                best_r = final_r
        except Exception:
            pass
    
    # Validate result
    if best_r <= 0 or not np.isfinite(best_r) or not np.all(np.isfinite(best_centres)):
        # Fallback
        return _make_hexagonal_init(n, scale=0.95)
    
    return best_centres
# EVOLVE-BLOCK-END
