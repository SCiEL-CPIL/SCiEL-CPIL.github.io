"""Circle packing in perimeter-4 rectangle: Voronoi-based Soft-Body Optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, voronoi_plot_2d  # voronoi_plot_2d not used, just Voronoi
import math

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _compute_voronoi_radii(x, y, w, h, tol=1e-12):
    """
    Compute the maximum radius for each circle such that it fits within the rectangle
    and does not overlap with any other circle, based on Voronoi cell inradii.
    
    For each circle i, the max radius is the minimum distance to:
    1. The rectangle boundaries
    2. The Voronoi cell edges (which correspond to bisectors with neighbors)
    
    We use a robust method: for each point, find the nearest point on the Voronoi cell boundary.
    """
    n = len(x)
    points = np.column_stack((x, y))
    
    # Add boundary points to ensure Voronoi cells are bounded
    # We add points far outside to create a bounding box effect
    # Or better: manually compute distance to boundaries and to other points
    
    # Method: For each circle, r_i = min(
    #   x_i, w - x_i, y_i, h - y_i,  # distance to walls
    #   min_{j != i} (dist(i,j) / 2)  # distance to nearest neighbor bisector
    # )
    
    # This is actually the exact condition for disjoint interiors in a packing.
    # If r_i <= dist(i,j)/2 for all j, and r_i <= dist(i, wall), then circles are disjoint.
    # The "Voronoi" interpretation: the inradius of the Voronoi cell of i in the domain
    # restricted to the set of other points is exactly this minimum distance.
    
    r = np.zeros(n)
    for i in range(n):
        # Distance to walls
        d_wall = min(x[i], w - x[i], y[i], h - y[i])
        
        # Distance to other points
        min_d = np.inf
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_d:
                min_d = d
        
        # The max radius is half the distance to the nearest neighbor
        # (since two circles of radius r_i and r_j are disjoint if dist >= r_i + r_j)
        # If we set r_i = r_j = min_d/2, they touch.
        # But we can have different radii. The constraint is r_i + r_j <= dist(i,j).
        # To maximize sum of radii, we want to solve this system.
        # However, a simple heuristic that works well: set r_i = min(d_wall, min_d/2)
        # This guarantees validity.
        
        r[i] = min(d_wall, min_d / 2.0)
        
        # Clamp to small positive
        if r[i] < 1e-9:
            r[i] = 1e-9
            
    return r


def _objective(z):
    """
    Objective function: minimize negative sum of radii.
    z = [x_0, y_0, ..., x_N-1, y_N-1, w]
    Radii are computed from positions and w.
    """
    n = N
    x = z[0:n]
    y = z[n:2*n]
    w = z[2*n]
    h = 2.0 - w
    
    if w < 0.2 or h < 0.2:
        return 1e6
    
    # Compute radii
    r = _compute_voronoi_radii(x, y, w, h)
    
    # Add small penalty if any radius is too small (to encourage spread)
    # But the main driver is maximizing sum(r)
    
    val = -np.sum(r)
    
    # Add a small regularization to prevent circles from collapsing to the same point
    # which might cause numerical issues in gradient estimation
    # The Voronoi radius computation is non-smooth at points where the nearest neighbor changes.
    # This is acceptable for SLSQP with finite differences, but might be slow.
    
    return val


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _force_relax(x, y, w, h, max_iter=200, dt=0.01):
    """
    Simple force-directed relaxation to spread points apart.
    This helps escape local minima in the Voronoi radius landscape.
    """
    n = len(x)
    x = x.copy()
    y = y.copy()
    
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion (inverse square law, capped)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                d = np.sqrt(d2) + 1e-10
                # Force magnitude
                f = 1.0 / (d2 + 0.01)
                fx[i] += f * dx / d
                fy[i] += f * dy / d
                fx[j] -= f * dx / d
                fy[j] -= f * dy / d
        
        # Wall repulsion
        for i in range(n):
            # Left
            if x[i] < 0.1:
                fx[i] += (0.1 - x[i]) * 10
            # Right
            if x[i] > w - 0.1:
                fx[i] -= (x[i] - (w - 0.1)) * 10
            # Bottom
            if y[i] < 0.1:
                fy[i] += (0.1 - y[i]) * 10
            # Top
            if y[i] > h - 0.1:
                fy[i] -= (y[i] - (h - 0.1)) * 10
        
        # Update
        x += dt * fx
        y += dt * fy
        
        # Clamp to bounds
        x = np.clip(x, 0.01, w - 0.01)
        y = np.clip(y, 0.01, h - 0.01)
        
        # Check convergence
        max_f = np.max(np.abs(fx)) + np.max(np.abs(fy))
        if max_f < 1e-6:
            break
            
    return x, y


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_result = None
    
    def check_and_update(x, y, w):
        nonlocal best_value, best_result
        h = 2.0 - w
        r = _compute_voronoi_radii(x, y, w, h)
        
        # Verify validity
        if not _is_valid(x, y, r, w, h, tol=1e-9):
            return False, 0.0
            
        val = float(np.sum(r))
        if val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w)
        return True, val
    
    # Initial guesses: diverse aspect ratios and random points
    rng = np.random.default_rng(42)
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.4, 1.1, 0.9]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Generate random points
        x = rng.uniform(0.05, w0 - 0.05, N)
        y = rng.uniform(0.05, h0 - 0.05, N)
        
        # Force relax to spread them
        x_rel, y_rel = _force_relax(x, y, w0, h0, max_iter=100, dt=0.02)
        
        # Construct vector for SLSQP
        z0 = np.zeros(2 * N + 1)
        z0[0:N] = x_rel
        z0[N:2*N] = y_rel
        z0[2*N] = w0
        
        # Optimize
        bounds = [(0.01, w0 - 0.01)] * N + [(0.01, h0 - 0.01)] * N + [(0.3, 1.7)]
        
        # SLSQP with finite differences
        # The objective is non-smooth, so we use a small step size for gradient estimation
        try:
            res = minimize(_objective, z0, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-8, 'eps': 1e-4})
            
            z_opt = res.x
            x_opt = z_opt[0:N]
            y_opt = z_opt[N:2*N]
            w_opt = z_opt[2*N]
            
            check_and_update(x_opt, y_opt, w_opt)
            
        except Exception:
            continue
        
        # Also try the relaxed points directly without SLSQP
        check_and_update(x_rel, y_rel, w0)
        
        # Try a few random perturbations of the relaxed points
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            x_pert = x_rel + rng.normal(0, 0.05, N)
            y_pert = y_rel + rng.normal(0, 0.05, N)
            x_pert = np.clip(x_pert, 0.01, w0 - 0.01)
            y_pert = np.clip(y_pert, 0.01, h0 - 0.01)
            check_and_update(x_pert, y_pert, w0)

    # Phase 2: Polish the best result with more iterations and tighter tolerances
    if best_result is not None and time.time() < deadline - 15:
        x_best, y_best, r_best, w_best = best_result
        h_best = 2.0 - w_best
        
        # Run SLSQP again from the best point with more iterations
        z0 = np.zeros(2 * N + 1)
        z0[0:N] = x_best
        z0[N:2*N] = y_best
        z0[2*N] = w_best
        
        bounds = [(0.01, w_best - 0.01)] * N + [(0.01, h_best - 0.01)] * N + [(0.3, 1.7)]
        
        try:
            res = minimize(_objective, z0, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'eps': 1e-5})
            
            z_opt = res.x
            x_opt = z_opt[0:N]
            y_opt = z_opt[N:2*N]
            w_opt = z_opt[2*N]
            
            check_and_update(x_opt, y_opt, w_opt)
            
        except Exception:
            pass
        
        # Final force relaxation polish
        if time.time() < deadline - 5:
            x_pol, y_pol = _force_relax(x_best, y_best, w_best, h_best, max_iter=200, dt=0.005)
            check_and_update(x_pol, y_pol, w_best)

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r, w = best_result
    h = 2.0 - w
    
    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h, tol=1e-12):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
