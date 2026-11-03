"""Optimal packing of 50 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50


def _radius(pts):
    """Compute the maximum common radius for a set of centres."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    r_boundary = 0.5 - np.max(np.abs(pts), axis=1)
    if np.any(r_boundary < -1e-12):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return max(0.0, min(min_pair, float(np.min(r_boundary))))


def _neg_radius(x):
    """Objective: negative radius (to be minimized)."""
    pts = x.reshape(N, 2)
    r = _radius(pts)
    if r <= 0:
        # Infeasible: penalize heavily
        r_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Return a large penalty if out of bounds
        if np.any(r_boundary < 0):
            return 1.0
        return -min_pair + 1.0
    return -r


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    best_pts = None
    best_r = 0.0
    
    # Initial guess: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    best_pts = grid.copy()
    best_r = _radius(best_pts)
    
    # SLSQP optimization
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline - 30:
            break
        
        # Create initial point: grid with small random perturbation
        if restart == 0:
            x0 = best_pts.flatten().copy()
        else:
            x0 = (rng.random((N, 2)) - 0.5).flatten()
            # Scale to be inside square with margin
            x0 = x0 * 0.8
        
        try:
            result = minimize(
                _neg_radius,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < best_r:
                r_candidate = -result.fun
                if r_candidate > best_r:
                    best_pts = result.x.reshape(N, 2)
                    best_r = r_candidate
                    
        except Exception:
            continue
        
        # Polish with tighter tolerance
        if time.time() > deadline - 20:
            break
        
        try:
            result2 = minimize(
                _neg_radius,
                best_pts.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            if result2.success:
                r_candidate = -result2.fun
                if r_candidate > best_r:
                    best_pts = result2.x.reshape(N, 2)
                    best_r = r_candidate
                    
        except Exception:
            continue
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print(f"Radius: {_radius(pts)}")
