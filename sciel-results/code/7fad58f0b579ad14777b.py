"""Circle packing in a unit square: 200 equal circles, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200


def _score(pts):
    """Compute the maximum common radius for the given centres."""
    # Distance to boundaries
    d_bound = np.minimum(
        np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])),
        np.array([0.5] * N)
    )
    # Minimum distance to boundary
    min_bound = np.min(d_bound)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_bound, min_pair)


def _init_hexagonal():
    """Generate a hexagonal lattice packing as initial configuration."""
    pts = []
    # Hexagonal close packing
    # For N=200, use a hexagonal arrangement
    # Start with a grid and shift alternate rows
    
    # Estimate grid size
    m = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    
    # Generate hexagonal points
    for row in range(m):
        y = (row + 0.5) / m * 1.0 - 0.5
        offset = (0.5 / m) if row % 2 == 0 else 0.0
        n_cols = m
        for col in range(n_cols):
            x = (col + 0.5) / m * 1.0 - 0.5 + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    # If we have too many, take first N
    if len(pts) >= N:
        pts = pts[:N]
    else:
        # Pad with random points if needed
        while len(pts) < N:
            pts = np.vstack([pts, [np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)]])
    
    return pts


def _objective_and_grad(pts_flat, pts_3d):
    """Objective: negative of minimum distance (to maximize)."""
    pts = pts_flat.reshape(-1, 2)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Distance to boundaries
    d_bound = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    
    # Minimum of all constraints
    min_val = min(np.min(dist) / 2.0, np.min(d_bound))
    
    # Compute gradient
    grad = np.zeros_like(pts_flat)
    
    # Find the minimum pair
    min_pair_dist = np.min(dist)
    min_idx = np.argmin(dist)
    i, j = np.unravel_index(min_idx, dist.shape)
    
    if min_val == min_pair_dist / 2.0:
        # Gradient from pair distance
        d_ij = pts[i] - pts[j]
        norm = np.linalg.norm(d_ij)
        if norm > 1e-15:
            grad[2*i] -= d_ij[0] / norm
            grad[2*i+1] -= d_ij[1] / norm
            grad[2*j] += d_ij[0] / norm
            grad[2*j+1] += d_ij[1] / norm
    else:
        # Gradient from boundary
        min_bound_idx = np.argmin(d_bound)
        if d_bound[min_bound_idx] == 0.5 - np.abs(pts[min_bound_idx, 0]):
            grad[2*min_bound_idx] = -1 if pts[min_bound_idx, 0] > 0 else 1
        else:
            grad[2*min_bound_idx+1] = -1 if pts[min_bound_idx, 1] > 0 else 1
    
    return -min_val, grad


def _optimize(pts_init, time_budget):
    """Run optimization with time budget."""
    start = time.time()
    
    pts = pts_init.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Use L-BFGS-B for optimization
    def objective(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        d_bound = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        return -min(np.min(dist) / 2.0, np.min(d_bound))
    
    # Try multiple restarts
    n_restarts = 3
    pts_list = [pts_init.copy()]
    
    for r in range(n_restarts - 1):
        # Perturbed initial
        perturbed = pts_init + np.random.normal(0, 0.01, size=pts_init.shape)
        # Clip to valid region
        perturbed = np.clip(perturbed, -0.49, 0.49)
        pts_list.append(perturbed)
    
    for init_pts in pts_list:
        if time.time() - start > time_budget - 10:
            break
        
        x0 = init_pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 10000}
            )
            
            if result.success or result.fun < -best_score:
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    
        except Exception:
            pass
    
    # Polish with finer settings
    if time.time() - start < time_budget - 5:
        x0 = best_pts.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result.fun < -best_score:
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts


def construct():
    """Construct centres of 200 equal circles inside [-1/2, 1/2]^2."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initialize with hexagonal lattice
    pts_init = _init_hexagonal()
    
    # Ensure valid initial points
    pts_init = np.clip(pts_init, -0.49, 0.49)
    
    # Run optimization with time budget
    time_budget = 130  # Leave 10s for safety
    best_pts = _optimize(pts_init, time_budget)
    
    # Check if we're still within deadline
    if time.time() < deadline:
        # Try a few more random restarts if time permits
        for _ in range(2):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            pts_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
            
            # Quick optimization
            def objective(x):
                pts = x.reshape(-1, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist, np.inf)
                d_bound = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
                return -min(np.min(dist) / 2.0, np.min(d_bound))
            
            try:
                result = minimize(
                    objective,
                    pts_rand.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 300, 'maxfun': 5000}
                )
                
                new_score = _score(result.x.reshape(-1, 2))
                if new_score > _score(best_pts):
                    best_pts = result.x.reshape(-1, 2)
            except Exception:
                pass
    
    # Final validation: ensure all points are within bounds
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    # Verify validity
    score = _score(best_pts)
    if score <= 0 or not np.all(np.isfinite(best_pts)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
