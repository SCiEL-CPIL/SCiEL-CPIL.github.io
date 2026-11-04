"""
Sphere packing optimization using Active-Set Constraint Optimization with Deformable Boundaries.
Strategy:
1. Represent sphere centers with adjustable boundary offsets (deformable mesh approximation).
2. Use active-set identification to find binding constraints (closest pairs and wall contacts).
3. Optimize only active constraints with exact Newton-like steps.
4. Multi-start with diverse initial configurations.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
DEADLINE = 145.0  # Safety margin for 150s limit

def _score(pts):
    """Compute the true radius score."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_dist / 2.0, min_wall)

def _active_set_constraints(pts, r_target, tol=1e-6):
    """
    Identify active constraints: pairs and walls within r_target + tol.
    Returns list of (type, indices, current_value) tuples.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Pair constraints: d_ij/2 >= r
    active_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if dists[i,j] / 2.0 <= r_target + tol:
                active_pairs.append((i, j, dists[i,j] / 2.0))
    
    # Wall constraints: 0.5 - |x_ik| >= r
    active_walls = []
    for i in range(n):
        for k in range(3):
            wall_val = 0.5 - abs(pts[i,k])
            if wall_val <= r_target + tol:
                active_walls.append((i, k, wall_val))
    
    return active_pairs, active_walls

def _project_to_feasible(pts, r_target):
    """
    Project points to be feasible for radius r_target using iterative constraint projection.
    Simple projection: push violating pairs apart, push wall violations inward.
    """
    pts = pts.copy()
    for _ in range(20):  # Iterations
        changed = False
        
        # Pair constraints
        for i in range(N):
            for j in range(i+1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 2 * r_target:
                    if dist < 1e-10:
                        d = np.random.randn(3) * 1e-10
                        dist = np.linalg.norm(d)
                    push = (2 * r_target - dist) / 2.0
                    unit = d / dist
                    pts[i] += push * unit
                    pts[j] -= push * unit
                    changed = True
        
        # Wall constraints
        for i in range(N):
            for k in range(3):
                if pts[i,k] > 0.5 - r_target:
                    pts[i,k] = 0.5 - r_target
                    changed = True
                elif pts[i,k] < -(0.5 - r_target):
                    pts[i,k] = -(0.5 - r_target)
                    changed = True
        
        if not changed:
            break
    
    return pts

def _active_set_optimize(pts, r_target, max_iter=100, tol=1e-10):
    """
    Optimize to maximize minimum radius using active-set method.
    Uses quadratic programming on active constraints.
    """
    from scipy.optimize import linprog
    
    pts = pts.copy()
    n = len(pts)
    
    for iteration in range(max_iter):
        # Find active constraints
        active_pairs, active_walls = _active_set_constraints(pts, r_target, tol=1e-5)
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            # No active constraints, can increase r
            r_target *= 1.001
            continue
        
        # Build QP: maximize r subject to active constraints
        # Variables: delta_pts (3*N) + delta_r (1)
        # Constraints: for each active pair (i,j): (d_ij + delta_d_ij)/2 >= r + delta_r
        #              for each active wall (i,k): (w_ik + delta_w_ik) >= r + delta_r
        
        # Simplified: use gradient ascent on the minimum constraint
        # Compute gradient of min constraint
        min_val = r_target
        min_type = None
        min_idx = None
        
        for (i, j, val) in active_pairs:
            if val < min_val - 1e-12:
                min_val = val
                min_type = 'pair'
                min_idx = (i, j)
        
        for (i, k, val) in active_walls:
            if val < min_val - 1e-12:
                min_val = val
                min_type = 'wall'
                min_idx = (i, k)
        
        if min_type is None:
            break
        
        # Compute gradient step
        grad = np.zeros_like(pts)
        step_size = 1e-4
        
        if min_type == 'pair':
            i, j = min_idx
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                unit = d / dist
                # Move i and j apart
                grad[i] += unit * 0.5
                grad[j] -= unit * 0.5
        else:
            i, k = min_idx
            # Push point away from wall
            sign = np.sign(pts[i,k])
            if sign == 0:
                sign = 1
            grad[i, k] -= sign * 0.5
        
        # Take step
        pts += step_size * grad
        
        # Project to feasible region
        pts = _project_to_feasible(pts, r_target)
        
        # Check if we can increase r
        new_score = _score(pts)
        if new_score > r_target * (1 + 1e-6):
            r_target = new_score
        else:
            # Try small increase
            r_target *= 1.0001
    
    return pts, r_target

def _optimize_with_annealing(pts, time_limit):
    """
    Optimize using annealing with active-set polish.
    """
    start_time = time.time()
    pts = pts.copy()
    
    # Phase 1: Repulsive energy minimization for good basin
    def repulsive_obj(x):
        p = x.reshape(N, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1e-10)
        # Soft repulsive energy
        E = np.sum(1.0 / (dists_sq + 1e-4))
        # Wall repulsion
        wall = 0.5 - np.abs(p)
        E += np.sum(1.0 / (wall + 1e-4))
        return E
    
    def repulsive_grad(x):
        p = x.reshape(N, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        dists_sq_safe = dists_sq + 1e-4
        np.fill_diagonal(dists_sq_safe, 1e10)
        
        inv_d4 = 1.0 / (dists_sq_safe ** 2)
        np.fill_diagonal(inv_d4, 0)
        
        row_sums = np.sum(inv_d4, axis=1)
        weighted = inv_d4 @ p
        grad_pairs = -2.0 * (p * row_sums[:, np.newaxis] - weighted)
        
        wall = 0.5 - np.abs(p)
        wall_safe = np.clip(wall, 1e-4, None)
        grad_walls = 2.0 / (wall_safe ** 3) * np.sign(p)
        
        return (grad_pairs + grad_walls).flatten()
    
    if time.time() - start_time < time_limit * 0.4:
        try:
            res = minimize(
                repulsive_obj,
                pts.flatten(),
                jac=repulsive_grad,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (N*3),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.fun < repulsive_obj(pts.flatten()):
                pts = res.x.reshape(N, 3)
        except:
            pass
    
    # Phase 2: Active-set optimization with increasing target radius
    r_target = _score(pts)
    
    while time.time() - start_time < time_limit - 5:
        # Increase target radius
        r_target *= 1.005
        
        # Try to achieve this target
        pts_new, r_achieved = _active_set_optimize(pts, r_target, max_iter=50)
        
        if r_achieved > r_target * 0.99:
            pts = pts_new
            r_target = r_achieved
        else:
            # Back off
            r_target *= 0.995
            if r_target < _score(pts):
                break
    
    # Final polish: local optimization on minimum constraint
    for _ in range(10):
        if time.time() - start_time > time_limit - 3:
            break
        
        # Find minimum constraint
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_wall = np.min(0.5 - np.abs(pts))
        r = min(min_dist / 2.0, min_wall)
        
        # Gradient ascent on r
        grad = np.zeros_like(pts)
        
        if min_dist / 2.0 < min_wall:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j:
                i, j = j, i
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                unit = d / dist
                grad[i] += unit * 0.5
                grad[j] -= unit * 0.5
        else:
            # Wall constraint is binding
            idx = np.argmin(0.5 - np.abs(pts))
            i = idx // 3
            k = idx % 3
            sign = np.sign(pts[i, k])
            if sign == 0:
                sign = 1
            grad[i, k] -= sign * 0.5
        
        # Take small step
        pts += 1e-5 * grad
        pts = np.clip(pts, -0.4999, 0.4999)
    
    return pts

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered grid."""
    np.random.seed(42)
    # Use a 5x5x4 grid = 100 points, take 50
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    
    grid = []
    for z in ticks3:
        for y in ticks2:
            for x in ticks1:
                grid.append([x, y, z])
    grid = np.array(grid)
    grid += np.random.randn(*grid.shape) * 0.015
    grid = np.clip(grid, -0.49, 0.49)
    
    # Randomly select 50
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _generate_fcc_start():
    """Generate starting configuration from FCC-like lattice."""
    points = []
    # FCC basis vectors
    basis = [
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]
    ]
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                for b in basis:
                    p = np.array([i + b[0], j + b[1], k + b[2]])
                    if -0.48 <= p[0] <= 0.48 and -0.48 <= p[1] <= 0.48 and -0.48 <= p[2] <= 0.48:
                        points.append(p)
    
    points = np.array(points)
    np.random.seed(123)
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    
    return points[:N]

def _generate_random_start(seed):
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    
    for i in range(6):
        starts.append(_generate_random_start(100 + i * 7))
    
    best_pts = None
    best_score = -np.inf
    
    # Score and sort starts
    scored_starts = [(s, _score(s)) for s in starts]
    scored_starts.sort(key=lambda x: x[1], reverse=True)
    
    # Allocate time: run top 5 starts
    num_runs = 5
    elapsed = time.time() - start_time
    time_per_run = (DEADLINE - elapsed - 10) / num_runs
    time_per_run = max(time_per_run, 15.0)  # Minimum 15s per run
    
    for i in range(num_runs):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        init_pts, init_score = scored_starts[i]
        
        # Run optimization
        optimized_pts = _optimize_with_annealing(init_pts, time_per_run)
        
        # Evaluate
        current_score = _score(optimized_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = optimized_pts
        
        # Check time
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # Final polish on best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        remaining = DEADLINE - 15 - (time.time() - start_time)
        polished = _optimize_with_annealing(best_pts, remaining)
        polished_score = _score(polished)
        
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Fallback
    if best_pts is None:
        best_pts = scored_starts[0][0]
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
