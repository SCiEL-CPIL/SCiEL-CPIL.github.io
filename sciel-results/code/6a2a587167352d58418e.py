"""Packing 200 equal circles in a unit square via hexagonal lattice + force-directed + SLSQP polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
DEADLINE = 110.0  # seconds, safety margin before 150s kill


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 2):
        return 0.0
    # Wall distances
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    wall_min = np.min(walls)
    
    # Minimum pairwise distance / 2
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_min = np.min(dists[:, 1]) / 2.0
    
    return min(wall_min, pair_min)


def _hex_init(n):
    """Generate initial positions using a hexagonal lattice pattern."""
    best = None
    best_r = -1
    
    for orientation in [0, np.pi/6, np.pi/3]:
        m = int(np.ceil(np.sqrt(n)))
        
        for scale in np.linspace(0.8, 1.2, 30):
            s = scale / m
            dx = s
            dy = s * np.sqrt(3) / 2
            
            pts = []
            for i in range(m):
                for j in range(m):
                    x = (j - (m-1)/2) * dx
                    y = (i - (m-1)/2) * dy
                    if i % 2 == 1:
                        x += dx / 2
                    pts.append([x, y])
            
            pts = np.array(pts[:n])
            
            if orientation != 0:
                c, s_rot = np.cos(orientation), np.sin(orientation)
                R = np.array([[c, -s_rot], [s_rot, c]])
                pts = pts @ R.T
            
            max_abs = np.max(np.abs(pts))
            if max_abs > 0.45:
                pts = pts * (0.45 / max_abs)
            
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best = pts.copy()
    
    return best if best is not None else _square_init(n)


def _square_init(n):
    """Fallback square grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _force_relax(pts, grid_anchor, time_limit=20.0):
    """
    Force-directed relaxation: repulsion between nearby pairs + attraction to grid.
    This is a structural change from pure SLSQP - uses physics-based optimization.
    """
    start_time = time.time()
    pts = pts.copy()
    current_r = _compute_radius(pts)
    
    repulsion_strength = 1.0
    attraction_strength = 0.1
    learning_rate = 0.002
    max_iters = 800
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsion between nearby pairs using KDTree
        tree = cKDTree(pts)
        cutoff = max(0.02, current_r * 3.0)
        pairs = tree.query_pairs(cutoff)
        
        for i, j in pairs:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                # Force magnitude proportional to 1/dist^2
                force = repulsion_strength / (dist * dist)
                forces[i] += force * diff / dist
                forces[j] -= force * diff / dist
        
        # Attraction to grid positions (soft anchor)
        forces += attraction_strength * (grid_anchor - pts)
        
        # Update positions
        pts += learning_rate * forces
        
        # Keep points in bounds
        pts = np.clip(pts, -0.49, 0.49)
        
        # Check improvement
        new_r = _compute_radius(pts)
        if new_r > current_r:
            current_r = new_r
        else:
            # Reduce learning rate if no improvement
            learning_rate *= 0.95
            if learning_rate < 1e-7:
                break
    
    return pts, current_r


def _slsqp_polish(centers, r_target, time_limit=10.0):
    """
    SLSQP feasibility check: can we achieve radius >= r_target?
    Returns (success, improved_centers).
    """
    start_time = time.time()
    pts = centers.copy()
    
    # Build constraints for close pairs
    tree = cKDTree(pts)
    cutoff = 2.5 * r_target * 2
    pairs = tree.query_pairs(cutoff)
    
    if len(pairs) == 0:
        walls = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        if np.min(walls) >= r_target:
            return True, pts
        return False, pts
    
    # Limit constraints for tractability
    max_constraints = 600
    if len(pairs) > max_constraints:
        pair_dists = []
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            pair_dists.append((d, i, j))
        pair_dists.sort()
        pairs = [(i, j) for d, i, j in pair_dists[:max_constraints]]
    
    def constraint_pairs(x):
        x = x.reshape(-1, 2)
        vals = []
        for i, j in pairs:
            diff = x[i] - x[j]
            vals.append(np.dot(diff, diff) - (2 * r_target) ** 2)
        return np.array(vals)
    
    def constraint_walls(x):
        x = x.reshape(-1, 2)
        vals = np.concatenate([
            x[:, 0] + 0.5 - r_target,
            0.5 - x[:, 0] - r_target,
            x[:, 1] + 0.5 - r_target,
            0.5 - x[:, 1] - r_target
        ])
        return vals
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_pairs},
        {'type': 'ineq', 'fun': constraint_walls}
    ]
    
    def objective(x):
        x = x.reshape(-1, 2)
        return np.sum((x - pts) ** 2)
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective, x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 150, 'ftol': 1e-9}
        )
        
        if result.success or result.fun < 1e-10:
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r >= r_target - 1e-9:
                return True, new_pts
        
        new_pts = result.x.reshape(-1, 2)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        new_r = _compute_radius(new_pts)
        if new_r >= r_target - 1e-9:
            return True, new_pts
            
    except Exception:
        pass
    
    return False, pts


def _local_optimize(centers, time_limit=40.0):
    """
    Combined local optimization:
    1. Force-directed relaxation to escape local minima
    2. Bisection on target radius with SLSQP feasibility
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Force-directed relaxation with grid anchor
    grid = _square_init(N)
    relaxed, r_relaxed = _force_relax(centers, grid, time_limit=min(15.0, time_limit * 0.3))
    if r_relaxed > best_radius:
        best_radius = r_relaxed
        best_centers = relaxed.copy()
    
    # Phase 2: Bisection on target radius with SLSQP
    r_low = best_radius
    r_high = best_radius * 1.3
    
    # Try to expand upper bound
    for _ in range(4):
        if time.time() - start_time > time_limit - 10:
            break
        r_test = r_high
        feasible, new_pts = _slsqp_polish(best_centers, r_test, time_limit=5.0)
        if feasible:
            r_low = r_test
            best_centers = new_pts
            best_radius = r_test
            r_high = r_test * 1.15
        else:
            r_high = r_test * 0.92
    
    # Bisect
    for _ in range(8):
        if time.time() - start_time > time_limit - 5:
            break
        
        r_mid = (r_low + r_high) / 2
        feasible, new_centers = _slsqp_polish(best_centers, r_mid, time_limit=8.0)
        
        if feasible:
            r_low = r_mid
            best_centers = new_centers
            best_radius = r_mid
        else:
            r_high = r_mid
        
        if r_high - r_low < 1e-7:
            break
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    
    # Initial position: hexagonal lattice
    centers = _hex_init(N)
    current_r = _compute_radius(centers)
    
    # Local optimization with force-directed + SLSQP bisection
    centers, r = _local_optimize(centers, time_limit=60.0)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    best_centers = centers.copy()
    best_r = r
    
    # Try restarts with perturbations
    if remaining > 15:
        for restart in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            perturbed = best_centers + np.random.normal(0, 0.004, (N, 2))
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
            
            time_per_restart = min(remaining - 5, 25.0)
            perturbed, r2 = _local_optimize(perturbed, time_limit=time_per_restart)
            
            if r2 > best_r:
                best_r = r2
                best_centers = perturbed.copy()
    
    # Final validation and clipping
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
    
    return best_centers
