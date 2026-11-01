"""Packing 200 equal circles in a unit square via hexagonal lattice + SLSQP NLP."""
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
        
        for scale in np.linspace(0.8, 1.2, 50):
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


def _slsqp_optimize(centers, r_target, time_limit=30.0):
    """
    Use SLSQP to find positions where all constraints are satisfied for radius r_target.
    Objective: minimize 0 (constant), subject to:
      - pairwise distances >= 2*r_target
      - wall distances >= r_target
    Returns (success, centers, achieved_radius)
    """
    start_time = time.time()
    n = len(centers)
    
    # Flatten centers to 1D for SLSQP
    x0 = centers.flatten()
    
    # Precompute pair indices for constraints
    pair_indices = []
    for i in range(n):
        for j in range(i+1, n):
            pair_indices.append((i, j))
    
    # Boundary constraints: for each point, 4 walls
    # We'll handle boundaries via bounds instead of constraints for efficiency
    
    def objective(x):
        return 0.0
    
    def objective_grad(x):
        return np.zeros(2*n)
    
    # Constraints: pairwise distances
    def make_pair_constraint(i, j):
        def constraint(x):
            di = x[2*i:2*i+2]
            dj = x[2*j:2*j+2]
            dist = np.sqrt(np.sum((di - dj)**2))
            return dist - 2*r_target
        return constraint
    
    # We can't easily pass all constraints as a list for 19900 pairs in SLSQP
    # Instead, use a single constraint function that returns an array
    
    def constraints(x):
        # Pairwise constraints
        pts = x.reshape(n, 2)
        # Only compute constraints for pairs that are close (within some threshold)
        # For efficiency, use KDTree to find nearby pairs
        tree = cKDTree(pts)
        threshold = 2*r_target + 0.05
        nearby = tree.query_pairs(threshold)
        
        cons = []
        for i, j in nearby:
            di = pts[i]
            dj = pts[j]
            dist = np.sqrt(np.sum((di - dj)**2))
            cons.append(dist - 2*r_target)
        
        # Wall constraints
        for i in range(n):
            x_i = pts[i, 0]
            y_i = pts[i, 1]
            cons.append(0.5 - x_i - r_target)  # right wall
            cons.append(x_i + 0.5 - r_target)  # left wall
            cons.append(0.5 - y_i - r_target)  # top wall
            cons.append(y_i + 0.5 - r_target)  # bottom wall
        
        return np.array(cons)
    
    # Bounds for each coordinate
    bounds = [(-0.5 + r_target, 0.5 - r_target)] * (2*n)
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': 50, 'ftol': 1e-8}
        )
        
        if result.success or result.status == 0:
            new_centers = result.x.reshape(n, 2)
            achieved_r = _compute_radius(new_centers)
            return True, new_centers, achieved_r
        else:
            return False, centers, _compute_radius(centers)
            
    except Exception:
        return False, centers, _compute_radius(centers)


def _bisection_slsqp(centers, time_limit=60.0):
    """
    Use bisection on radius r, with SLSQP to check feasibility.
    Start with current radius, try to increase.
    """
    start_time = time.time()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Upper bound: theoretical max is 0.5/sqrt(200) ≈ 0.0354, but hexagonal can do better
    # Actually for 200 circles, the max radius is around 0.0366
    r_low = current_r
    r_high = current_r * 1.15  # Try to improve by 15%
    
    # First, verify that current r is feasible
    success, _, _ = _slsqp_optimize(centers, r_low, time_limit=5.0)
    
    # Bisection
    max_iterations = 15
    for it in range(max_iterations):
        if time.time() - start_time > time_limit - 5:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Check if r_mid is feasible
        success, new_centers, achieved_r = _slsqp_optimize(centers, r_mid, time_limit=10.0)
        
        if success and achieved_r >= r_mid - 1e-10:
            # r_mid is feasible, try higher
            r_low = r_mid
            best_centers = new_centers.copy()
            best_r = max(best_r, achieved_r)
        else:
            # r_mid is not feasible, try lower
            r_high = r_mid
    
    return best_centers, best_r


def _local_refine(centers, time_limit=30.0):
    """
    Local refinement: find the bottleneck constraint and adjust.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    n = len(centers)
    
    step_size = 0.005
    n_iter = 0
    
    while time.time() - start_time < time_limit:
        # Compute current radius components
        walls = np.minimum(
            np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
            np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
        )
        wall_min = np.min(walls)
        wall_argmin = np.argmin(walls)
        
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=2)
        pair_min = np.min(dists[:, 1]) / 2.0
        pair_argmin = np.argmin(dists[:, 1])
        
        current_r = min(wall_min, pair_min)
        
        if current_r > best_radius + 1e-12:
            best_radius = current_r
            best_centers = centers.copy()
        
        # Determine which constraint is binding
        if wall_min < pair_min:
            # Move the point closest to wall away from wall
            i = wall_argmin
            if centers[i, 0] > 0:
                dir_x = -1
            else:
                dir_x = 1
            if centers[i, 1] > 0:
                dir_y = -1
            else:
                dir_y = 1
            
            norm = np.sqrt(dir_x**2 + dir_y**2)
            if norm > 0:
                centers[i, 0] += step_size * dir_x / norm
                centers[i, 1] += step_size * dir_y / norm
        else:
            # Move the pair with minimum distance apart
            i = pair_argmin
            j = idx[i, 1]
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                centers[i] += step_size * diff / norm
                centers[j] -= step_size * diff / norm
        
        # Clip to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        n_iter += 1
        
        # Adaptive step size
        if n_iter % 50 == 0:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    
    # Initial position: hexagonal lattice
    centers = _hex_init(N)
    current_r = _compute_radius(centers)
    
    # Phase 1: Local refinement to get a good starting point
    centers, current_r = _local_refine(centers, time_limit=20.0)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 2: SLSQP with bisection to push the radius higher
    if remaining > 20:
        centers, slsqp_r = _bisection_slsqp(centers, time_limit=min(remaining - 10, 60.0))
        if slsqp_r > current_r:
            current_r = slsqp_r
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 3: Final local refinement
    if remaining > 10:
        centers, final_r = _local_refine(centers, time_limit=min(remaining - 5, 20.0))
        if final_r > current_r:
            current_r = final_r
    
    # Final validation and clipping
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    return centers
