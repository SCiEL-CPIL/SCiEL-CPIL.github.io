"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    # Check bounds
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    # Check pairwise non-overlap using Delaunay for efficiency
    try:
        pts = np.column_stack([x, y])
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                i, j = simplex[k], simplex[(k+1)%3]
                if i > j:
                    i, j = j, i
                edges.add((i, j))
        for i, j in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
        # Also check all pairs for safety (Delaunay may miss some close pairs)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-7:
                    return False
    except Exception:
        # Fallback: full O(N^2) check
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-7:
                    return False
    return True

def _get_delaunay_pairs(pts):
    """Get pairs of points that are close (Delaunay neighbors + buffer)."""
    n = pts.shape[0]
    try:
        tri = Delaunay(pts)
        pairs = set()
        for simplex in tri.simplices:
            for k in range(3):
                i, j = simplex[k], simplex[(k+1)%3]
                if i > j:
                    i, j = j, i
                pairs.add((i, j))
        # Add buffer: also include pairs within 2.5x min distance
        if len(pairs) > 0:
            min_dist = np.inf
            for i, j in pairs:
                d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                min_dist = min(min_dist, d)
            threshold = 2.5 * min_dist
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in pairs:
                        d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                        if d < threshold:
                            pairs.add((i,j))
        return list(pairs)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(n) for j in range(i+1, n)]

def _objective(params):
    """
    params: array of length 3*n: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Objective: minimize -sum(r_i) + penalty for violations
    Vectorized for speed.
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for going out of bounds (vectorized)
    penalty = 0.0
    lb_x = np.maximum(0, r - x)
    penalty += np.sum(lb_x ** 2)
    ub_x = np.maximum(0, x + r - 1)
    penalty += np.sum(ub_x ** 2)
    lb_y = np.maximum(0, r - y)
    penalty += np.sum(lb_y ** 2)
    ub_y = np.maximum(0, y + r - 1)
    penalty += np.sum(ub_y ** 2)
    neg_r = np.maximum(0, -r)
    penalty += np.sum(neg_r ** 2)
    
    # Penalty for overlap (vectorized over all pairs)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    overlap = np.maximum(0, sum_r - dist)
    # Only upper triangle
    triu = np.triu_indices(n, 1)
    penalty += np.sum(overlap[triu] ** 2)
    
    return -np.sum(r) + 100.0 * penalty

def _initial_config_hex():
    """
    Generate an initial configuration using a hexagonal-like arrangement.
    We'll use a grid with slight perturbations.
    """
    # Use a 5x5 grid (25 circles) plus one more
    # Actually, let's use a smarter initial layout
    # Start with a 5x5 grid
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            # Offset odd rows for hexagonal packing
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    # Add the 26th circle - try to place it in a gap
    # For now, just add another at a random-ish position
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    
    return np.array(positions[:N])

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _optimize_from(xyz, time_budget=30.0):
    """
    Optimize from a given initial configuration.
    Returns the best valid configuration found.
    """
    start_time = time.time()
    params = xyz.flatten()
    
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Use L-BFGS-B with multiple penalty weights
    for penalty_weight in [10.0, 50.0, 100.0, 300.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p, pw=penalty_weight):
            n = N
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            
            # Vectorized penalty
            penalty = 0.0
            lb_x = np.maximum(0, r - x)
            penalty += np.sum(lb_x ** 2)
            ub_x = np.maximum(0, x + r - 1)
            penalty += np.sum(ub_x ** 2)
            lb_y = np.maximum(0, r - y)
            penalty += np.sum(lb_y ** 2)
            ub_y = np.maximum(0, y + r - 1)
            penalty += np.sum(ub_y ** 2)
            neg_r = np.maximum(0, -r)
            penalty += np.sum(neg_r ** 2)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            sum_r = r[:, None] + r[None, :]
            overlap = np.maximum(0, sum_r - dist)
            triu = np.triu_indices(n, 1)
            penalty += np.sum(overlap[triu] ** 2)
            
            return -np.sum(r) + pw * penalty
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if result.fun < obj(best_params):
                best_params = result.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass
        
        if time.time() - start_time > time_budget:
            break
    
    # Final polish: high-precision optimization with tight tolerances
    if time.time() - start_time < time_budget - 1.0:
        def obj_polish(p):
            n = N
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            
            penalty = 0.0
            lb_x = np.maximum(0, r - x)
            penalty += np.sum(lb_x ** 2)
            ub_x = np.maximum(0, x + r - 1)
            penalty += np.sum(ub_x ** 2)
            lb_y = np.maximum(0, r - y)
            penalty += np.sum(lb_y ** 2)
            ub_y = np.maximum(0, y + r - 1)
            penalty += np.sum(ub_y ** 2)
            neg_r = np.maximum(0, -r)
            penalty += np.sum(neg_r ** 2)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            sum_r = r[:, None] + r[None, :]
            overlap = np.maximum(0, sum_r - dist)
            triu = np.triu_indices(n, 1)
            penalty += np.sum(overlap[triu] ** 2)
            
            return -np.sum(r) + 500.0 * penalty
        
        try:
            result = minimize(obj_polish, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
            if result.fun < obj_polish(best_params):
                best_params = result.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass
    
    # Final validation
    xyz_best = best_params.reshape(N, 3)
    if _validate(xyz_best):
        return xyz_best
    
    # If invalid, try to fix by reducing radii
    xyz_fixed = xyz_best.copy()
    for _ in range(200):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        xyz_fixed[:, 0] = np.clip(xyz_fixed[:, 0], xyz_fixed[:, 2], 1 - xyz_fixed[:, 2])
        xyz_fixed[:, 1] = np.clip(xyz_fixed[:, 1], xyz_fixed[:, 2], 1 - xyz_fixed[:, 2])
    
    if _validate(xyz_fixed):
        return xyz_fixed
    
    return _initial_config_grid()

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations with different seeds
    for seed in [42, 123, 456, 789, 101]:
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 4. Concentric rings pattern
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.12 * 1.8 * np.cos(angle)
        y = 0.5 + 0.12 * 1.8 * np.sin(angle)
        positions.append([x, y, 0.12 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.10 * 3.5 * np.cos(angle)
        y = 0.5 + 0.10 * 3.5 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    
    initial_configs.append(np.array(positions[:N]))
    
    # 5. Corner-weighted random (circles near corners can be larger)
    np.random.seed(2024)
    x = np.random.uniform(0.05, 0.95, N)
    y = np.random.uniform(0.05, 0.95, N)
    r = np.random.uniform(0.06, 0.14, N)
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    initial_configs.append(np.column_stack([x, y, r]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 8.0:
            break
        
        # Give more time to earlier configs, less to later ones
        time_budget = min(25.0, remaining - 3.0)
        if time_budget < 3.0:
            break
        
        xyz_opt = _optimize_from(xyz0, time_budget=time_budget)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final polish on the best configuration
    elapsed = time.time() - global_start
    if elapsed < deadline - 5.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 2.0:
            xyz_polished = _optimize_from(best_xyz, time_budget=remaining)
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_polished
    
    # Final check
    if not _validate(best_xyz):
        best_xyz = best_xyz.copy()
        for _ in range(300):
            if _validate(best_xyz):
                break
            best_xyz[:, 2] *= 0.95
            best_xyz[:, 0] = np.clip(best_xyz[:, 0], best_xyz[:, 2], 1 - best_xyz[:, 2])
            best_xyz[:, 1] = np.clip(best_xyz[:, 1], best_xyz[:, 2], 1 - best_xyz[:, 2])
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
