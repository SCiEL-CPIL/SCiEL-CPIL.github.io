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
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(pts):
    """Get Delaunay edges from point cloud."""
    n = pts.shape[0]
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                i, j = simplex[k], simplex[(k + 1) % 3]
                if i > j:
                    i, j = j, i
                edges.add((i, j))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _objective_full(params):
    """
    Full pairwise objective with penalty for violations.
    params: array of length 3*n: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for going out of bounds
    penalty = 0.0
    penalty += np.sum(np.maximum(0, r - x) ** 2)
    penalty += np.sum(np.maximum(0, x + r - 1) ** 2)
    penalty += np.sum(np.maximum(0, r - y) ** 2)
    penalty += np.sum(np.maximum(0, y + r - 1) ** 2)
    penalty += np.sum(np.maximum(0, -r) ** 2)
    
    # Penalty for overlap (all pairs)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    overlap = np.maximum(0, sum_r - dist)
    triu = np.triu_indices(n, 1)
    penalty += np.sum(overlap[triu] ** 2)
    
    return -np.sum(r) + 100.0 * penalty

def _objective_delaunay(params, edges):
    """
    Objective using only Delaunay edges for overlap penalty.
    This exploits sparsity of the contact graph.
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for going out of bounds
    penalty = 0.0
    penalty += np.sum(np.maximum(0, r - x) ** 2)
    penalty += np.sum(np.sum(np.maximum(0, x + r - 1)) ** 2)
    penalty += np.sum(np.maximum(0, r - y) ** 2)
    penalty += np.sum(np.maximum(0, y + r - 1) ** 2)
    penalty += np.sum(np.maximum(0, -r) ** 2)
    
    # Penalty for overlap on Delaunay edges only
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = max(0, r[i] + r[j] - dist)
        penalty += overlap ** 2
    
    # Also check a few non-edge pairs that might be close
    # Use a threshold based on minimum edge length
    if len(edges) > 0:
        min_edge_len = np.inf
        for i, j in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_edge_len = min(min_edge_len, d)
        threshold = 2.5 * min_edge_len
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edges and (j, i) not in edges:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < threshold:
                        overlap = max(0, r[i] + r[j] - d)
                        penalty += overlap ** 2
    
    return -np.sum(r) + 100.0 * penalty

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    positions = []
    side = 5
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
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

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii."""
    xyz = xyz.copy()
    for _ in range(200):
        if _validate(xyz):
            break
        xyz[:, 2] *= 0.95
        xyz[:, 0] = np.clip(xyz[:, 0], xyz[:, 2], 1 - xyz[:, 2])
        xyz[:, 1] = np.clip(xyz[:, 1], xyz[:, 2], 1 - xyz[:, 2])
    return xyz

def _optimize_delaunay(xyz, time_budget=30.0):
    """
    Optimize using Delaunay-edge-based objective.
    Uses a grow-and-relax loop to push radii higher.
    """
    start_time = time.time()
    params = xyz.flatten()
    
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Get initial Delaunay edges
    pts = np.column_stack([params[0::3], params[1::3]])
    edges = _get_delaunay_edges(pts)
    
    # Phase 1: Optimize with Delaunay edges
    def obj(p):
        # Update edges periodically
        nonlocal edges
        p_pts = np.column_stack([p[0::3], p[1::3]])
        edges = _get_delaunay_edges(p_pts)
        return _objective_delaunay(p, edges)
    
    try:
        result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if result.fun < obj(best_params):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget:
        return _fix_config(best_params.reshape(N, 3))
    
    # Phase 2: Grow-and-relax loop
    # Try to increase radii and re-optimize
    for grow_iter in range(10):
        if time.time() - start_time > time_budget - 2.0:
            break
        
        # Try to grow radii slightly
        current_r = best_params[2::3]
        grow_factor = 1.0 + 0.02 * (10 - grow_iter) / 10
        new_r = np.clip(current_r * grow_factor, 0, 0.5)
        
        # Check if growth is feasible
        test_params = best_params.copy()
        test_params[2::3] = new_r
        test_pts = np.column_stack([test_params[0::3], test_params[1::3]])
        test_edges = _get_delaunay_edges(test_pts)
        
        # Check for overlaps
        has_overlap = False
        for i, j in test_edges:
            dx = test_params[3*i] - test_params[3*j]
            dy = test_params[3*i+1] - test_params[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < test_params[3*i+2] + test_params[3*j+2] - 1e-8:
                has_overlap = True
                break
        
        if has_overlap:
            # Too much growth, try smaller
            grow_factor = 1.0 + 0.01 * (10 - grow_iter) / 10
            new_r = np.clip(current_r * grow_factor, 0, 0.5)
            test_params = best_params.copy()
            test_params[2::3] = new_r
        
        # Optimize from the grown configuration
        test_pts = np.column_stack([test_params[0::3], test_params[1::3]])
        test_edges = _get_delaunay_edges(test_pts)
        
        def obj_grow(p, te=test_edges):
            p_pts = np.column_stack([p[0::3], p[1::3]])
            te = _get_delaunay_edges(p_pts)
            return _objective_delaunay(p, te)
        
        try:
            result = minimize(obj_grow, test_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            if result.fun < obj_grow(test_params):
                new_score = np.sum(result.x[2::3])
                if new_score > best_score:
                    best_params = result.x
                    best_score = new_score
        except Exception:
            pass
    
    # Final polish with full pairwise objective
    if time.time() - start_time < time_budget - 1.0:
        try:
            result = minimize(_objective_full, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if result.fun < _objective_full(best_params):
                best_params = result.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass
    
    xyz_best = best_params.reshape(N, 3)
    return _fix_config(xyz_best)

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 4. Corner-weighted random
    np.random.seed(2024)
    x = np.random.uniform(0.05, 0.95, N)
    y = np.random.uniform(0.05, 0.95, N)
    r = np.random.uniform(0.06, 0.14, N)
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    initial_configs.append(np.column_stack([x, y, r]))
    
    # 5. Ring pattern
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
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 10.0:
            break
        
        # Give more time to earlier configs
        time_budget = min(15.0, remaining - 5.0)
        if time_budget < 3.0:
            break
        
        xyz_opt = _optimize_delaunay(xyz0, time_budget=time_budget)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final polish on the best configuration
    elapsed = time.time() - global_start
    if elapsed < deadline - 5.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 3.0:
            xyz_polished = _optimize_delaunay(best_xyz, time_budget=remaining)
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_polished
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
