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
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _objective_delaunay(params, edges_cache):
    """
    Objective using Delaunay-based constraints.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    # Get Delaunay edges
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edges_cache['edges'] = edges
    
    # Overlap penalties only for Delaunay neighbors
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap ** 2
    
    # Also check a few nearest neighbors not in Delaunay to be safe
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edges or (j, i) in edges:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            max_r = max(r[i], r[j])
            if dist_sq < (4 * max_r * max_r):
                dist = np.sqrt(dist_sq)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2
    
    return -np.sum(r) + 50.0 * penalty

def _objective_full(params):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + 100.0 * penalty

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
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

def _optimize_delaunay(xyz, time_budget=30.0):
    """
    Optimize using Delaunay-based constraints, then polish with full pairwise.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    edges_cache = {'edges': None}
    
    # Phase 1: Delaunay-based optimization (faster, fewer constraints)
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    try:
        result = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(edges_cache,),
                          options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_delaunay(best_params, edges_cache):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget:
        return best_params.reshape(N, 3)
    
    # Phase 2: Polish with full pairwise constraints
    try:
        result = minimize(_objective_full, best_params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_full(best_params):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations
    np.random.seed(42)
    for trial in range(4):
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
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        positions.append([x, y, 0.10])
    for k in range(16):
        angle = 2 * np.pi * k / 16
        x = 0.5 + 0.45 * np.cos(angle)
        y = 0.5 + 0.45 * np.sin(angle)
        positions.append([x, y, 0.08])
    initial_configs.append(np.array(positions[:N]))
    
    # 5. Corner-heavy pattern
    positions = []
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        positions.append([cx, cy, 0.15])
    for i in range(22):
        angle = 2 * np.pi * i / 22
        x = 0.5 + 0.35 * np.cos(angle)
        y = 0.5 + 0.35 * np.sin(angle)
        positions.append([x, y, 0.10])
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize each initial configuration
    for idx, init_xyz in enumerate(initial_configs):
        if time.time() - global_start > deadline - 10:
            break
        
        # Fix initial config if needed
        init_xyz = _fix_config(init_xyz)
        if not _validate(init_xyz):
            continue
        
        score = np.sum(init_xyz[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = init_xyz.copy()
        
        # Time budget per optimization
        remaining = deadline - 15 - (time.time() - global_start)
        if remaining < 5:
            break
        
        time_budget = min(25.0, remaining * 0.8)
        
        try:
            opt_xyz = _optimize_delaunay(init_xyz, time_budget=time_budget)
            
            # Fix and validate
            opt_xyz = _fix_config(opt_xyz)
            if _validate(opt_xyz):
                opt_score = np.sum(opt_xyz[:, 2])
                if opt_score > best_score:
                    best_score = opt_score
                    best_xyz = opt_xyz.copy()
        except Exception:
            pass
    
    # Final SLSQP polish with explicit constraints if time allows
    if time.time() - global_start < deadline - 15:
        try:
            def flatten(xyz):
                return xyz.flatten()
            
            def unflatten(v):
                return v.reshape(N, 3)
            
            def slsqp_objective(v):
                xyz = unflatten(v)
                return -np.sum(xyz[:, 2])
            
            slsqp_bounds = []
            for i in range(N):
                slsqp_bounds.append((0, 1))  # x
                slsqp_bounds.append((0, 1))  # y
                slsqp_bounds.append((0, 0.5))  # r
            
            slsqp_cons = []
            # Wall constraints
            for i in range(N):
                slsqp_cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
                slsqp_cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
                slsqp_cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
                slsqp_cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
            # Non-overlap
            for i in range(N):
                for j in range(i + 1, N):
                    def nonoverlap(v, i=i, j=j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        return np.sqrt(dx*dx + dy*dy) - v[3*i+2] - v[3*j+2]
                    slsqp_cons.append({'type': 'ineq', 'fun': nonoverlap})
            
            x0 = flatten(best_xyz)
            result = minimize(slsqp_objective, x0, method='SLSQP',
                             bounds=slsqp_bounds, constraints=slsqp_cons,
                             options={'maxiter': 100, 'ftol': 1e-10})
            
            if result.success:
                res_xyz = unflatten(result.x)
                res_xyz = _fix_config(res_xyz)
                if _validate(res_xyz):
                    res_score = np.sum(res_xyz[:, 2])
                    if res_score > best_score:
                        best_score = res_score
                        best_xyz = res_xyz.copy()
        except Exception:
            pass
    
    return best_xyz

# EVOLVE-BLOCK-END
