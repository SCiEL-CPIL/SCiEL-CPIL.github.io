"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

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
        # Fallback to all pairs if Delaunay fails
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _create_constraints(params, pairs, use_delaunay=False):
    """
    Create constraint functions for SLSQP.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    pairs: list of (i, j) tuples for overlap constraints.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    constraints = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r, r >= 0
    # SLSQP uses 'ineq' meaning constraint >= 0
    for i in range(N):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
        # r_i >= 0 (handled by bounds, but explicit is safer)
        # constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2]})

    # Overlap constraints: dist(i,j) >= r_i + r_j  => dist(i,j) - r_i - r_j >= 0
    # To avoid sqrt in constraint for speed, we can square if both sides positive, 
    # but dist - (ri+rj) is standard. We'll use the direct form for stability.
    for i, j in pairs:
        def overlap_constraint(p, i=i, j=j):
            xi = p[3*i]
            yi = p[3*i+1]
            ri = p[3*i+2]
            xj = p[3*j]
            yj = p[3*j+1]
            rj = p[3*j+2]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - ri - rj
        constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
    return constraints

def _objective(params):
    """Objective function: maximize sum of radii => minimize -sum(r)."""
    r = params[2::3]
    return -np.sum(r)

def _fix_boundaries(xyz):
    """Ensure circles are within boundaries by clipping radii and positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        # Reduce radii slightly to create slack
        xyz_fixed[:, 2] *= 0.98
        # Clip positions
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def _optimize_slsqp(xyz0, global_start, deadline, max_iter_phase1=200, max_iter_phase2=100):
    """
    Optimize using SLSQP with explicit constraints.
    Phase 1: Delaunay neighbors (fast, fewer constraints)
    Phase 2: All pairs (accurate, more constraints)
    """
    params = xyz0.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5)) # Max radius bound
    
    # Phase 1: Delaunay-based constraints
    if _remaining(global_start, deadline) > 5.0:
        centers = np.column_stack([params[0::3], params[1::3]])
        edges = _get_delaunay_edges(centers)
        constraints = _create_constraints(params, edges)
        
        try:
            result = minimize(_objective, best_params, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': max_iter_phase1, 'ftol': 1e-9})
            if result.success and _validate(result.x.reshape(N, 3)):
                new_score = np.sum(result.x[2::3])
                if new_score > best_score:
                    best_params = result.x
                    best_score = new_score
            elif result.fun < _objective(best_params):
                # Even if not fully valid or not strictly better in objective, 
                # if it's a valid improvement in radius sum, take it.
                # But we must ensure validity for the "best" tracking.
                # Let's check validity of result.x
                res_xyz = result.x.reshape(N, 3)
                if _validate(res_xyz):
                    new_score = np.sum(res_xyz[:, 2])
                    if new_score > best_score:
                        best_params = result.x
                        best_score = new_score
        except Exception:
            pass

    # Phase 2: Full pairwise constraints for polish
    if _remaining(global_start, deadline) > 5.0:
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        constraints = _create_constraints(params, all_pairs)
        
        try:
            result = minimize(_objective, best_params, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': max_iter_phase2, 'ftol': 1e-10})
            
            res_xyz = result.x.reshape(N, 3)
            if _validate(res_xyz):
                new_score = np.sum(res_xyz[:, 2])
                if new_score > best_score:
                    best_params = result.x
                    best_score = new_score
            else:
                # If SLSQP failed to find a valid solution, try to fix the best_params
                # by local adjustments
                fixed = _fix_boundaries(best_params.reshape(N, 3))
                if _validate(fixed):
                    fixed_score = np.sum(fixed[:, 2])
                    if fixed_score > best_score:
                        best_params = fixed.flatten()
                        best_score = fixed_score
        except Exception:
            pass

    return best_params.reshape(N, 3)

def _initial_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.85
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_hex():
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
            r = 0.5 / side * 0.9
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_corner_weighted():
    """Initial configuration with circles weighted towards corners."""
    positions = []
    # Corner circles
    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.12])
    
    # Edge midpoints
    edges = [[0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
    for ex, ey in edges:
        positions.append([ex, ey, 0.10])
    
    # Center
    positions.append([0.5, 0.5, 0.11])
    
    # Fill remaining with random positions
    np.random.seed(123)
    for _ in range(N - len(positions)):
        x = np.random.uniform(0.2, 0.8)
        y = np.random.uniform(0.2, 0.8)
        r = np.random.uniform(0.06, 0.10)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        positions.append([x, y, r])
    
    return np.array(positions[:N])

def _initial_rings():
    """Concentric rings pattern."""
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
    for k in range(7):
        angle = 2 * np.pi * k / 7
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    return np.array(positions[:N])

def _remaining(global_start, deadline):
    return deadline - (time.time() - global_start)

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit (5s buffer)
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_initial_grid())
    initial_configs.append(_initial_hex())
    initial_configs.append(_initial_corner_weighted())
    initial_configs.append(_initial_rings())
    
    # Random configurations
    np.random.seed(42)
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
        
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        if _remaining(global_start, deadline) < 10.0:
            break
        
        # Adjust max iterations based on remaining time
        # SLSQP is slower. Estimate ~1-2s per phase for small N.
        # If we have 50s left, we can afford more iterations.
        # If we have 15s left, we should be quick.
        
        xyz_opt = _optimize_slsqp(xyz0, global_start, deadline)
        
        # Fix if needed
        if not _validate(xyz_opt):
            xyz_opt = _fix_boundaries(xyz_opt)
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation and fix
    if not _validate(best_xyz):
        best_xyz = _fix_boundaries(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
