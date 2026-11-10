# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _densify_path(path):
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)

def _point_in_base(u, v):
    # Base L-shape: {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}
    arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return arm1 | arm2

def _evaluate_path(path, grid_x, grid_y):
    dense = _densify_path(path)
    n_points = len(grid_x)
    inside = np.ones(n_points, dtype=bool)
    
    for i in range(len(dense)):
        x, y, a_deg = dense[i]
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Rotate point by -a: R(-a) * (p - c)
        dx = grid_x - x
        dy = grid_y - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside = inside & in_base
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _negative_area(params, grid_x, grid_y):
    n = len(params) // 3
    path = params.reshape(n, 3)
    # Ensure angles are in reasonable range
    area = _evaluate_path(path, grid_x, grid_y)
    return -area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    grid_x, grid_y = _build_grid()
    
    # Initial path: a reasonable starting guess
    # The path should go from angle ~0 to ~90
    # Let's use a smooth arc
    n_ctrl = 8
    angles = np.linspace(10, 80, n_ctrl)
    # x,y positions: inner corner should move appropriately
    # At angle 0, corner at (0,0); at angle 90, corner at (-2,0)
    # The corner should trace a path that allows maximum clearance
    t = np.linspace(0, 1, n_ctrl)
    x = -2.0 * t  # linear from 0 to -2
    y = 0.5 * np.sin(np.pi * t)  # slight arc
    
    initial_params = np.column_stack([x, y, angles]).flatten()
    
    best_params = initial_params.copy()
    best_area = _evaluate_path(initial_params.reshape(n_ctrl, 3), grid_x, grid_y)
    
    # Try multiple optimization strategies
    # Strategy 1: Nelder-Mead
    try:
        result = minimize(
            _negative_area,
            initial_params,
            args=(grid_x, grid_y),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001}
        )
        if result.fun < 0:
            area = -result.fun
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
    except:
        pass
    
    if time.time() > deadline:
        return best_params.reshape(n_ctrl, 3)
    
    # Strategy 2: COBYLA with more iterations
    try:
        result = minimize(
            _negative_area,
            best_params,
            args=(grid_x, grid_y),
            method='COBYLA',
            options={'maxiter': 300, 'rhobeg': 0.1}
        )
        if result.fun < 0:
            area = -result.fun
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
    except:
        pass
    
    if time.time() > deadline:
        return best_params.reshape(n_ctrl, 3)
    
    # Strategy 3: Try with more control points
    n_ctrl2 = 12
    t2 = np.linspace(0, 1, n_ctrl2)
    angles2 = np.linspace(8, 82, n_ctrl2)
    x2 = -2.0 * t2
    y2 = 0.5 * np.sin(np.pi * t2)
    initial_params2 = np.column_stack([x2, y2, angles2]).flatten()
    
    try:
        result = minimize(
            _negative_area,
            initial_params2,
            args=(grid_x, grid_y),
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.0001}
        )
        if result.fun < 0:
            area = -result.fun
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
                n_ctrl = n_ctrl2
    except:
        pass
    
    if time.time() > deadline:
        return best_params.reshape(n_ctrl, 3)
    
    # Local refinement
    try:
        result = minimize(
            _negative_area,
            best_params,
            args=(grid_x, grid_y),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 0.005, 'fatol': 0.00005}
        )
        if result.fun < 0:
            area = -result.fun
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
    except:
        pass
    
    return best_params.reshape(n_ctrl, 3)

# EVOLVE-BLOCK-END
