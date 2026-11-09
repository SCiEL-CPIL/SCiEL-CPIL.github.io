# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Find centers of 50 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the minimum distance to walls and between circles.
    
    Uses a constraint-satisfaction formulation:
    Maximize r
    Subject to:
      1. Centers within bounds adjusted by r
      2. Pairwise distances >= 2r
    """
    N = 50
    
    # Objective: minimize -r
    def objective(x):
        # x = [x_1, y_1, ..., x_N, y_N, r]
        return -x[-1]
    
    # Vectorized Constraints
    def wall_constraints(x):
        xs = x[0::2]
        ys = x[1::2]
        r = x[-1]
        # 0.5 - x_i - r >= 0
        # 0.5 + x_i - r >= 0
        # 0.5 - y_i - r >= 0
        # 0.5 + y_i - r >= 0
        return np.concatenate([
            0.5 - xs - r,
            0.5 + xs - r,
            0.5 - ys - r,
            0.5 + ys - r
        ])
        
    def pairwise_constraints(x):
        coords = x[:2*N].reshape(N, 2)
        r = x[-1]
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        return (dists - 2*r).flatten()
        
    constraints = [
        {'type': 'ineq', 'fun': wall_constraints},
        {'type': 'ineq', 'fun': pairwise_constraints}
    ]
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]
    
    # Initial guess: grid layout
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0_coords = grid[:N].flatten()
    
    # Estimate initial r
    centers0 = x0_coords.reshape(N, 2)
    min_wall = np.min(np.stack([
        0.5 - np.abs(centers0[:, 0]),
        0.5 - np.abs(centers0[:, 1])
    ]))
    diffs = centers0[:, np.newaxis, :] - centers0[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    r0 = min(min_wall, min_pair) * 0.9
    x0_full = np.append(x0_coords, r0)
    
    best_x = x0_full.copy()
    best_r = r0
    
    np.random.seed(42)
    
    # Try multiple random initializations
    for trial in range(30):
        if trial == 0:
            x_start = x0_full.copy()
        elif trial < 15:
            x_start_coords = x0_coords + np.random.randn(2 * N) * 0.02
            x_start_coords = np.clip(x_start_coords, -0.5, 0.5)
            r_start = r0 * (0.8 + 0.4 * np.random.rand())
            x_start = np.append(x_start_coords, r_start)
        else:
            x_start_coords = np.random.uniform(-0.4, 0.4, size=2 * N)
            r_start = 0.05
            x_start = np.append(x_start_coords, r_start)
            
        try:
            result = minimize(
                objective,
                x_start,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success and not np.isnan(result.fun):
                current_r = result.x[-1]
                if current_r > best_r:
                    best_r = current_r
                    best_x = result.x
        except Exception:
            continue
            
    # Final refinement
    try:
        result = minimize(
            objective,
            best_x,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
        )
        if result.success and not np.isnan(result.fun):
            current_r = result.x[-1]
            if current_r > best_r:
                best_r = current_r
                best_x = result.x
    except Exception:
        pass
        
    centers = best_x[:2*N].reshape(N, 2)
    return centers

# EVOLVE-BLOCK-END
