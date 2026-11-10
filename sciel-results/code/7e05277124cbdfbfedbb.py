# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _exact_volume_2d(base_x, base_y, N, c_samples=None):
    """
    Compute the exact volume of the union of 3D tubes by integrating the
    2D union area of rectangles at each height c.
    """
    if c_samples is None:
        c_samples = np.linspace(0, 1, 128)
        
    total_area = 0.0
    num_c = len(c_samples)
    
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    
    # Pre-compute indices for all 64 tubes
    # Tube (i, j)
    i_idx = np.repeat(i_vals, N) # 0,0,..,1,1,..
    j_idx = np.tile(j_vals, N)   # 0,1,..,7, 0,1,..,7
    
    for c in c_samples:
        x_starts = base_x[i_idx] + c * i_idx / N
        x_ends = x_starts + 1.0 / N
        y_starts = base_y[j_idx] + c * j_idx / N
        y_ends = y_starts + 1.0 / N
        
        x_coords = np.unique(np.concatenate([x_starts, x_ends]))
        y_coords = np.unique(np.concatenate([y_starts, y_ends]))
        
        if len(x_coords) < 2 or len(y_coords) < 2:
            area = 0.0
        else:
            x_centers = (x_coords[:-1] + x_coords[1:]) / 2.0
            y_centers = (y_coords[:-1] + y_coords[1:]) / 2.0
            
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)
            
            X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
            
            rx0 = x_starts
            rx1 = x_ends
            ry0 = y_starts
            ry1 = y_ends
            
            in_x = (X_grid[None, :, :] >= rx0[:, None, None]) & (X_grid[None, :, :] <= rx1[:, None, None])
            in_y = (Y_grid[None, :, :] >= ry0[:, None, None]) & (Y_grid[None, :, :] <= ry1[:, None, None])
            in_rect = in_x & in_y
            
            covered = np.any(in_rect, axis=0)
            
            cell_areas = np.outer(dy, dx)
            area = np.sum(covered * cell_areas)
            
        total_area += area
        
    return total_area / num_c


def _smooth_volume_2d(base_x, base_y, N, c_samples=None, beta=50.0):
    """
    Compute a smooth approximation of the volume using soft-max for union area.
    This allows gradient-based optimization.
    """
    if c_samples is None:
        c_samples = np.linspace(0, 1, 64)
        
    total_area = 0.0
    num_c = len(c_samples)
    
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    
    i_idx = np.repeat(i_vals, N)
    j_idx = np.tile(j_vals, N)
    
    for c in c_samples:
        x_starts = base_x[i_idx] + c * i_idx / N
        x_ends = x_starts + 1.0 / N
        y_starts = base_y[j_idx] + c * j_idx / N
        y_ends = y_starts + 1.0 / N
        
        x_coords = np.unique(np.concatenate([x_starts, x_ends]))
        y_coords = np.unique(np.concatenate([y_starts, y_ends]))
        
        if len(x_coords) < 2 or len(y_coords) < 2:
            area = 0.0
        else:
            x_centers = (x_coords[:-1] + x_coords[1:]) / 2.0
            y_centers = (y_coords[:-1] + y_coords[1:]) / 2.0
            
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)
            
            X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
            
            rx0 = x_starts
            rx1 = x_ends
            ry0 = y_starts
            ry1 = y_ends
            
            in_x = (X_grid[None, :, :] >= rx0[:, None, None]) & (X_grid[None, :, :] <= rx1[:, None, None])
            in_y = (Y_grid[None, :, :] >= ry0[:, None, None]) & (Y_grid[None, :, :] <= ry1[:, None, None])
            in_rect = in_x & in_y
            
            # Soft-max approximation of union: sum(exp(beta * in_rect)) / exp(beta * max)
            # For binary indicators, this approximates the max (union)
            rect_indicators = in_rect.astype(np.float64)
            max_val = np.max(rect_indicators, axis=0)
            
            # Where max_val == 0, all are 0, so union is 0
            # Where max_val == 1, use soft-max
            soft_union = np.zeros_like(max_val)
            mask = max_val > 0
            if np.any(mask):
                # For cells covered by at least one rectangle
                vals = rect_indicators[:, mask]
                # Soft-max: exp(beta * vals) / sum(exp(beta * vals)) * max
                # But since vals are 0 or 1, this is exp(beta) / (num_covered * exp(beta) + (total - num_covered) * 1)
                # Simplified: use 1 - exp(-beta * num_covered) as approximation
                num_covered = np.sum(vals > 0.5, axis=0)
                soft_union[mask] = 1.0 - np.exp(-beta * num_covered)
            
            cell_areas = np.outer(dy, dx)
            area = np.sum(soft_union * cell_areas)
            
        total_area += area
        
    return total_area / num_c


def _cost_3d(base_x, base_y, N):
    """Full 3D cost: Exact union volume."""
    return _exact_volume_2d(base_x, base_y, N)


def _cost_3d_smooth(base_x, base_y, N, beta=50.0):
    """Smooth 3D cost for gradient-based optimization."""
    return _smooth_volume_2d(base_x, base_y, N, beta=beta)


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses simulated annealing with smooth surrogate for polishing.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_x = keich.copy()
    best_y = keich.copy()
    best_cost = _cost_3d(best_x, best_y, N)
    
    # Simulated annealing with adaptive step size
    T = 0.02
    T_min = 1e-7
    cooling = 0.997
    step_size = 0.03
    max_steps = 80000
    
    current_x = best_x.copy()
    current_y = best_y.copy()
    current_cost = best_cost
    
    rng = np.random.default_rng(42)
    no_improve_counter = 0
    
    for step in range(max_steps):
        if time.time() > deadline - 120.0:
            break
        
        # Adaptive step size: decrease if no improvement
        if no_improve_counter > 500:
            step_size *= 0.95
            no_improve_counter = 0
        
        # Choose which dimension to perturb
        if rng.random() < 0.5:
            # Perturb x
            idx = rng.integers(N)
            delta = rng.normal(0, step_size)
            new_x = current_x.copy()
            new_x[idx] += delta
            new_cost = _cost_3d(new_x, current_y, N)
            if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / T):
                current_x = new_x
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = current_x.copy()
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
        else:
            # Perturb y
            idx = rng.integers(N)
            delta = rng.normal(0, step_size)
            new_y = current_y.copy()
            new_y[idx] += delta
            new_cost = _cost_3d(current_x, new_y, N)
            if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / T):
                current_y = new_y
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_y = current_y.copy()
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
        
        T = max(T_min, T * cooling)
    
    # Polish with scipy using smooth surrogate if time remains
    if time.time() < deadline - 90.0:
        def objective_smooth(params):
            bx = params[:N]
            by = params[N:]
            return _cost_3d_smooth(bx, by, N, beta=50.0)
        
        initial_params = np.concatenate([best_x, best_y])
        
        # Multiple restarts with smooth objective
        best_smooth_cost = best_cost
        best_smooth_x = best_x.copy()
        best_smooth_y = best_y.copy()
        
        for restart in range(3):
            try:
                if restart == 0:
                    init = initial_params.copy()
                else:
                    # Add small perturbation for diversity
                    init = initial_params + rng.normal(0, 0.01, 2*N)
                
                res = minimize(objective_smooth, init, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < best_smooth_cost:
                    # Verify with exact cost
                    exact_cost = _cost_3d(res.x[:N], res.x[N:], N)
                    if exact_cost < best_cost:
                        best_cost = exact_cost
                        best_x = res.x[:N].copy()
                        best_y = res.x[N:].copy()
                        best_smooth_cost = exact_cost
                        best_smooth_x = res.x[:N].copy()
                        best_smooth_y = res.x[N:].copy()
            except Exception:
                pass
        
        # Final high-precision polish on the best candidate
        if time.time() < deadline - 30.0:
            try:
                def objective_exact(params):
                    bx = params[:N]
                    by = params[N:]
                    return _cost_3d(bx, by, N)
                
                init_final = np.concatenate([best_x, best_y])
                res_final = minimize(objective_exact, init_final, method='L-BFGS-B',
                                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res_final.fun < best_cost:
                    best_cost = res_final.fun
                    best_x = res_final.x[:N].copy()
                    best_y = res_final.x[N:].copy()
            except Exception:
                pass
    
    # Build positions array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
