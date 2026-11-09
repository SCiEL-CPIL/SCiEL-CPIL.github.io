# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product
from scipy.optimize import linprog, minimize

def _check_validity(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _optimize_grid_structure(n_cols, n_rows, n_squares=10, n_restarts=50, time_limit=5.0):
    """
    Optimize a grid structure with n_cols columns and n_rows rows.
    Place n_squares squares in the grid cells.
    """
    total_cells = n_cols * n_rows
    if total_cells < n_squares:
        return -np.inf, None, None
    
    # Variables: (n_cols-1) col boundaries + (n_rows-1) row boundaries
    n_vars = (n_cols - 1) + (n_rows - 1)
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        col_bounds = np.sort(np.random.uniform(0, 1, n_cols - 1))
        row_bounds = np.sort(np.random.uniform(0, 1, n_rows - 1))
        params_init = np.concatenate([col_bounds, row_bounds])
        
        bounds = [(0, 1)] * n_vars
        
        def objective(p):
            col_b = p[:n_cols-1]
            row_b = p[n_cols-1:]
            
            # Check sorted
            if np.any(np.diff(col_b) < 0) or np.any(np.diff(row_b) < 0):
                return 1000.0
            
            col_edges = np.concatenate([[0], col_b, [1]])
            row_edges = np.concatenate([[0], row_b, [1]])
            col_widths = np.diff(col_edges)
            row_heights = np.diff(row_edges)
            
            total_side = 0.0
            centers = []
            sides = []
            
            count = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    if count >= n_squares:
                        break
                    w = col_widths[c]
                    h = row_heights[r]
                    s = min(w, h)
                    total_side += s
                    cx = col_edges[c] + w/2
                    cy = row_edges[r] + h/2
                    centers.append([cx, cy])
                    sides.append(s)
                    count += 1
                if count >= n_squares:
                    break
            
            return -total_side
        
        res = minimize(objective, params_init, method='Nelder-Mead', bounds=bounds,
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < 0:  # Valid solution
            score = -res.fun
            if score > best_score:
                # Reconstruct centers and sides
                col_b = res.x[:n_cols-1]
                row_b = res.x[n_cols-1:]
                col_edges = np.concatenate([[0], col_b, [1]])
                row_edges = np.concatenate([[0], row_b, [1]])
                col_widths = np.diff(col_edges)
                row_heights = np.diff(row_edges)
                
                centers = []
                sides = []
                count = 0
                for r in range(n_rows):
                    for c in range(n_cols):
                        if count >= n_squares:
                            break
                        w = col_widths[c]
                        h = row_heights[r]
                        s = min(w, h)
                        cx = col_edges[c] + w/2
                        cy = row_edges[r] + h/2
                        centers.append([cx, cy])
                        sides.append(s)
                        count += 1
                    if count >= n_squares:
                        break
                
                best_score = score
                best_centers = np.array(centers)
                best_sides = np.array(sides)
    
    return best_score, best_centers, best_sides

def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 squares in unit square.
    Uses grid structure optimization with multiple grid configurations.
    """
    start_time = time.time()
    deadline = 590.0
    
    best_score = 0.0
    best_centers = None
    best_sides = None
    
    # Try different grid configurations
    grid_configs = [
        (1, 10), (2, 5), (2, 6), (3, 4), (3, 5), (4, 3), (5, 2), (5, 3), (10, 1),
        (2, 4), (2, 3), (3, 3), (4, 2), (4, 3), (6, 2), (7, 2), (8, 2), (9, 2),
    ]
    
    for n_cols, n_rows in grid_configs:
        if time.time() > deadline:
            break
        
        # Adjust n_restarts based on time remaining
        time_left = deadline - time.time()
        n_restarts = max(10, min(50, int(time_left / 10)))
        time_limit = min(5.0, time_left / 2)
        
        score, centers, sides = _optimize_grid_structure(n_cols, n_rows, 10, n_restarts, time_limit)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            best_sides = sides
    
    # Fallback
    if best_centers is None or not _check_validity(best_centers, best_sides):
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
