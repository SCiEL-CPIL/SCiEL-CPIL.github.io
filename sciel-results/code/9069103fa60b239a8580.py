import numpy as np
from scipy.optimize import minimize
import time

def _union_area_at_c(x, y, c):
    N = 8
    # Precompute rectangle coordinates
    # x0_i = x[i] + c*i/N
    # y0_j = y[j] + c*j/N
    # rect (i,j) is [x0_i, x0_i+1/N] x [y0_j, y0_j+1/N]
    
    x0s = x + c * np.arange(N) / N
    y0s = y + c * np.arange(N) / N
    
    # Collect unique x and y coords
    xs = np.unique(np.concatenate([x0s, x0s + 1.0/N]))
    ys = np.unique(np.concatenate([y0s, y0s + 1.0/N]))
    
    # Grid of cells
    # For each cell, check if covered by any rect
    # We can use a 2D boolean array
    # But N is small, so a simple loop is fine.
    
    area = 0.0
    for k in range(len(xs) - 1):
        cx0 = xs[k]
        cx1 = xs[k+1]
        for l in range(len(ys) - 1):
            cy0 = ys[l]
            cy1 = ys[l+1]
            # Check coverage
            # A rect (i,j) covers the cell if:
            # x0_i <= cx0 and cx1 <= x0_i + 1/N
            # y0_j <= cy0 and cy1 <= y0_j + 1/N
            # This is equivalent to:
            # cx0 in [x0_i, x0_i+1/N] and cy0 in [y0_j, y0_j+1/N]
            # Since the cell is small, we can just check the center or corners.
            # Actually, the cell is fully inside a rect if the rect contains the cell.
            # We can check if there exists i,j such that:
            # x0_i <= cx0 and x0_i + 1/N >= cx1
            # y0_j <= cy0 and y0_j + 1/N >= cy1
            # We can vectorize this check.
            
            # For a given cell, find all i such that x0_i <= cx0 and x0_i + 1/N >= cx1
            # i must satisfy: cx1 - 1/N <= x0_i <= cx0
            # Similarly for j.
            
            # Let's just use a simple loop for clarity, N=8 is small.
            covered = False
            for i in range(N):
                if x0s[i] <= cx0 and x0s[i] + 1.0/N >= cx1:
                    for j in range(N):
                        if y0s[j] <= cy0 and y0s[j] + 1.0/N >= cy1:
                            covered = True
                            break
                if covered: break
            if covered:
                area += (cx1 - cx0) * (cy1 - cy0)
    return area

def _cost_func(params):
    x = params[:8]
    y = params[8:]
    c_samples = np.linspace(0, 1, 30)
    total_area = 0.0
    for c in c_samples:
        total_area += _union_area_at_c(x, y, c)
    return total_area / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    N = 8
    
    # Initial guess
    np.random.seed(42)
    x0 = np.random.uniform(0, 1, N)
    y0 = np.random.uniform(0, 1, N)
    params0 = np.concatenate([x0, y0])
    
    best_params = params0.copy()
    best_cost = _cost_func(params0)
    
    # Optimize
    try:
        res = minimize(_cost_func, params0, method='Powell', options={'maxiter': 2000, 'xtol': 1e-5, 'ftol': 1e-5})
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > 550:
        pass
        
    x_opt = best_params[:8]
    y_opt = best_params[8:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions
