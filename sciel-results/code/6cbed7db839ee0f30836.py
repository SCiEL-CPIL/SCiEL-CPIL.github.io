# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _union_length_at_heights(x_vals, N, c_vals):
    """
    Compute the union length of intervals [x_i + c*i/N, x_i + c*i/N + 1/N]
    for each c in c_vals.
    """
    lengths = np.zeros_like(c_vals)
    for idx, c in enumerate(c_vals):
        lefts = x_vals + c * np.arange(N) / N
        rights = lefts + 1.0 / N
        # Sort intervals by left endpoint
        order = np.argsort(lefts)
        lefts = lefts[order]
        rights = rights[order]
        
        # Merge intervals
        total_len = 0.0
        cur_l = lefts[0]
        cur_r = rights[0]
        for k in range(1, N):
            if lefts[k] > cur_r:
                total_len += cur_r - cur_l
                cur_l = lefts[k]
                cur_r = rights[k]
            else:
                cur_r = max(cur_r, rights[k])
        total_len += cur_r - cur_l
        lengths[idx] = total_len
    return lengths


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Use a fixed grid for c to estimate the integral
    # 2000 points is fast and accurate
    c_grid = np.linspace(0.0, 1.0, 2001)
    
    def objective(params):
        x_vals = params[:N]
        y_vals = params[N:]
        
        Lx = _union_length_at_heights(x_vals, N, c_grid)
        Ly = _union_length_at_heights(y_vals, N, c_grid)
        
        # Integrate the product
        V = np.trapz(Lx * Ly, c_grid)
        return V

    # Bounds: x and y positions should be in [0, 1] to keep the box small
    # The actual box is determined by min/max, but constraining to [0,1] 
    # encourages compactness.
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    # Run differential evolution
    # seed for reproducibility
    result = differential_evolution(
        objective, 
        bounds, 
        seed=12345, 
        maxiter=1000, 
        tol=1e-8, 
        polish=False, # Nelder-Mead polish can be slow, DE is usually good enough
        init='latinhypercube',
        popsize=15
    )
    
    x_vals = result.x[:N]
    y_vals = result.x[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions = np.stack([x_vals[i_flat], y_vals[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
