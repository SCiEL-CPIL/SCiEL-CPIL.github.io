# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of the tubes to minimize the union volume.
    Uses the property that the union of the grid of rectangles is the Cartesian
    product of the union of x-intervals and y-intervals.
    """
    N = 8
    # We optimize x_0 ... x_{N-1}. By symmetry, we set y_j = x_j.
    # Initial guess: linear star
    x0 = -np.arange(N) / (2.0 * N)
    
    def union_width(x, c):
        # x: array of shape (N,)
        # c: scalar
        starts = x + c * np.arange(N) / N
        ends = starts + 1.0 / N
        
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Merge intervals
        # Initialize with first interval
        current_end = e_sorted[0]
        total_width = 0.0
        
        for k in range(1, N):
            if s_sorted[k] <= current_end:
                # Overlapping
                if e_sorted[k] > current_end:
                    current_end = e_sorted[k]
            else:
                # Disjoint
                total_width += current_end - s_sorted[k-1] # Wait, this logic is slightly off for the first one
                # Let's do it more robustly
                pass
        
        # Robust merge
        if N == 0: return 0.0
        merged_s = s_sorted[0]
        merged_e = e_sorted[0]
        width = 0.0
        for k in range(1, N):
            if s_sorted[k] <= merged_e:
                if e_sorted[k] > merged_e:
                    merged_e = e_sorted[k]
            else:
                width += merged_e - merged_s
                merged_s = s_sorted[k]
                merged_e = e_sorted[k]
        width += merged_e - merged_s
        return width

    def cost(x):
        # Approximate integral of W(c)^2 from 0 to 1
        # Use 20 points for speed
        c_vals = np.linspace(0, 1, 20)
        total = 0.0
        for c in c_vals:
            w = union_width(x, c)
            total += w * w
        return total

    # Optimize
    # Nelder-Mead is robust for small dimensions
    res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    x_opt = res.x
    # Symmetry: y = x
    y_opt = x_opt.copy()
    
    # Construct the (64, 2) array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions

# EVOLVE-BLOCK-END
