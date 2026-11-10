# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].
    
    The union volume V factors as V = V_x * V_y, where V_x and V_y are the 
    integrated 1D union lengths in the x and y projections respectively.
    We minimize V_x and V_y independently using Nelder-Mead optimization.
    """
    N = 8
    deadline = time.time() + 600  # 600s wall-clock limit
    
    # 1D Union Length Calculation
    def calc_1d_union_length(base_pos, c, N):
        """
        Calculate the length of the union of N intervals at height c.
        base_pos: array of N base positions
        c: height scalar
        N: grid size
        """
        # Intervals are [base_pos[i] + c*i/N, base_pos[i] + c*i/N + 1/N]
        starts = base_pos + c * np.arange(N) / N
        ends = starts + 1.0 / N
        
        # Sort by start point
        order = np.argsort(starts)
        starts_sorted = starts[order]
        ends_sorted = ends[order]
        
        # Merge intervals
        total_len = 0.0
        current_end = ends_sorted[0]
        for i in range(1, N):
            if starts_sorted[i] <= current_end:
                current_end = max(current_end, ends_sorted[i])
            else:
                total_len += current_end - starts_sorted[i-1]
                current_end = ends_sorted[i]
        total_len += current_end - starts_sorted[N-1]
        return total_len

    # Objective Function for 1D Optimization
    # We integrate over c in [0, 1] using quadrature
    # To speed up, we use a fixed set of quadrature points
    c_quad = np.linspace(0, 1, 200)
    weights = np.ones_like(c_quad) / len(c_quad) # Simple Riemann sum
    
    def objective_1d(base_pos):
        """
        Calculate the integral of the 1D union length over c in [0, 1].
        """
        total_len = 0.0
        # Vectorized calculation for speed
        # starts shape: (200, 8)
        starts = base_pos[None, :] + c_quad[:, None] * np.arange(N)[None, :] / N
        ends = starts + 1.0 / N
        
        # Sort along axis 1
        order = np.argsort(starts, axis=1)
        starts_sorted = np.take_along_axis(starts, order, axis=1)
        ends_sorted = np.take_along_axis(ends, order, axis=1)
        
        # Merge logic vectorized
        # current_end starts as ends_sorted[:, 0]
        current_end = ends_sorted[:, 0].copy()
        total_len_arr = np.zeros(len(c_quad))
        
        for i in range(1, N):
            s_i = starts_sorted[:, i]
            e_i = ends_sorted[:, i]
            
            # If start <= current_end, extend current_end
            mask = s_i <= current_end
            current_end = np.where(mask, np.maximum(current_end, e_i), current_end)
            
            # If start > current_end, add the previous segment length
            # The previous segment is from starts_sorted[:, i-1] to current_end (before update)
            # Wait, the logic above updates current_end in place. 
            # Let's re-do the merge logic carefully.
            
        # Re-implementing merge logic correctly for vectorization
        total_len_arr = np.zeros(len(c_quad))
        current_end = ends_sorted[:, 0].copy()
        
        for i in range(1, N):
            s_i = starts_sorted[:, i]
            e_i = ends_sorted[:, i]
            
            # Check if interval i overlaps with the current merged interval
            overlaps = s_i <= current_end
            
            # If it doesn't overlap, add the length of the current merged interval
            # and start a new merged interval
            # If it does overlap, extend the current merged interval
            
            # Length of current merged interval is current_end - starts_sorted[:, i-1]
            # But this is only true if the previous interval was part of the same merged component.
            # Actually, the standard merge algorithm accumulates length when a gap is found.
            
            # Let's stick to a simpler, slightly slower but correct loop if vectorization is tricky, 
            # or use a robust vectorized approach.
            # Given N=8 and 200 points, a loop over N inside the objective is fine.
            
            pass
        
        # Let's just use the loop version for clarity and correctness, it's fast enough for N=8.
        total_len = 0.0
        for k in range(len(c_quad)):
            starts_k = base_pos + c_quad[k] * np.arange(N) / N
            ends_k = starts_k + 1.0 / N
            order_k = np.argsort(starts_k)
            starts_sorted_k = starts_k[order_k]
            ends_sorted_k = ends_k[order_k]
            
            cur_end = ends_sorted_k[0]
            len_k = 0.0
            for i in range(1, N):
                if starts_sorted_k[i] <= cur_end:
                    cur_end = max(cur_end, ends_sorted_k[i])
                else:
                    len_k += cur_end - starts_sorted_k[i-1]
                    cur_end = ends_sorted_k[i]
            len_k += cur_end - starts_sorted_k[N-1]
            total_len += len_k * weights[k]
            
        return total_len

    # Initial Guess: Keich Construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x0 = keich.copy()
    y0 = keich.copy()
    
    # Optimization Settings
    options = {
        'maxiter': 200,
        'maxfev': 500,
        'xatol': 1e-8,
        'fatol': 1e-10
    }
    
    best_x = x0
    best_y = y0
    best_fx = objective_1d(x0)
    best_fy = objective_1d(y0)
    
    # Optimize X
    if time.time() < deadline - 10:
        res_x = minimize(objective_1d, x0, method='Nelder-Mead', options=options)
        if res_x.success and res_x.fun < best_fx:
            best_x = res_x.x
            best_fx = res_x.fun
            
    # Optimize Y
    if time.time() < deadline - 10:
        res_y = minimize(objective_1d, y0, method='Nelder-Mead', options=options)
        if res_y.success and res_y.fun < best_fy:
            best_y = res_y.x
            best_fy = res_y.fun
            
    # Construct Positions
    # Row k is tube (i, j) with i = k // 8 and j = k % 8
    # base_x depends on i, base_y depends on j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        return np.stack([keich[i_flat], keich[j_flat]], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
