# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

def compute_1d_union_length(positions):
    """
    Computes the integral from 0 to 1 of the union length of the intervals
    [x_i + c*i/8, x_i + 1/8 + c*i/8] for i=0..7.
    positions: array of shape (8,) with base x coordinates.
    """
    N = 8
    # Breakpoints are where the order of interval endpoints might change,
    # or simply we can sample densely. However, for exactness in a 1D case
    # with linear motion, the union length is piecewise linear in c?
    # Actually, the union of intervals on a line:
    # Intervals are [L_i(c), R_i(c)] where L_i = x_i + c*i/N, R_i = L_i + 1/N.
    # The union length is a piecewise linear function of c?
    # No, the merging of intervals can change.
    # For N=8, we can just use a dense grid of c to approximate the integral
    # very accurately, or use the fact that the "critical" c values are where
    # L_i(c) = R_j(c) or L_i(c) = L_j(c) or R_i(c) = R_j(c).
    # L_i = x_i + c*i/N
    # R_j = x_j + c*j/N + 1/N
    # L_i = R_j => x_i - x_j - 1/N = c*(j-i)/N
    # If i != j, c = N*(x_i - x_j - 1/N) / (j-i)
    # We collect all such c in (0,1), plus 0 and 1, sort them, and integrate
    # linearly between them.
    
    c_points = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            # L_i(c) = R_j(c)
            # x_i + c*i/N = x_j + c*j/N + 1/N
            # c*(i-j)/N = x_j - x_i + 1/N
            # c = N*(x_j - x_i + 1/N) / (i-j)
            denom = i - j
            if denom != 0:
                c_val = N * (positions[j] - positions[i] + 1.0/N) / denom
                if 0 < c_val < 1:
                    c_points.append(c_val)
            
            # Also check L_i = L_j and R_i = R_j? 
            # L_i = L_j => x_i + c*i/N = x_j + c*j/N => c = N*(x_j-x_i)/(i-j).
            # If intervals overlap, the union is determined by min L and max R.
            # The changes in union length occur when intervals start/stop overlapping.
            # This happens when L_i = R_j or L_j = R_i.
            # We covered L_i = R_j above. L_j = R_i is symmetric (swap i,j).
            # So the set of critical points is sufficient.
            
    c_points = np.sort(np.unique(c_points))
    
    # Evaluate union length at midpoints of intervals between c_points
    # and at the points themselves (though it's continuous).
    # Since it's piecewise linear, we can just average the values at the endpoints of each segment.
    
    lengths = np.zeros(len(c_points))
    for idx, c in enumerate(c_points):
        Ls = positions + c * np.arange(N) / N
        Rs = Ls + 1.0/N
        # Sort intervals by L
        order = np.argsort(Ls)
        Ls_sorted = Ls[order]
        Rs_sorted = Rs[order]
        
        current_max_r = -np.inf
        union_len = 0.0
        for k in range(N):
            if Ls_sorted[k] > current_max_r:
                union_len += Rs_sorted[k] - Ls_sorted[k]
                current_max_r = Rs_sorted[k]
            else:
                if Rs_sorted[k] > current_max_r:
                    union_len += Rs_sorted[k] - current_max_r
                    current_max_r = Rs_sorted[k]
        lengths[idx] = union_len
    
    # Integrate using trapezoidal rule on the sorted c points
    integral = np.trapz(lengths, c_points)
    return integral

def compute_2d_volume(positions):
    """
    Computes the volume of the union of the 64 tubes.
    positions: array of shape (64, 2) or flattened (128,).
    """
    if positions.ndim == 1:
        positions = positions.reshape(64, 2)
    
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    
    # Group by i
    # i = k // 8
    # For each i, we have 8 tubes with same i, different j.
    # The x-interval at height c for tube (i,j) is [x_{ij} + c*i/8, x_{ij} + c*i/8 + 1/8]
    # The y-interval is [y_{ij} + c*j/8, y_{ij} + c*j/8 + 1/8]
    
    # To compute the volume, we integrate the area of the union of squares at height c.
    # Area(c) = Union of [Lx, Rx] x [Ly, Ry].
    # This is a 2D union of rectangles.
    # We can compute this exactly for a fixed c by sweeping or using a grid?
    # Since N=8 is small, we can use a dense sampling of c to approximate the volume integral.
    # Given the time limit, a very dense Monte Carlo or Riemann sum over c is safer than
    # trying to derive the exact 2D piecewise polynomial which is complex.
    # However, the inspirations suggest an analytic or semi-analytic approach was used.
    # Let's try a dense Riemann sum over c.
    
    num_c_samples = 200
    cs = np.linspace(0, 1, num_c_samples)
    
    areas = np.zeros(num_c_samples)
    
    # Precompute i, j indices
    i_idx = np.arange(64) // 8
    j_idx = np.arange(64) % 8
    
    for idx_c, c in enumerate(cs):
        Lxs = x_pos + c * i_idx / 8.0
        Rxs = Lxs + 1.0/8.0
        Lys = y_pos + c * j_idx / 8.0
        Rys = Lys + 1.0/8.0
        
        # Compute union area of rectangles
        # Use a simple grid-based approximation for the area?
        # Or exact 2D union area algorithm.
        # Exact 2D union area:
        # Sort all x-coordinates (Lx, Rx) and y-coordinates (Ly, Ry).
        # Create a grid of cells. For each cell, check if it's covered.
        # With 64 rectangles, we have 128 unique x and 128 unique y coords.
        # Grid size 128x128 = 16384 cells.
        # For each cell, checking 64 rects is 1M ops.
        # 200 samples * 1M ops = 200M ops. This might be too slow in Python loop.
        
        # Alternative: Since the rectangles are axis aligned and we just need the union area,
        # we can use the "sweep line" or simpler:
        # The area is sum of widths of x-intervals times average y-coverage? No.
        
        # Let's optimize the inner loop.
        # We can vectorize the cell checking.
        
        unique_xs = np.unique(np.concatenate([Lxs, Rxs]))
        unique_ys = np.unique(np.concatenate([Lys, Rys]))
        
        # Create meshgrid of cell centers? Or just check coverage of cells.
        # Cell (a,b) is [unique_xs[a], unique_xs[a+1]] x [unique_ys[b], unique_ys[b+1]]
        # It is covered if there exists a rect k such that
        # Lxs[k] <= unique_xs[a] and Rxs[k] >= unique_xs[a+1] and
        # Lys[k] <= unique_ys[b] and Rys[k] >= unique_ys[b+1]
        
        # This is still O(num_cells * num_rects).
        
        # Let's try a different approach: Monte Carlo for the area at each c?
        # No, we need deterministic volume for the optimizer.
        
        # Let's use a simpler approximation:
        # The union of squares.
        # If we assume the squares are small, the area is approx sum of areas - overlaps.
        # But we need accuracy.
        
        # Let's stick to the exact grid method but vectorize it.
        
        # Grid indices
        x_grid = unique_xs[:-1]
        x_widths = np.diff(unique_xs)
        y_grid = unique_ys[:-1]
        y_heights = np.diff(unique_ys)
        
        # Cell areas
        cell_areas = np.outer(x_widths, y_heights)
        
        # Check coverage
        # For each cell (i,j), is it covered?
        # We can broadcast.
        # Lxs shape (64,), Rxs (64,)
        # x_grid shape (nx,), y_grid shape (ny,)
        
        # covered_x[i, k] = (Lxs[k] <= x_grid[i] + epsilon) & (Rxs[k] >= x_grid[i] + x_widths[i] - epsilon)
        # This is O(nx * 64).
        
        eps = 1e-12
        x_start = x_grid[None, :] # (1, nx)
        x_end = (x_grid + x_widths)[None, :] # (1, nx)
        
        y_start = y_grid[None, :] # (1, ny)
        y_end = (y_grid + y_heights)[None, :] # (1, ny)
        
        # Lxs shape (64,) -> (64, 1)
        Lxs_b = Lxs[:, None]
        Rxs_b = Rxs[:, None]
        Lys_b = Lys[:, None]
        Rys_b = Rys[:, None]
        
        # x_covered[k, i] = (Lxs[k] <= x_start[i]) & (Rxs[k] >= x_end[i])
        x_covered = (Lxs_b <= x_start) & (Rxs_b >= x_end) # (64, nx)
        # y_covered[k, j] = (Lys[k] <= y_start[j]) & (Rys[k] >= y_end[j])
        y_covered = (Lys_b <= y_start) & (Rys_b >= y_end) # (64, ny)
        
        # A cell (i, j) is covered if there exists k such that x_covered[k, i] AND y_covered[k, j]
        # This is a bit tricky to vectorize directly as "exists k".
        # We can compute for each k, the set of cells covered: outer(x_covered[k], y_covered[k])
        # Then OR them all together.
        
        # covered_cells[i, j] = OR_k (x_covered[k, i] AND y_covered[k, j])
        
        # This is equivalent to:
        # For each cell, check if the intersection of the set of rects covering x-interval i
        # and the set of rects covering y-interval j is non-empty.
        
        # Let's just loop over k? 64 iterations.
        # Inside, do outer product and OR.
        # 64 * (nx * ny) ops.
        # nx, ny <= 128. 128*128 = 16384. 64 * 16384 = 1M ops per c.
        # 200 c samples -> 200M ops. In NumPy, this is fast (~0.1s).
        
        covered = np.zeros((len(x_grid), len(y_grid)), dtype=bool)
        for k in range(64):
            if np.any(x_covered[k]) and np.any(y_covered[k]):
                covered |= np.outer(x_covered[k], y_covered[k])
        
        areas[idx_c] = np.sum(cell_areas[covered])
        
    volume = np.trapz(areas, cs)
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    """
    Returns a finite array of shape (64, 2) of base positions (x_k, y_k).
    """
    start_time = time.time()
    deadline = 600.0
    
    # Initial guess: The seed program's Keich construction
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    init_x = keich[i_flat]
    init_y = keich[j_flat]
    
    # Flatten for optimizer
    init_flat = np.concatenate([init_x, init_y])
    
    # Bounds
    # The box X0..X1, Y0..Y1 is determined by the positions.
    # To keep the box small, we should constrain positions.
    # If we don't constrain, the optimizer might spread them out to reduce overlap?
    # No, spreading out increases the box size, which increases the volume estimate
    # if the union doesn't fill the box efficiently.
    # Actually, V = fraction * BoxArea.
    # If we spread out, BoxArea increases. Fraction might decrease (less overlap).
    # We want to minimize V.
    # Usually, compacting the tubes is better.
    # Let's bound the positions to a reasonable range, e.g., [-1, 2].
    
    bounds = [(-1.0, 2.0)] * 128
    
    # Differential Evolution
    # maxiter needs to be small enough to finish in time.
    # Each function evaluation takes ~0.1s (estimate).
    # 600s / 0.1s = 6000 evaluations.
    # DE population size ~ 10 * dim = 1280.
    # Generations ~ 6000 / 1280 ~ 4.
    # This is very few.
    
    # We need to speed up compute_2d_volume.
    # The inner loop is the bottleneck.
    
    # Let's try to optimize the volume calculation.
    # Instead of 200 samples, maybe 50 is enough?
    # Or vectorize the c-loop?
    # The grid structure changes with c, so it's hard to vectorize across c.
    
    # Alternative: Use a coarser grid for the optimizer, and refine at the end?
    
    # Let's try to make the volume calc faster.
    # The `covered` calculation is the main cost.
    
    # Let's reduce num_c_samples to 50 for the optimizer.
    num_c_samples_opt = 50
    
    def fast_volume(positions):
        if positions.ndim == 1:
            positions = positions.reshape(64, 2)
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]
        
        i_idx = np.arange(64) // 8
        j_idx = np.arange(64) % 8
        
        cs = np.linspace(0, 1, num_c_samples_opt)
        areas = np.zeros(num_c_samples_opt)
        
        for idx_c, c in enumerate(cs):
            Lxs = x_pos + c * i_idx / 8.0
            Rxs = Lxs + 1.0/8.0
            Lys = y_pos + c * j_idx / 8.0
            Rys = Lys + 1.0/8.0
            
            unique_xs = np.unique(np.concatenate([Lxs, Rxs]))
            unique_ys = np.unique(np.concatenate([Lys, Rys]))
            
            x_grid = unique_xs[:-1]
            x_widths = np.diff(unique_xs)
            y_grid = unique_ys[:-1]
            y_heights = np.diff(unique_ys)
            
            cell_areas = np.outer(x_widths, y_heights)
            
            x_start = x_grid[None, :]
            x_end = (x_grid + x_widths)[None, :]
            y_start = y_grid[None, :]
            y_end = (y_grid + y_heights)[None, :]
            
            Lxs_b = Lxs[:, None]
            Rxs_b = Rxs[:, None]
            Lys_b = Lys[:, None]
            Rys_b = Rys[:, None]
            
            x_covered = (Lxs_b <= x_start) & (Rxs_b >= x_end)
            y_covered = (Lys_b <= y_start) & (Rys_b >= y_end)
            
            covered = np.zeros((len(x_grid), len(y_grid)), dtype=bool)
            # Optimization: Only check k if it covers any x and y
            for k in range(64):
                if x_covered[k].any() and y_covered[k].any():
                    covered |= np.outer(x_covered[k], y_covered[k])
            
            areas[idx_c] = np.sum(cell_areas[covered])
        
        return np.trapz(areas, cs)
    
    # Test speed
    t0 = time.time()
    v0 = fast_volume(init_flat)
    t1 = time.time()
    eval_time = t1 - t0
    sys.stderr.write(f"Eval time: {eval_time}s\n")
    
    # Adjust maxiter based on eval time
    # We want to use 90% of the time for optimization.
    time_budget = 500.0
    max_evals = int(time_budget / eval_time)
    # DE population is ~ 15 * dim? Default popsize=15.
    # Dim = 128. Pop = 1920.
    # Generations = max_evals / 1920.
    
    # If eval_time is too high, we might not be able to run DE.
    # If eval_time > 1s, max_evals < 500, which is less than 1 generation.
    # In that case, we should fall back to a simpler method or just return seed.
    
    if eval_time > 0.5:
        # Too slow for DE
        # Try to improve speed or use a different optimizer?
        # Maybe reduce num_c_samples further?
        pass
        
    # Use differential_evolution
    # bounds is a list of tuples
    try:
        result = differential_evolution(
            fast_volume,
            bounds,
            maxiter=100, # Will be cut off by timeout anyway if we add a callback
            popsize=10,
            tol=0.01,
            seed=42,
            polish=False, # Polish is expensive
            callback=None
        )
        
        # Check time
        if time.time() > deadline - 30:
            # Return best found so far? DE doesn't expose best easily in callback without storing.
            # If we are here, it finished.
            pass
            
    except Exception as e:
        sys.stderr.write(f"Optimization failed: {e}\n")
        # Fall back to seed
        return np.stack([init_x, init_y], axis=1)
    
    if result.success or result.fun < v0:
        best_flat = result.x
    else:
        best_flat = init_flat
        
    # Refine with a higher resolution volume calc?
    # If we have time left.
    remaining = deadline - 30 - time.time()
    if remaining > 5:
        # Maybe run a local optimization?
        # Or just use the result.
        pass
        
    best_positions = best_flat.reshape(64, 2)
    
    # Ensure finite
    if not np.all(np.isfinite(best_positions)):
        return np.stack([init_x, init_y], axis=1)
        
    return best_positions

# EVOLVE-BLOCK-END
