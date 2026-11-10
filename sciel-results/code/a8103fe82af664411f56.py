# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    We use a Keich-like binary construction in x and y, then refine with
    coordinate descent and a bounded scalar minimizer (scipy) to reduce union volume.
    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin; we want to return well before 630s

    N = 8

    # Initial Keich construction
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

    base_x = keich[i_flat].copy()
    base_y = keich[j_flat].copy()
    positions = np.stack([base_x, base_y], axis=1)

    # Helper to compute volume
    def compute_volume(pos_arr):
        bx = pos_arr[:, 0]
        by = pos_arr[:, 1]
        X0 = np.min(bx)
        X1 = np.max(bx + (i_flat + 1) / N)
        Y0 = np.min(by)
        Y1 = np.max(by + (j_flat + 1) / N)
        if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
            return np.inf
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, 500000)
        b = rng.uniform(Y0, Y1, 500000)
        c = rng.uniform(0, 1, 500000)
        # Check if point (a,b,c) is inside any tube
        # Tube k: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
        # Vectorized check: for each tube, compute if a is in x-range and b is in y-range
        # x-range for tube k: [bx[k] + c * i_flat[k]/N, bx[k] + c * i_flat[k]/N + 1/N]
        # y-range for tube k: [by[k] + c * j_flat[k]/N, by[k] + c * j_flat[k]/N + 1/N]
        # We need to check if for any k, a is in x-range_k and b is in y-range_k
        # This is a 64 x 500000 boolean check, which is 32 million elements, feasible
        # x_left = bx[None, :] + c[:, None] * i_flat[None, :] / N
        # x_right = x_left + 1/N
        # y_left = by[None, :] + c[:, None] * j_flat[None, :] / N
        # y_right = y_left + 1/N
        # in_x = (a[:, None] >= x_left) & (a[:, None] <= x_right)
        # in_y = (b[:, None] >= y_left) & (b[:, None] <= y_right)
        # inside = np.any(in_x & in_y, axis=1)
        # V = np.mean(inside) * (X1 - X0) * (Y1 - Y0)
        
        # More efficient: loop over tubes to avoid huge memory if needed, but 64*500k is fine
        # Let's do it in chunks to be safe on memory
        inside = np.zeros(500000, dtype=bool)
        chunk_size = 100000
        for start in range(0, 500000, chunk_size):
            end = min(start + chunk_size, 500000)
            a_chunk = a[start:end]
            b_chunk = b[start:end]
            c_chunk = c[start:end]
            # x_left shape: (64, chunk)
            x_left = bx[None, :] + c_chunk[:, None] * i_flat[None, :] / N
            x_right = x_left + 1/N
            y_left = by[None, :] + c_chunk[:, None] * j_flat[None, :] / N
            y_right = y_left + 1/N
            in_x = (a_chunk[:, None] >= x_left) & (a_chunk[:, None] <= x_right)
            in_y = (b_chunk[:, None] >= y_left) & (b_chunk[:, None] <= y_right)
            inside[start:end] = np.any(in_x & in_y, axis=1)
        V = np.mean(inside) * (X1 - X0) * (Y1 - Y0)
        return V

    current_vol = compute_volume(positions)

    # Coordinate descent refinement
    # Since x and y are separable in the initial construction, we can optimize them independently
    # But the volume is a joint function. However, due to the product structure of the Keich construction,
    # the x-positions depend only on i, and y-positions only on j.
    # So we can optimize the 8 x-positions and 8 y-positions separately.
    
    # Extract unique x and y positions
    # x positions are determined by i_flat, which cycles 0..7, 0..7, etc.
    # So base_x[0:8] are the x-positions for i=0..7
    # base_y[0:8] are the y-positions for j=0..7 (since j_flat[0:8] = 0..7)
    
    x_pos = base_x[0:N].copy()
    y_pos = base_y[0:N].copy()

    def set_positions(x_pos, y_pos):
        nonlocal positions
        base_x = np.repeat(x_pos, N)
        base_y = np.tile(y_pos, N)
        positions = np.stack([base_x, base_y], axis=1)

    # Optimize x_pos
    def vol_x(x_new):
        set_positions(x_new, y_pos)
        return compute_volume(positions)

    # Optimize y_pos
    def vol_y(y_new):
        set_positions(x_pos, y_new)
        return compute_volume(positions)

    # Use scipy minimize_scalar for each coordinate
    from scipy.optimize import minimize_scalar

    # We need to define bounds. The Keich construction gives positions in [0, 1].
    # Let's use a wider range, say [-1, 2], to allow for better packing.
    bounds = (-1.0, 2.0)

    # Iterate coordinate descent
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        improved = False
        
        # Optimize each x coordinate
        for k in range(N):
            if time.time() > deadline - 30:
                break
            def func_x(t):
                x_temp = x_pos.copy()
                x_temp[k] = t
                return vol_x(x_temp)
            
            res = minimize_scalar(func_x, bounds=bounds, method='bounded', options={'xatol': 1e-6})
            if res.fun < current_vol - 1e-10:
                x_pos[k] = res.x
                current_vol = res.fun
                improved = True
        
        if time.time() > deadline - 30:
            break
        
        # Optimize each y coordinate
        for k in range(N):
            if time.time() > deadline - 30:
                break
            def func_y(t):
                y_temp = y_pos.copy()
                y_temp[k] = t
                return vol_y(y_temp)
            
            res = minimize_scalar(func_y, bounds=bounds, method='bounded', options={'xatol': 1e-6})
            if res.fun < current_vol - 1e-10:
                y_pos[k] = res.x
                current_vol = res.fun
                improved = True
        
        if not improved:
            break

    # Finalize positions
    set_positions(x_pos, y_pos)
    
    # Check validity
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
