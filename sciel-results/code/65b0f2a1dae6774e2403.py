import numpy as np
import time
from scipy.optimize import minimize

def _get_breakpoints(X, N=8):
    """
    Calculate all c in (0, 1) where the order of endpoints changes.
    """
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(i + 1, N):
            denom = i - j
            # L_i = L_j
            c1 = N * (X[j] - X[i]) / denom
            if 0.0 < c1 < 1.0:
                breakpoints.add(c1)
            # L_i = R_j
            c2 = N * (X[j] + 1.0 / N - X[i]) / denom
            if 0.0 < c2 < 1.0:
                breakpoints.add(c2)
            # R_i = L_j
            c3 = N * (X[j] - X[i] - 1.0 / N) / denom
            if 0.0 < c3 < 1.0:
                breakpoints.add(c3)
            # R_i = R_j
            c4 = N * (X[j] - X[i]) / denom
            if 0.0 < c4 < 1.0:
                breakpoints.add(c4)
    return sorted(breakpoints)

def _compute_1d_cost_squared(X, N=8):
    """
    Compute integral of (Union Length)^2 over c in [0, 1].
    """
    if not np.all(np.isfinite(X)):
        return 1e10
        
    breakpoints = _get_breakpoints(X, N)
    
    total_integral = 0.0
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        if c_end <= c_start:
            continue
            
        c_mid = 0.5 * (c_start + c_end)
        
        # Endpoints at c_mid
        Ls = [X[i] + (i / N) * c_mid for i in range(N)]
        Rs = [X[i] + (i / N) * c_mid + 1.0 / N for i in range(N)]
        
        # Merge intervals
        intervals = sorted(zip(Ls, Rs))
        merged = []
        cur_l, cur_r = intervals[0]
        for l, r in intervals[1:]:
            if l > cur_r:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
            else:
                cur_r = max(cur_r, r)
        merged.append((cur_l, cur_r))
        
        # Calculate L(c_mid)
        L_mid = sum(r - l for l, r in merged)
        
        # Calculate slope using numerical differentiation
        delta_c = 1e-6
        c_plus = c_mid + delta_c
        if c_plus > c_end:
            c_plus = c_mid
            delta_c = 0
            slope = 0
        else:
            Ls_p = [X[i] + (i / N) * c_plus for i in range(N)]
            Rs_p = [X[i] + (i / N) * c_plus + 1.0 / N for i in range(N)]
            intervals_p = sorted(zip(Ls_p, Rs_p))
            merged_p = []
            cur_l, cur_r = intervals_p[0]
            for l, r in intervals_p[1:]:
                if l > cur_r:
                    merged_p.append((cur_l, cur_r))
                    cur_l, cur_r = l, r
                else:
                    cur_r = max(cur_r, r)
            merged_p.append((cur_l, cur_r))
            L_plus = sum(r - l for l, r in merged_p)
            slope = (L_plus - L_mid) / delta_c
            
        h = (c_end - c_start) / 2.0
        A = slope
        B = L_mid
        integral_segment = (2.0 * A * A * h**3) / 3.0 + 2.0 * B * B * h
        total_integral += integral_segment
        
    return total_integral

def _compute_3d_volume(positions, N=8, n_samples=50000):
    """
    Compute exact 3D union volume using Monte Carlo with fewer samples for speed.
    """
    if positions.shape != (N*N, 2):
        return 1e10
    
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    # Determine bounds
    i_vals = np.repeat(np.arange(N), N)
    j_vals = np.tile(np.arange(N), N)
    
    X0 = np.min(x_coords)
    X1 = np.max(x_coords + (i_vals + 1) / N)
    Y0 = np.min(y_coords)
    Y1 = np.max(y_coords + (j_vals + 1) / N)
    
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return 1e10
    
    # Use vectorized check with fewer samples for speed
    rng = np.random.default_rng(42)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is inside any tube
    # Tube k has i = k // N, j = k % N
    # At height c, tube k is [x_k + c*i/N, x_k + c*i/N + 1/N] x [y_k + c*j/N, y_k + c*j/N + 1/N]
    
    # Vectorize over all tubes
    # For each tube k, check if a is in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
    # and b is in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    
    # This is O(N^2 * n_samples) which is 64 * 50000 = 3.2M operations, should be fast
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(N * N):
        i_k = k // N
        j_k = k % N
        x_k = x_coords[k]
        y_k = y_coords[k]
        
        # Check if a is in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
        a_min = x_k + c * i_k / N
        a_max = a_min + 1.0 / N
        # Check if b is in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
        b_min = y_k + c * j_k / N
        b_max = b_min + 1.0 / N
        
        mask = (a >= a_min) & (a <= a_max) & (b >= b_min) & (b <= b_max)
        inside |= mask
    
    fraction = np.mean(inside)
    volume = fraction * area
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    N = 8
    
    # Initial guesses
    inits = [
        np.zeros(N),
        np.arange(N) / N,
        -np.arange(N) / N,
        np.random.default_rng(12345).uniform(-0.5, 0.5, N),
        np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]) / N, # Clustered
    ]
    
    best_X = None
    best_cost = float('inf')
    
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        try:
            # Use Nelder-Mead for small N
            res = minimize(_compute_1d_cost_squared, init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception as e:
            pass
    
    if best_X is None:
        best_X = np.zeros(N)
    
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_X[i_flat]
    base_y = best_X[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Final polish: try to improve using 3D volume as objective
    if time.time() < deadline - 60:
        try:
            def _cost_3d(X):
                i_grid2, j_grid2 = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
                i_flat2 = i_grid2.flatten()
                j_flat2 = j_grid2.flatten()
                bx = X[i_flat2]
                by = X[j_flat2]
                pos = np.stack([bx, by], axis=1)
                return _compute_3d_volume(pos, N=N, n_samples=10000)
            
            res3d = minimize(_cost_3d, best_X, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res3d.fun < _compute_3d_volume(positions, N=N, n_samples=10000):
                best_X = res3d.x.copy()
                # Reconstruct positions
                base_x = best_X[i_flat]
                base_y = best_X[j_flat]
                positions = np.stack([base_x, base_y], axis=1)
        except Exception:
            pass
    
    return positions
