import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 630.0  # safety margin before 630s kill
    
    # With separable structure: x_k depends only on i, y_k depends only on j
    # So we have 16 parameters: X[0..7] and Y[0..7]
    # At height c, x-union = union of [X[i] + c*i/N, X[i] + c*i/N + w] for i=0..7
    # y-union = union of [Y[j] + c*j/N, Y[j] + c*j/N + w] for j=0..7
    # Volume = integral_0^1 (x_union_len(c) * y_union_len(c)) dc
    
    def compute_union_length(positions, c_vals, w_val):
        """Compute union length of intervals [pos[i] + c*i/N, pos[i] + c*i/N + w] for i=0..N-1"""
        # positions: array of shape (N,)
        # c_vals: array of shape (M,)
        # Returns: array of shape (M,) with union lengths
        n = len(positions)
        m = len(c_vals)
        # Left endpoints: shape (m, n)
        left = positions[np.newaxis, :] + c_vals[:, np.newaxis] * np.arange(n) / n
        right = left + w_val
        # Sort by left endpoint for each c
        # Use a simple approach: for each c, find union of intervals
        # Vectorized: for each c, sort intervals and merge
        # For speed, we'll do it in a loop over c if M is small, or use a trick
        # Actually, let's just compute it directly for the MC evaluation
        
        # Alternative: for each sample point in [min_left, max_right], check if covered
        # But that's what the full MC does. Let's use a 1D approach.
        
        # For each c, the union length can be computed by:
        # 1. Get all 2n endpoints
        # 2. Sort them
        # 3. Sweep to find total covered length
        # This is O(n log n) per c value
        
        # For the fast optimizer, we'll use MC in 1D
        # But actually, let's just do the full 3D MC as before but with the separable structure
        pass
    
    def compute_volume_fast(X, Y, num_samples=50000):
        """Fast volume estimator using the separable structure."""
        # X: shape (8,), Y: shape (8,)
        # Bounding box
        i_arr = np.arange(N) / N
        X0 = np.min(X)
        X1 = np.max(X + i_arr)
        Y0 = np.min(Y)
        Y1 = np.max(Y + i_arr)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples).astype(np.float32)
        b = rng.uniform(Y0, Y1, num_samples).astype(np.float32)
        c = rng.uniform(0, 1, num_samples).astype(np.float32)
        
        # Check if point (a, b, c) is in any tube
        # Tube (i,j): a in [X[i] + c*i/N, X[i] + c*i/N + w] AND b in [Y[j] + c*j/N, Y[j] + c*j/N + w]
        # Since we have all (i,j) combinations, the point is in the union iff:
        # a is in x-union at c AND b is in y-union at c
        
        # Compute x-union membership: for each sample, is a in any [X[i] + c*i/N, X[i] + c*i/N + w]?
        # x_left[i] = X[i] + c[i_sample]*i/N
        # Vectorized: shape (num_samples, N)
        c_col = c[:, np.newaxis]
        x_left = X[np.newaxis, :] + c_col * (np.arange(N) / N)[np.newaxis, :]
        y_left = Y[np.newaxis, :] + c_col * (np.arange(N) / N)[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        
        in_x_union = np.any(in_x, axis=1)
        in_y_union = np.any(in_y, axis=1)
        
        in_any = in_x_union & in_y_union
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    def compute_volume_full(X, Y, num_samples=500000):
        """Full precision volume estimator."""
        i_arr = np.arange(N) / N
        X0 = np.min(X)
        X1 = np.max(X + i_arr)
        Y0 = np.min(Y)
        Y1 = np.max(Y + i_arr)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        c_col = c[:, np.newaxis]
        x_left = X[np.newaxis, :] + c_col * i_arr[np.newaxis, :]
        y_left = Y[np.newaxis, :] + c_col * i_arr[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        
        in_x_union = np.any(in_x, axis=1)
        in_y_union = np.any(in_y, axis=1)
        
        in_any = in_x_union & in_y_union
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    # Initial configuration: X[i] = -i * alpha, Y[j] = -j * alpha
    # This makes tubes converge as c increases
    def initial_config(alpha=0.0625):
        X = -np.arange(N) * alpha
        Y = -np.arange(N) * alpha
        return X, Y
    
    X, Y = initial_config(0.0625)
    best_X, best_Y = X.copy(), Y.copy()
    best_vol = compute_volume_fast(X, Y, 50000)
    
    # Simulated Annealing on 16 parameters
    T_start = 0.5
    T_end = 0.001
    alpha_cool = 0.999
    
    T = T_start
    rng_sa = np.random.default_rng(42)
    
    # Each evaluation takes ~2-5ms with 50k samples
    # 600s / 0.005s ~ 120,000 iterations possible
    while time.time() - start_time < deadline - 15:
        # Perturb a random subset of the 16 parameters
        num_perturb = rng_sa.integers(1, 9)
        indices = rng_sa.choice(16, size=num_perturb, replace=False)
        
        # Step size scaled by temperature
        step_size = 0.3 * T / T_start + 0.01
        dx = rng_sa.uniform(-step_size, step_size, size=num_perturb)
        
        X_new = X.copy()
        Y_new = Y.copy()
        
        # Split indices: 0-7 for X, 8-15 for Y
        x_indices = [idx for idx in indices if idx < 8]
        y_indices = [idx for idx in indices if idx >= 8]
        
        if len(x_indices) > 0:
            X_new[x_indices] += dx[:len(x_indices)]
        if len(y_indices) > 0:
            Y_new[y_indices] += dx[len(x_indices):]
        
        vol_new = compute_volume_fast(X_new, Y_new, 50000)
        
        # Metropolis criterion
        if vol_new < best_vol:
            X, Y = X_new, Y_new
            best_X, best_Y = X.copy(), Y.copy()
            best_vol = vol_new
        else:
            delta_e = (vol_new - best_vol) * 50.0
            if delta_e < 0:
                accept = True
            else:
                p = np.exp(-delta_e / T)
                accept = rng_sa.random() < p
            
            if accept:
                X, Y = X_new, Y_new
        
        T *= alpha_cool
        if T < T_end:
            T = T_end
    
    # Final evaluation with full samples
    final_vol = compute_volume_full(best_X, best_Y, 500000)
    
    # Convert back to (64, 2) format
    # Row k is tube (i, j) with i = k // 8 and j = k % 8
    # x_k = X[i], y_k = Y[j]
    positions = np.zeros((64, 2))
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = best_X[i]
        positions[k, 1] = best_Y[j]
    
    return positions

