import numpy as np
import time
import sys

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 600.0
    
    # Precompute grids
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float32)
    j_flat = j_grid.flatten().astype(np.float32)
    
    # Pre-compute shear factors
    shear_x = (i_flat / N).astype(np.float32)
    shear_y = (j_flat / N).astype(np.float32)

    def compute_volume_fast(xk, yk, num_samples=20000):
        """Fast volume estimator for optimization. Uses float32 for speed."""
        xk_f = xk.astype(np.float32)
        yk_f = yk.astype(np.float32)
        
        # Compute bounding box
        X0 = np.min(xk_f)
        X1 = np.max(xk_f + shear_x)
        Y0 = np.min(yk_f)
        Y1 = np.max(yk_f + shear_y)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples).astype(np.float32)
        b = rng.uniform(Y0, Y1, num_samples).astype(np.float32)
        c = rng.uniform(0, 1, num_samples).astype(np.float32)
        
        # Vectorized check
        c_col = c[:, np.newaxis]
        x_left = xk_f[np.newaxis, :] + c_col * shear_x[np.newaxis, :]
        y_left = yk_f[np.newaxis, :] + c_col * shear_y[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume

    def compute_volume_full(xk, yk, num_samples=500000):
        """Full precision volume estimator for final evaluation."""
        xk_f = xk.astype(np.float32)
        yk_f = yk.astype(np.float32)
        
        X0 = np.min(xk_f)
        X1 = np.max(xk_f + shear_x)
        Y0 = np.min(yk_f)
        Y1 = np.max(yk_f + shear_y)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples).astype(np.float32)
        b = rng.uniform(Y0, Y1, num_samples).astype(np.float32)
        c = rng.uniform(0, 1, num_samples).astype(np.float32)
        
        c_col = c[:, np.newaxis]
        x_left = xk_f[np.newaxis, :] + c_col * shear_x[np.newaxis, :]
        y_left = yk_f[np.newaxis, :] + c_col * shear_y[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume

    # Initial configuration: symmetric "fan" construction
    # x_k depends only on i, y_k depends only on j
    # Try to make tubes converge as c increases
    def initial_config(alpha=0.0625):
        xk = (-i_flat * alpha).astype(np.float64)
        yk = (-j_flat * alpha).astype(np.float64)
        return xk, yk

    xk, yk = initial_config(0.0625)
    best_xk, best_yk = xk.copy(), yk.copy()
    best_vol = compute_volume_fast(xk, yk, 20000)
    
    # Simulated Annealing
    # Parameters
    T_start = 0.1
    T_end = 0.001
    alpha_cool = 0.995
    
    # We need to run many iterations. 
    # Each iteration involves one volume eval (~2-5ms with 20k samples).
    # 600s / 0.005s ~ 120,000 iterations possible.
    # We will loop until time limit.
    
    T = T_start
    rng_sa = np.random.default_rng(42)
    
    while time.time() - start_time < deadline - 10:
        # Generate a random perturbation
        # Step size should be comparable to the scale of the problem (0.1-0.5)
        # We perturb a random subset of tubes to avoid huge jumps
        num_perturb = rng_sa.integers(1, 8)
        indices = rng_sa.choice(64, size=num_perturb, replace=False)
        
        # Random walk step
        step_size = 0.05 * T / T_start + 0.005
        dx = rng_sa.uniform(-step_size, step_size, size=num_perturb)
        dy = rng_sa.uniform(-step_size, step_size, size=num_perturb)
        
        xk_new = xk.copy()
        yk_new = yk.copy()
        xk_new[indices] += dx
        yk_new[indices] += dy
        
        vol_new = compute_volume_fast(xk_new, yk_new, 20000)
        
        # Metropolis criterion
        if vol_new < best_vol:
            xk, yk = xk_new, yk_new
            best_xk, best_yk = xk.copy(), yk.copy()
            best_vol = vol_new
        else:
            # Accept with probability exp(-(vol_new - best_vol) / T)
            # vol is small, so difference is small. 
            # Scale difference to be in range [0, 1] roughly?
            # Actually, standard SA uses energy difference. 
            # Let's use a scaled difference to make T meaningful.
            # If V ~ 0.5, diff ~ 0.001. 
            # exp(-0.001 / 0.1) ~ 0.99. Too permissive.
            # exp(-0.001 / 0.001) ~ 0.37.
            # Let's scale the energy difference by 100 to be sensitive.
            delta_e = (vol_new - best_vol) * 100.0
            if delta_e < 0:
                accept = True
            else:
                p = np.exp(-delta_e / T)
                accept = rng_sa.random() < p
            
            if accept:
                xk, yk = xk_new, yk_new
                # Do not update best_xk if we accepted a worse solution
                # We keep tracking the global best separately
                
        T *= alpha_cool
        if T < T_end:
            T = T_end

    # Final evaluation with full samples
    # Ensure we return the configuration that yields the best volume
    # The 'best_xk' might be stale if we accepted worse moves but didn't update it.
    # Wait, in the logic above:
    # if vol_new < best_vol: update best_xk
    # else: if accept: update xk, yk (current state)
    # So best_xk holds the all-time minimum.
    # However, the current state (xk, yk) might be different.
    # We should evaluate the best_xk.
    
    final_vol = compute_volume_full(best_xk, best_yk, 500000)
    
    # Reshape to (64, 2)
    positions = np.stack([best_xk, best_yk], axis=1)
    return positions
