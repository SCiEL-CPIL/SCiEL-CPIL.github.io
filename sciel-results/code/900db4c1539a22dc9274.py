import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 630.0  # wall-clock limit; keep a safety margin
    
    # Precompute grids
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()  # shape (64,)
    j_flat = j_grid.flatten()  # shape (64,)
    i_frac = i_flat / N  # (64,)
    j_frac = j_flat / N  # (64,)
    
    # Pre-generate random samples for deterministic objective
    # Using 100,000 samples for balance of accuracy and speed
    # The evaluator uses 500,000, but for optimization, 100k is a good proxy
    # We use a fixed seed to ensure the objective is smooth/deterministic
    rng_opt = np.random.default_rng(42)
    # We need to map uniform [0,1] to [X0, X1] and [Y0, Y1] inside the objective
    # So we just store the [0,1] uniforms and scale them
    M = 100000
    u_a = rng_opt.uniform(0, 1, M)
    u_b = rng_opt.uniform(0, 1, M)
    u_c = rng_opt.uniform(0, 1, M)
    
    def compute_volume_opt(xk, yk):
        """Deterministic volume estimator using pre-generated samples."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_frac)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_frac)
        
        dx = X1 - X0
        dy = Y1 - Y0
        area = dx * dy
        if area <= 1e-9:
            return float('inf')
        
        # Map u_a, u_b to [X0, X1], [Y0, Y1]
        a = X0 + u_a * dx
        b = Y0 + u_b * dy
        c = u_c
        
        c_col = c[:, np.newaxis]  # (M, 1)
        x_left = xk[np.newaxis, :] + c_col * i_frac[np.newaxis, :]  # (M, 64)
        y_left = yk[np.newaxis, :] + c_col * j_frac[np.newaxis, :]  # (M, 64)
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume

    def compute_volume_full(xk, yk, num_samples=500000):
        """Full precision volume estimator for final evaluation."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_frac)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_frac)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        c_col = c[:, np.newaxis]
        x_left = xk[np.newaxis, :] + c_col * i_frac[np.newaxis, :]
        y_left = yk[np.newaxis, :] + c_col * j_frac[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume

    # Initial configuration: symmetric "fan" construction
    # x_k depends only on i, y_k depends only on j
    # We want tubes to converge. 
    # At c=1, x_pos = x_k + i/8. 
    # If we set x_k = -i * alpha, then x_pos = i(1/8 - alpha).
    # To make them converge to 0, alpha = 1/8.
    # Then at c=0, x_pos = -i/8.
    # This is the "fan" that closes.
    alpha = 1.0 / N
    xk = -i_flat * alpha
    yk = -j_flat * alpha
    
    # Flatten parameters
    params0 = np.concatenate([xk, yk])
    
    def objective(params):
        if time.time() - start_time > deadline - 10:
            return 1e10
        xk_curr = params[:64]
        yk_curr = params[64:]
        return compute_volume_opt(xk_curr, yk_curr)
    
    # Use Nelder-Mead for non-smooth optimization
    # We allow many iterations
    # maxiter is a soft limit, we also check time
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'disp': False
    }
    
    try:
        result = scipy.optimize.minimize(
            objective, 
            params0, 
            method='Nelder-Mead', 
            options=options
        )
        best_params = result.x
    except Exception:
        best_params = params0

    # Unpack
    best_xk = best_params[:64]
    best_yk = best_params[64:]
    
    # Final evaluation with full samples
    # Note: The evaluator uses specific samples. We return the positions.
    # The evaluator will compute V.
    
    positions = np.stack([best_xk, best_yk], axis=1)
    return positions
