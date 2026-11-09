import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 580.0  # safety margin before 600s limit
    
    def compute_volume_fast(X, Y, num_samples=500000):
        """
        Compute the union volume using Monte Carlo.
        X and Y are 2D arrays of shape (N, N).
        Returns the estimated volume.
        """
        N_val = len(X)
        i_grid, j_grid = np.meshgrid(np.arange(N_val), np.arange(N_val), indexing="ij")
        
        # Compute bounding box
        X1 = np.max(X + i_grid / N_val)
        Y1 = np.max(Y + j_grid / N_val)
        X0 = np.min(X)
        Y0 = np.min(Y)
        
        if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        # Vectorized check
        c_col = c[:, np.newaxis, np.newaxis]
        a_col = a[:, np.newaxis, np.newaxis]
        b_col = b[:, np.newaxis, np.newaxis]
        
        x_left = X[np.newaxis, :, :] + c_col * i_grid[np.newaxis, :, :] / N_val
        y_left = Y[np.newaxis, :, :] + c_col * j_grid[np.newaxis, :, :] / N_val
        x_right = x_left + w
        y_right = y_left + w
        
        in_tube = (a_col >= x_left) & (a_col <= x_right) & (b_col >= y_left) & (b_col <= y_right)
        in_any = np.any(in_tube, axis=(1, 2))
        fraction = np.mean(in_any)
        volume = fraction * (X1 - X0) * (Y1 - Y0)
        return volume
    
    def local_search(X, Y, max_time=400.0, initial_step=0.05):
        """
        Coordinate descent local search to minimize union volume.
        """
        best_X = X.copy()
        best_Y = Y.copy()
        best_vol = compute_volume_fast(best_X, best_Y, num_samples=50000)
        step = initial_step
        N_val = len(best_X)
        i_grid, j_grid = np.meshgrid(np.arange(N_val), np.arange(N_val), indexing="ij")
        
        while time.time() - start_time < max_time:
            improved = False
            for i in range(N_val):
                for j in range(N_val):
                    if time.time() - start_time > max_time:
                        break
                    # Try perturbing X[i,j]
                    for direction in [-1, 1]:
                        trial_X = best_X.copy()
                        trial_Y = best_Y.copy()
                        trial_X[i, j] += direction * step
                        vol = compute_volume_fast(trial_X, trial_Y, num_samples=20000)
                        if vol < best_vol:
                            best_X = trial_X
                            best_Y = trial_Y
                            best_vol = vol
                            improved = True
                    # Try perturbing Y[i,j]
                    for direction in [-1, 1]:
                        trial_X = best_X.copy()
                        trial_Y = best_Y.copy()
                        trial_Y[i, j] += direction * step
                        vol = compute_volume_fast(trial_X, trial_Y, num_samples=20000)
                        if vol < best_vol:
                            best_X = trial_X
                            best_Y = trial_Y
                            best_vol = vol
                            improved = True
                if time.time() - start_time > max_time:
                    break
            if not improved:
                step *= 0.8
                if step < 1e-5:
                    break
            else:
                step *= 0.95
        
        return best_X, best_Y, best_vol
    
    # Initialize with fan construction
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    X = -i_grid / 16.0
    Y = -j_grid / 16.0
    
    # Run local search
    X, Y, vol = local_search(X, Y, max_time=450.0, initial_step=0.05)
    
    # Final evaluation with full sample count
    final_vol = compute_volume_fast(X, Y, num_samples=500000)
    
    # Reshape to (64, 2)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)
    return positions
