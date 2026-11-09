import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    start_time = time.time()
    deadline = 590.0  # safety margin before 630s limit

    def compute_volume(X, Y, num_samples=500000, seed=12345):
        """
        Compute the union volume using Monte Carlo.
        X and Y are 2D arrays of shape (N, N).
        Returns the estimated volume.
        """
        N_val = N
        i_grid, j_grid = np.meshgrid(np.arange(N_val), np.arange(N_val), indexing="ij")
        
        # Compute bounding box
        X1 = np.max(X + i_grid / N_val)
        Y1 = np.max(Y + j_grid / N_val)
        X0 = np.min(X)
        Y0 = np.min(Y)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(seed)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        # Process in batches to avoid huge memory usage
        batch_size = 50000
        in_any_count = 0
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            s = batch_idx * batch_size
            e = min(s + batch_size, num_samples)
            a_b = a[s:e]
            b_b = b[s:e]
            c_b = c[s:e]
            
            # For each tube, check if point is inside
            # Tube (i,j): x in [X[i,j] + c*i/N, X[i,j] + c*i/N + w]
            #              y in [Y[i,j] + c*j/N, Y[i,j] + c*j/N + w]
            in_any_batch = np.zeros(len(a_b), dtype=bool)
            
            for i in range(N_val):
                for j in range(N_val):
                    x_left = X[i, j] + c_b * i / N_val
                    y_left = Y[i, j] + c_b * j / N_val
                    in_tube = (a_b >= x_left) & (a_b <= x_left + w) & \
                               (b_b >= y_left) & (b_b <= y_left + w)
                    in_any_batch |= in_tube
            
            in_any_count += np.sum(in_any_batch)
        
        fraction = in_any_count / num_samples
        volume = fraction * box_area
        return volume

    def compute_volume_fast(X, Y, num_samples=500000, seed=12345):
        """Faster volume computation using vectorized tube checks."""
        N_val = N
        i_grid, j_grid = np.meshgrid(np.arange(N_val), np.arange(N_val), indexing="ij")
        
        X1 = np.max(X + i_grid / N_val)
        Y1 = np.max(Y + j_grid / N_val)
        X0 = np.min(X)
        Y0 = np.min(Y)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(seed)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        # Use a different vectorization: for each sample, check all tubes
        # But do it in chunks to manage memory
        chunk_size = 100000
        in_any_count = 0
        
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            a_c = a[start:end]
            b_c = b[start:end]
            c_c = c[start:end]
            
            # Shape: (chunk, N, N)
            c_col = c_c[:, np.newaxis, np.newaxis]
            a_col = a_c[:, np.newaxis, np.newaxis]
            b_col = b_c[:, np.newaxis, np.newaxis]
            
            x_left = X[np.newaxis, :, :] + c_col * i_grid[np.newaxis, :, :] / N_val
            y_left = Y[np.newaxis, :, :] + c_col * j_grid[np.newaxis, :, :] / N_val
            
            in_tube = (a_col >= x_left) & (a_col <= x_left + w) & \
                       (b_col >= y_left) & (b_col <= y_left + w)
            
            in_any_chunk = np.any(in_tube, axis=(1, 2))
            in_any_count += np.sum(in_any_chunk)
        
        fraction = in_any_count / num_samples
        volume = fraction * box_area
        return volume

    def local_search(X, Y, max_time=400.0, initial_step=0.1):
        """
        Coordinate descent local search to minimize union volume.
        """
        best_X = X.copy()
        best_Y = Y.copy()
        best_vol = compute_volume_fast(best_X, best_Y, num_samples=50000, seed=12345)
        step = initial_step
        N_val = N
        
        while time.time() - start_time < max_time:
            improved = False
            for i in range(N_val):
                for j in range(N_val):
                    if time.time() - start_time > max_time:
                        break
                    
                    # Try perturbing X[i,j]
                    for direction in [-1, 1]:
                        trial_X = best_X.copy()
                        trial_X[i, j] += direction * step
                        vol = compute_volume_fast(trial_X, best_Y, num_samples=20000, seed=12345)
                        if vol < best_vol:
                            best_X = trial_X
                            best_vol = vol
                            improved = True
                    
                    # Try perturbing Y[i,j]
                    for direction in [-1, 1]:
                        trial_Y = best_Y.copy()
                        trial_Y[i, j] += direction * step
                        vol = compute_volume_fast(best_X, trial_Y, num_samples=20000, seed=12345)
                        if vol < best_vol:
                            best_Y = trial_Y
                            best_vol = vol
                            improved = True
                if time.time() - start_time > max_time:
                    break
            
            if not improved:
                step *= 0.7
                if step < 1e-6:
                    break
            else:
                step *= 0.95
        
        return best_X, best_Y, best_vol

    # Initialize with a structured configuration
    # Key insight: tubes with same i have same x-shear, same j have same y-shear
    # Position tubes to maximize overlap
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    
    # Start with a configuration where tubes are arranged to overlap
    # The shear for tube (i,j) is (i/8, j/8) per unit height
    # To make tubes overlap, we want x_i + c*i/8 to be similar for tubes that should overlap
    # A good initial guess: arrange in a grid pattern that accounts for shear
    X = -i_grid / 16.0
    Y = -j_grid / 16.0
    
    # Run local search with time budget
    X, Y, vol = local_search(X, Y, max_time=450.0, initial_step=0.1)
    
    # Final evaluation with full sample count
    final_vol = compute_volume_fast(X, Y, num_samples=500000, seed=12345)
    
    # Reshape to (64, 2)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)
    return positions
