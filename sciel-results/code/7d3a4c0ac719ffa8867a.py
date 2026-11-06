import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0 # 110s safety margin out of 120s

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values just in case
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)

    # Scale x to [-1, 1] for numerical stability in Legendre fit
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case: all x are the same
        # Fit a constant (degree 0)
        c0 = np.mean(y)
        return np.array([c0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Stage 1: Nadaraya-Watson smoothing to denoise the target
    # This estimates E[y|x] more robustly than raw y
    x_sorted, sort_idx = np.sort(x), np.argsort(x)
    y_sorted = y[sort_idx]
    
    # Choose bandwidth based on data spread
    # A common heuristic is Scott's rule or similar, scaled to the data range
    # For 1D kernel regression, h ~ n^(-1/5) * sigma
    n = len(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    bandwidth = sigma * (n ** (-1.0/5.0))
    
    # Ensure bandwidth is not too small or too large
    min_bw = 1e-6
    max_bw = x_range * 0.1
    bandwidth = np.clip(bandwidth, min_bw, max_bw)
    
    # Compute smoothed y values
    # We do this efficiently by sorting and using a sliding window or just full computation
    # Given n is likely manageable, let's do a vectorized-ish approach or simple loop
    # Actually, for large N, full O(N^2) might be slow. Let's check N.
    # If N is huge, we might need to subsample or use a faster method.
    # But typically training sets for these tasks are moderate.
    
    y_smoothed = np.zeros_like(y_sorted)
    
    # To speed up, we can use the fact that x is sorted.
    # For each point, only points within ~3*bandwidth contribute significantly.
    # However, implementing a sliding window kernel sum is complex.
    # Let's try a direct calculation but optimized with numpy broadcasting if possible,
    # or just a loop if N is small.
    
    # Let's check if N is small enough for O(N^2)
    if n < 5000:
        # Direct O(N^2) calculation is fine
        for i in range(n):
            dist = np.abs(x_sorted - x_sorted[i])
            weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
            y_smoothed[i] = np.sum(weights * y_sorted) / np.sum(weights)
    else:
        # For large N, subsample or use a faster approximation
        # Let's subsample the training data for the smoothing step to keep it fast
        # We'll pick a subset of points to compute the kernel sum against
        # This is an approximation but should be good enough for denoising
        if n > 10000:
            idx_sub = np.random.choice(n, 5000, replace=False)
            x_sub = x_sorted[idx_sub]
            y_sub = y_sorted[idx_sub]
        else:
            x_sub = x_sorted
            y_sub = y_sorted
        
        # Compute smoothed values using the subset
        # This is still O(N * M) where M is subset size
        # We can do this in chunks to avoid memory issues
        chunk_size = 1000
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            x_chunk = x_sorted[i_start:i_end]
            # Distances to all points in subset
            dist = np.abs(x_chunk[:, np.newaxis] - x_sub[np.newaxis, :])
            weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
            y_smoothed[i_start:i_end] = np.sum(weights * y_sub, axis=1) / np.sum(weights, axis=1)
            
    # Unsort y_smoothed to match original x order
    y_smoothed_original = np.empty_like(y_smoothed)
    y_smoothed_original[sort_idx] = y_smoothed

    # Check time
    if time.time() > deadline:
        # Fallback to raw fit if smoothing took too long
        x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)
        leg_coeffs = np.polynomial.legendre.legfit(x_scaled, y, 5)
        monomial_u = np.polynomial.legendre.leg2poly(leg_coeffs)
        c_x = np.zeros(6)
        A = 1.0 / (x_range / 2.0)
        B = -(x_min + x_max) / 2.0 / (x_range / 2.0)
        powers = []
        current_power = np.array([1.0])
        for i in range(6):
            powers.append(current_power.copy())
            if i < 5:
                new_len = len(current_power) + 1
                new_power = np.zeros(new_len)
                new_power[0] = current_power[0] * B
                for k in range(1, len(current_power)):
                    new_power[k] = current_power[k-1] * A + current_power[k] * B
                new_power[-1] = current_power[-1] * A
                current_power = new_power
        for i in range(6):
            p = powers[i]
            padded_p = np.zeros(6)
            padded_p[:len(p)] = p
            c_x += monomial_u[i] * padded_p
        return c_x

    # Stage 2: Fit polynomial to the smoothed target
    # Use Legendre fit on scaled x for stability
    x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)
    
    # Fit Legendre polynomial of degree 5 to the smoothed data
    leg_coeffs = np.polynomial.legendre.legfit(x_scaled, y_smoothed_original, 5)
    
    # Convert Legendre coefficients to standard monomial coefficients
    monomial_u = np.polynomial.legendre.leg2poly(leg_coeffs)
    
    # Transform from scaled variable u to original x
    c_x = np.zeros(6)
    
    # u = A * x + B
    A = 1.0 / (x_range / 2.0)
    B = -(x_min + x_max) / 2.0 / (x_range / 2.0)
    
    # Compute powers of (A*x+B)
    powers = []
    current_power = np.array([1.0]) # constant 1
    for i in range(6):
        powers.append(current_power.copy())
        if i < 5:
            new_len = len(current_power) + 1
            new_power = np.zeros(new_len)
            new_power[0] = current_power[0] * B
            for k in range(1, len(current_power)):
                new_power[k] = current_power[k-1] * A + current_power[k] * B
            new_power[-1] = current_power[-1] * A
            current_power = new_power
            
    # Sum a_i * powers[i]
    for i in range(6):
        p = powers[i]
        padded_p = np.zeros(6)
        padded_p[:len(p)] = p
        c_x += monomial_u[i] * padded_p
        
    # Check time
    if time.time() > deadline:
        return c_x
        
    return c_x
