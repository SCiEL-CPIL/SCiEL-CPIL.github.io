# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _keich_positions(N):
    """Keich binary construction for N = 2^k base positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _compute_volume(positions, N=8, n_samples=500000, seed=12345):
    """Compute the Monte Carlo estimate of the union volume exactly as in the task."""
    # positions shape (N*N, 2)
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    # At height c, tube (i,j) occupies:
    # x in [x_k + c*i/N, x_k + c*i/N + 1/N]
    # y in [y_k + c*j/N, y_k + c*j/N + 1/N]
    
    X0 = np.min(x)
    X1 = np.max(x + (i_flat + 1) / N)  # Wait, the task says max_k (x_k + (i+1)/8)
    # Let me re-read: X1 = max_k (x_k + (i+1)/8)
    # Actually, looking at the tube definition: [x_k + c*i/8, x_k + c*i/8 + 1/8]
    # So the x-range at height c is [x_k + c*i/N, x_k + c*i/N + 1/N]
    # The overall x-range is from min over all k and c of (x_k + c*i/N) to max over all k and c of (x_k + c*i/N + 1/N)
    # Since c in [0,1], the min x for tube k is x_k + 0*i/N = x_k (if i>=0)
    # The max x for tube k is x_k + 1*i/N + 1/N = x_k + (i+1)/N
    # So X0 = min_k x_k, X1 = max_k (x_k + (i+1)/N)
    
    X1 = np.max(x + (i_flat + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_flat + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0  # Invalid
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is inside at least one tube
    # For each tube k: x_k + c*i_k/N <= a <= x_k + c*i_k/N + 1/N
    #                   y_k + c*j_k/N <= b <= y_k + c*j_k/N + 1/N
    # This is equivalent to:
    #   a - c*i_k/N - 1/N <= x_k <= a - c*i_k/N
    #   b - c*j_k/N - 1/N <= y_k <= b - c*j_k/N
    
    # Vectorized check: for each point, check if any tube contains it
    # a: (n_samples,), b: (n_samples,), c: (n_samples,)
    # x: (64,), i_flat: (64,), y: (64,), j_flat: (64,)
    
    # For efficiency, we can check all tubes for all points
    # a - c[:,None]*i_flat[None,:]/N - 1/N <= x[None,:] <= a - c[:,None]*i_flat[None,:]/N
    # This creates a (n_samples, 64) boolean array, which might be large
    # n_samples = 500000, 64 tubes -> 32 million booleans, ~32 MB, should be fine
    
    a_exp = a[:, None]
    b_exp = b[:, None]
    c_exp = c[:, None]
    
    x_shifted = a_exp - c_exp * i_flat[None, :] / N
    y_shifted = b_exp - c_exp * j_flat[None, :] / N
    
    in_x = (x_shifted >= x[None, :]) & (x_shifted <= x[None, :] + 1/N)
    in_y = (y_shifted >= y[None, :]) & (y_shifted <= y[None, :] + 1/N)
    
    inside = np.any(in_x & in_y, axis=1)
    
    fraction = np.mean(inside)
    V = fraction * box_area
    
    if not np.isfinite(V) or V <= 0:
        return 0.0
    return V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with Keich construction
    keich = _keich_positions(N)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat].copy()
    base_y = keich[j_flat].copy()
    positions = np.stack([base_x, base_y], axis=1)
    
    # Evaluate initial volume
    V_current = _compute_volume(positions, N, n_samples=500000, seed=12345)
    
    if V_current <= 0 or not np.isfinite(V_current):
        return positions
    
    # Try to improve using local optimization
    # We'll use a simple gradient-free approach with random perturbations
    # and keep the best result
    
    best_positions = positions.copy()
    best_V = V_current
    
    # Use scipy.optimize if available
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy:
        # Define objective function for optimization
        # We want to minimize V
        # But V is computed via Monte Carlo, which is noisy
        # Instead, let's use a deterministic approximation or just do random search
        
        # Let's try a random restart approach with local perturbations
        rng_opt = np.random.default_rng(42)
        
        n_restarts = 20
        n_local_steps = 50
        step_size = 0.05
        
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization near Keich
            if restart == 0:
                x_opt = base_x.copy()
                y_opt = base_y.copy()
            else:
                x_opt = base_x + rng_opt.uniform(-0.2, 0.2, size=N*N)
                y_opt = base_y + rng_opt.uniform(-0.2, 0.2, size=N*N)
            
            V_opt = _compute_volume(np.stack([x_opt, y_opt], axis=1), N, n_samples=50000, seed=12345)
            
            if V_opt <= 0 or not np.isfinite(V_opt):
                continue
            
            for step in range(n_local_steps):
                if time.time() > deadline - 30:
                    break
                
                # Random perturbation
                dx = rng_opt.uniform(-step_size, step_size, size=N*N)
                dy = rng_opt.uniform(-step_size, step_size, size=N*N)
                
                x_new = x_opt + dx
                y_new = y_opt + dy
                
                V_new = _compute_volume(np.stack([x_new, y_new], axis=1), N, n_samples=50000, seed=12345)
                
                if V_new > 0 and np.isfinite(V_new) and V_new < V_opt:
                    x_opt = x_new
                    y_opt = y_new
                    V_opt = V_new
                    step_size = max(0.01, step_size * 0.9)
                else:
                    step_size = min(0.5, step_size * 1.1)
            
            # Final evaluation with full samples
            V_final = _compute_volume(np.stack([x_opt, y_opt], axis=1), N, n_samples=500000, seed=12345)
            
            if V_final > 0 and np.isfinite(V_final) and V_final < best_V:
                best_V = V_final
                best_positions = np.stack([x_opt, y_opt], axis=1)
    
    # Return the best result found
    return best_positions


# EVOLVE-BLOCK-END
