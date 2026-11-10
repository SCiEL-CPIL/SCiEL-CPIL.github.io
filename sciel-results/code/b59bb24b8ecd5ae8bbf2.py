import numpy as np
from scipy.optimize import minimize
import time

def _compute_volume(positions, n_samples=500000, seed=12345):
    """
    Compute the Monte Carlo estimate of the volume of the union of the 64 tubes.
    
    Parameters:
    positions (np.ndarray): Array of shape (64, 2) containing the base positions (x_k, y_k).
    n_samples (int): Number of Monte Carlo samples.
    seed (int): Random seed for reproducibility.
    
    Returns:
    float: Estimated volume of the union.
    """
    N = 8
    i = (np.arange(N * N) // N).astype(float)
    j = (np.arange(N * N) % N).astype(float)
    
    # Generate random points in the unit cube [0, 1]^3
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, n_samples)
    b = rng.uniform(0, 1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Compute the positions of the tubes at height c
    # x_k + c * i / 8
    x_shifted = positions[:, 0] + c * i / N
    y_shifted = positions[:, 1] + c * j / N
    
    # Check if each point (a, b, c) is inside at least one tube
    # A point (a, b, c) is inside tube k if:
    # x_shifted[k] <= a <= x_shifted[k] + 1/N
    # and y_shifted[k] <= b <= y_shifted[k] + 1/N
    
    # Vectorized check for all tubes and all points
    # Shape of x_shifted: (64, 1)
    # Shape of a: (500000,)
    # We need to check for each point if there exists a tube k such that the condition holds.
    
    # For each tube k, the condition is:
    # (a >= x_shifted[k]) & (a <= x_shifted[k] + 1/N) & (b >= y_shifted[k]) & (b <= y_shifted[k] + 1/N)
    
    # To avoid a large memory footprint, we can process in batches or use a different strategy.
    # However, for n_samples=500000 and 64 tubes, the memory might be an issue if we create a (64, 500000) boolean array.
    # Let's try a more memory-efficient approach by iterating over tubes and updating a "hit" array.
    
    hit = np.zeros(n_samples, dtype=bool)
    for k in range(N * N):
        x_low = x_shifted[k]
        x_high = x_shifted[k] + 1.0 / N
        y_low = y_shifted[k]
        y_high = y_shifted[k] + 1.0 / N
        
        in_x = (a >= x_low) & (a <= x_high)
        in_y = (b >= y_low) & (b <= y_high)
        hit |= (in_x & in_y)
        
        # Early stopping if all points are hit
        if np.all(hit):
            break
    
    # Compute the bounding box
    x_min = np.min(positions[:, 0])
    x_max = np.max(positions[:, 0] + (i + 1) / N)
    y_min = np.min(positions[:, 1])
    y_max = np.max(positions[:, 1] + (j + 1) / N)
    
    volume = np.mean(hit) * (x_max - x_min) * (y_max - y_min)
    
    if not np.isfinite(volume) or volume <= 0:
        return 1e10  # Return a large value for invalid configurations
    
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    """
    Find the 64 base positions that minimize the volume of the union of the 64 sheared tubes.
    
    Returns:
    np.ndarray: Array of shape (64, 2) containing the optimized base positions.
    """
    N = 8
    initial_pos = np.random.default_rng(0).uniform(-0.5, 0.5, (N * N, 2))
    
    # Define the objective function for the optimizer
    def objective(x):
        pos = x.reshape(N * N, 2)
        return _compute_volume(pos)
    
    # Set bounds to prevent divergence
    bounds = [(-1.0, 2.0)] * (N * N * 2)
    
    # Start the optimization
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit
    
    try:
        result = minimize(
            objective,
            initial_pos.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        # Check if the optimization improved the result
        if result.fun < objective(initial_pos.flatten()):
            optimized_pos = result.x.reshape(N * N, 2)
        else:
            optimized_pos = initial_pos
            
    except Exception as e:
        # If optimization fails, return the initial position
        optimized_pos = initial_pos
    
    return optimized_pos
