import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def construct():
    N = 24
    # Pre-compute DFT matrix for fast evaluation
    num_samples = 2048
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    k_vals = np.arange(N)
    Z = np.exp(1j * np.outer(k_vals, angles)) # Shape (24, 2048)
    
    def fast_max_modulus(c):
        p_vals = Z.T @ c
        return np.max(np.abs(p_vals))
    
    def accurate_max_modulus(c):
        # Sample 2048 points
        p_vals = Z.T @ c
        mags = np.abs(p_vals)
        
        # Find top 5 peaks
        top_indices = np.argsort(mags)[-5:]
        max_val = 0.0
        
        for idx in top_indices:
            # Define a small interval around the peak
            # The grid spacing is 2*pi / 2048
            delta = 2 * np.pi / num_samples
            # Search in [angle - delta, angle + delta]
            angle_center = angles[idx]
            bounds = (angle_center - delta, angle_center + delta)
            
            # Objective: minimize -|P(z)| to maximize |P(z)|
            def objective(theta):
                z = np.exp(1j * theta)
                # Compute P(z)
                # c is array of 1/-1
                # P(z) = sum c_k z^k
                # Use dot product with powers
                # z_powers = z ** k_vals
                # This is slow if done in python loop, but we only do it ~5 times per eval
                # Vectorized:
                z_powers = np.exp(1j * theta * k_vals)
                val = np.sum(c * z_powers)
                return -np.abs(val)
            
            # Use golden section search (minimize_scalar with method='golden')
            res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-10})
            local_max = -res.fun
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    # Simulated Annealing
    # Initialize with Golay-Rudin-Shapiro (good starting point)
    p, q = [1], [1]
    while len(p) < 32:
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:N], dtype=np.float64)
    
    current_cost = fast_max_modulus(coeffs)
    best_coeffs = coeffs.copy()
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for polish and return
    
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random bit to flip
        i = random.randint(0, N - 1)
        old_val = coeffs[i]
        coeffs[i] = -old_val
        
        new_cost = fast_max_modulus(coeffs)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_coeffs = coeffs.copy()
        else:
            coeffs[i] = old_val
            
        T *= cooling
        if T < 0.01:
            break

    # Local Polish with Accurate Cost
    coeffs = best_coeffs.copy()
    current_accurate_cost = accurate_max_modulus(coeffs)
    
    improved = True
    while improved:
        improved = False
        for i in range(N):
            old_val = coeffs[i]
            coeffs[i] = -old_val
            new_accurate_cost = accurate_max_modulus(coeffs)
            if new_accurate_cost < current_accurate_cost:
                current_accurate_cost = new_accurate_cost
                improved = True
            else:
                coeffs[i] = old_val

    return [int(x) for x in coeffs]
