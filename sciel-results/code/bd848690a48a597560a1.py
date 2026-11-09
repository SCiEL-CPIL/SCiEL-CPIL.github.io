import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_coeffs = 24
    sqrt_n = np.sqrt(n_coeffs)
    
    # Initial guess: Golay-Rudin-Shapiro sequence truncated to 24
    # P -> (P, Q), Q -> (P, -Q)
    p, q = [1], [1]
    while len(p) < 32:  # Generate 32 to be safe, then truncate
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:n_coeffs], dtype=np.int8)
    
    # Precompute angle grid for fast evaluation
    # Use a dense grid to approximate the sup norm
    n_grid = 2048
    thetas = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    # V[k, i] = exp(i * k * theta_i)
    # We can compute this efficiently
    # k is 0..23
    k_vals = np.arange(n_coeffs)
    # thetas shape (n_grid,), k_vals shape (n_coeffs,)
    # We want matrix of shape (n_coeffs, n_grid)
    # exp(i * k * theta)
    phases = np.exp(1j * np.outer(k_vals, thetas))
    
    def evaluate(coeffs):
        # coeffs is int8 array of length n_coeffs
        # Compute P(theta) = sum c_k * exp(i*k*theta)
        # This is a matrix-vector product: phases.T @ coeffs
        # phases shape (24, 2048), coeffs shape (24,)
        # Result shape (2048,)
        vals = phases.T @ coeffs.astype(np.complex128)
        max_val = np.max(np.abs(vals))
        return max_val / sqrt_n

    best_score = evaluate(best_coeffs)
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.9999
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.integers(0, n_coeffs)
        
        # Flip
        old_val = current_coeffs[idx]
        current_coeffs[idx] = -old_val
        
        new_score = evaluate(current_coeffs)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_coeffs = current_coeffs.copy()
        else:
            # Reject
            current_coeffs[idx] = old_val
            
        T = max(T_min, T * cooling_rate)
        
        # If we've cooled down significantly and found a good local min, maybe restart?
        # For now, just keep going.
        
    # Ensure we return Python integers
    return [int(c) for c in best_coeffs]
# EVOLVE-BLOCK-END
