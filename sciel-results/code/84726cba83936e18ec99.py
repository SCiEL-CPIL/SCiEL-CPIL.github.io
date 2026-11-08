import numpy as np
import time
import random
from scipy.optimize import minimize

N = 400
DEADLINE = 130.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution.
    """
    # signed_step is an array of N values
    # Use FFT to compute the autoconvolution efficiently
    # We need to pad to at least 2*N to avoid circular convolution effects
    size = 1 << (2 * N - 1).bit_length()  # Next power of 2 >= 2*N
    
    # Build the signal on a fine grid
    # Each interval has width h = 1/N
    # We'll represent the function on a grid of 'size' points
    # Optimization: vectorized construction
    f = np.zeros(size)
    # Map indices [0, size) to interval indices [0, N)
    # Since size is a multiple of N (power of 2 >= 2N, N=400, size=1024? No, 2*400=800, next pow2 is 1024. 1024/400 is not integer.)
    # Wait, 1 << (800).bit_length() is 1024.
    # 1024 is not divisible by 400.
    # The original code used:
    # start = i * (size // N)
    # end = (i + 1) * (size // N)
    # This truncates. Let's stick to the original logic to ensure consistency with the seed's metric definition, 
    # although a more accurate mapping might be better, the task says "Keep the function names and signatures... execution must explicitly select a metric".
    # The seed's metric is defined by that specific loop. I will replicate it exactly for the discrete part.
    # For the continuous part, I will use a different, more accurate mapping or the same one?
    # To be safe and consistent with the "metric" implied by the seed's score of 1.0, I should probably use the same metric.
    # However, the continuous optimization might benefit from a smoother metric.
    # Let's stick to the seed's metric for _compute_c3 to ensure comparability.
    
    # Actually, the seed's loop is slow. Let's vectorize it exactly.
    # size // N = 1024 // 400 = 2.
    # So each interval gets 2 samples?
    # i=0: [0, 2)
    # i=1: [2, 4)
    # ...
    # i=399: [798, 800)
    # The rest [800, 1024) is 0.
    # This effectively ignores the tail of the FFT padding.
    
    block_size = size // N
    # Create an array of indices for each sample in f
    # idx in [0, size). interval_idx = idx // block_size
    # But only for idx < N * block_size.
    sample_indices = np.arange(N * block_size) // block_size
    # sample_indices is [0, 0, 1, 1, ..., 399, 399]
    f[:N * block_size] = signed_step[sample_indices]
    
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # Scale by the width of each grid cell
    dx = 0.5 / size
    autoconv *= dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def _compute_c3_continuous(x):
    """
    Compute C3 for continuous heights x.
    Uses the same discretization logic as _compute_c3 for consistency.
    """
    # x is shape (N,)
    return _compute_c3(x)

def construct():
    start_time = time.time()
    
    # Initialize with all ones
    best_step = np.ones(N, dtype=np.float64)
    best_c3 = _compute_c3(best_step)
    
    # Random seed
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Continuous Optimization
    # We want to minimize max(|autoconv|). This is non-smooth.
    # We can approximate max(|y|) by sum(y^2) or use a smooth max approximation.
    # Or just minimize the L2 norm of the autoconvolution, which often correlates with reducing the max.
    # Let's try minimizing the sum of squares of the autoconvolution values.
    # This is differentiable.
    
    def objective(x):
        if time.time() - start_time > DEADLINE - 10.0:
            return 1e10 # Force stop
        
        c3_val = _compute_c3(x)
        # To make it smoother for gradient descent, let's use the sum of squares
        # But _compute_c3 returns max. We need the autoconvolution values.
        # Let's refactor to compute autoconv directly.
        
        size = 1 << (2 * N - 1).bit_length()
        block_size = size // N
        f = np.zeros(size)
        sample_indices = np.arange(N * block_size) // block_size
        f[:N * block_size] = x[sample_indices]
        
        fft_f = np.fft.fft(f)
        autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
        dx = 0.5 / size
        autoconv *= dx
        
        # Minimize L2 norm squared
        return np.sum(autoconv**2)
    
    # Initial guess: all ones
    x0 = np.ones(N)
    
    # Use L-BFGS-B
    # We can't easily compute gradients for the FFT-based objective without manual derivation or finite differences.
    # Finite differences might be too slow for 400 dimensions in 120s.
    # Let's try 'BFGS' or 'L-BFGS-B' with numerical gradients.
    # 400 dimensions * 120s. 
    # One gradient evaluation takes ~ 401 function evaluations.
    # Each function eval is an FFT of size 1024. Very fast.
    # So numerical gradients are feasible.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x_opt = res.x
    except Exception:
        x_opt = x0
    
    # Convert continuous solution to discrete {-1, 1}
    # Simple thresholding
    discrete_step = np.where(x_opt > 0, 1.0, -1.0)
    
    c3_disc = _compute_c3(discrete_step)
    if c3_disc < best_c3:
        best_c3 = c3_disc
        best_step = discrete_step.copy()
    
    # Phase 2: Discrete Local Search (Hill Climbing)
    # Start from the best discrete solution found so far
    current_step = best_step.astype(np.int8)
    current_c3 = best_c3
    
    no_improve_count = 0
    patience = 500
    
    while time.time() - start_time < DEADLINE - 2.0:
        if no_improve_count >= patience:
            break
            
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        
        if new_c3 < current_c3:
            current_c3 = new_c3
            no_improve_count = 0
            # Update best
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
        else:
            current_step[idx] = old_val
            no_improve_count += 1
            
    return best_step.astype(np.int8)

# Keep the original function name and signature
# The task says "Keep the function names and signatures of the seed program."
# Seed had: _compute_c3(signed_step) and construct()
# I have kept them.

# One detail: The seed returned `best_step` which was `np.ones(N, dtype=np.int8)`.
# My construct returns `best_step.astype(np.int8)`.
# This is consistent.

if __name__ == "__main__":
    result = construct()
    print(f"Result C3: {_compute_c3(result)}")
