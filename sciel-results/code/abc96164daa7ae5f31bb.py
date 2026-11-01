"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_c3(heights):
    """Compute the C3 upper bound: max |autoconvolution| / (sum |heights|)^2."""
    # Autoconvolution via FFT
    n = len(heights)
    f = np.fft.fft(heights, n=2*n)
    autoconv = np.fft.ifft(f * np.conj(f)).real[:n]
    peak = np.max(np.abs(autoconv))
    norm = np.sum(np.abs(heights))
    if norm == 0:
        return float('inf')
    return peak / (norm * norm)


def _smooth_objective(x):
    """
    Smooth approximation of the C3 objective for gradient-based optimization.
    x is a vector of length N. h = tanh(x) gives values in (-1, 1).
    We use soft-max and smooth absolute value.
    """
    h = np.tanh(x)
    
    # Compute autoconvolution via FFT for speed
    n = len(h)
    f = np.fft.fft(h, n=2*n)
    autoconv = np.fft.ifft(f * np.conj(f)).real[:n]
    
    # Smooth absolute value
    eps = 1e-12
    abs_conv = np.sqrt(autoconv**2 + eps)
    
    # Soft max with temperature
    T = 0.5
    max_abs = np.max(abs_conv)
    shifted = (abs_conv - max_abs) / T
    # Clip for numerical stability
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
    soft_max = T * log_sum_exp
    
    # Sum of absolute values of h (smooth)
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    if sum_abs_h < 1e-10:
        return 1e10
        
    value = soft_max / (sum_abs_h ** 2)
    return value


def _hill_climb(heights, deadline, max_no_improve=20):
    """
    Discrete hill-climbing refinement on signed step function.
    Returns best heights and best c3 found.
    """
    best_heights = heights.copy()
    best_c3 = _compute_c3(best_heights)
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        # Try flipping each bit
        for idx in range(N):
            if time.time() >= deadline:
                break
            new_heights = best_heights.copy()
            new_heights[idx] = -new_heights[idx]
            new_c3 = _compute_c3(new_heights)
            if new_c3 < best_c3:
                best_heights = new_heights
                best_c3 = new_c3
                improved = True
                no_improve_count = 0
                break  # Restart scan after an improvement
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
                
    return best_heights, best_c3


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_heights = None
    best_c3 = float('inf')
    
    # Phase 1: Gradient-based optimization to find good starting points
    # Use multiple initializations for the smooth problem
    
    # Initialization 1: Alternating signs (high frequency)
    x0_alt = np.zeros(N)
    x0_alt[::2] = 3.0
    x0_alt[1::2] = -3.0
    
    # Initialization 2: All positive
    x0_pos = np.full(N, 3.0)
    
    # Initialization 3: Random
    np.random.seed(42)
    x0_rand = np.random.randn(N) * 2.0
    
    # Initialization 4: Block structure
    x0_block = np.zeros(N)
    block_size = 10
    for i in range(N // block_size):
        sign = 1 if i % 2 == 0 else -1
        x0_block[i*block_size:(i+1)*block_size] = sign * 3.0
    
    initializations = [x0_alt, x0_pos, x0_rand, x0_block]
    
    for x0 in initializations:
        if time.time() >= deadline:
            break
            
        try:
            res = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Convert back to signed step function
            h_opt = np.tanh(res.x)
            # Threshold to get signed step function
            heights = np.sign(h_opt)
            # Handle zeros (should be rare)
            heights[heights == 0] = 1
            
            c3 = _compute_c3(heights)
            if c3 < best_c3:
                best_c3 = c3
                best_heights = heights.copy()
                
        except Exception:
            continue
    
    # Phase 2: Discrete hill-climbing refinement
    if best_heights is None:
        best_heights = np.ones(N)
        best_c3 = _compute_c3(best_heights)
        
    best_heights, best_c3 = _hill_climb(best_heights, deadline)
    
    # Phase 3: Random restarts with hill-climbing
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for restart in range(5):
            if time.time() >= deadline:
                break
                
            # Create perturbed version
            perturbed = best_heights.copy()
            n_flips = np.random.randint(5, 20)
            flip_indices = np.random.choice(N, size=n_flips, replace=False)
            perturbed[flip_indices] *= -1
            
            # Refine from perturbed state
            refined_heights, refined_c3 = _hill_climb(perturbed, deadline, max_no_improve=10)
            
            if refined_c3 < best_c3:
                best_c3 = refined_c3
                best_heights = refined_heights.copy()
    
    # Final check: ensure we return a valid signed step function
    best_heights = np.sign(best_heights)
    best_heights[best_heights == 0] = 1
    
    return best_heights
# EVOLVE-BLOCK-END
