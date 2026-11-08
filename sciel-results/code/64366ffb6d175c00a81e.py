import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 400
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_c3_discrete(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution.
    """
    # signed_step is an array of N values in {-1, 1}
    # We use a fine grid to compute the autoconvolution accurately.
    # The intervals are on [-0.25, 0.25], so total width is 0.5.
    # Each interval has width h = 0.5 / N.
    
    # To avoid circular convolution artifacts, we need a grid size >= 2*N.
    # Let's use a power of 2 for FFT efficiency.
    size = 1 << (2 * N - 1).bit_length()
    
    # Construct the signal on the fine grid
    f = np.zeros(size)
    # Map each of the N intervals to 'size // N' points in the fine grid
    # Note: size is a multiple of N? No, size is power of 2.
    # We need to map indices carefully.
    # Actually, simpler: just use the direct convolution on the N-length vector 
    # if we treat it as a discrete sequence? 
    # The task says "signed step function with 400 equal intervals".
    # The autoconvolution of a step function is a continuous function.
    # The maximum absolute value of this continuous function is the metric.
    
    # Let's stick to the FFT method used in the seed but ensure it's fast.
    # The seed used a loop to fill f. We can vectorize this.
    # indices for each bin
    # The fine grid has 'size' points over [-0.5, 0.5] (periodic).
    # The function is defined on [-0.25, 0.25] and is 0 elsewhere? 
    # Or is it periodic? "Step function on [-1/4, 1/4]".
    # Usually, in these contexts, it's a function on the torus or R.
    # If it's on R, the support is [-0.25, 0.25].
    # The autoconvolution support is [-0.5, 0.5].
    
    # Let's assume the standard interpretation:
    # f(x) = sum_{i=0}^{N-1} s_i * 1_{[-0.25 + i*h, -0.25 + (i+1)*h)}(x)
    # where h = 0.5/N.
    
    # The autoconvolution (f * f)(t) = integral f(x) f(t-x) dx.
    
    # We can compute this via FFT on a grid that covers [-0.5, 0.5].
    # Let grid size M be such that M * dx = 1.0 (period).
    # dx = 1.0 / M.
    # We need M to be large enough to resolve the step function.
    # M >= 2*N is good. Let M = 4*N or next power of 2.
    
    M = 1 << (2 * N - 1).bit_length()
    dx = 1.0 / M
    
    # Create the signal f on the grid [0, 1)
    # The function is non-zero on [-0.25, 0.25], which corresponds to [0.75, 1.0) U [0.0, 0.25) in [0, 1).
    # This is messy. Let's shift the domain to [0, 1) for the function support.
    # Let g(u) be defined on [0, 1) such that g(u) = f(u - 0.25) for u in [0.25, 0.75]?
    # No, simpler:
    # Define f on [0, 1) with support [0.25, 0.75].
    # The intervals are [0.25, 0.25+h), ..., [0.75-h, 0.75).
    
    f_grid = np.zeros(M)
    start_idx = int(0.25 * M)
    end_idx = int(0.75 * M)
    
    # Distribute the N values into the [start_idx, end_idx) range
    # The width of the support is 0.5, which is M/2 points.
    # So each of the N intervals corresponds to M/(2N) points.
    block_size = M // (2 * N)
    
    # Fill the grid
    # For i in 0..N-1:
    #   f_grid[start_idx + i*block_size : start_idx + (i+1)*block_size] = signed_step[i]
    
    for i in range(N):
        s_idx = start_idx + i * block_size
        e_idx = s_idx + block_size
        f_grid[s_idx:e_idx] = signed_step[i]
        
    # Compute autoconvolution
    # Note: numpy fft does circular convolution.
    # Since support is in [0.25, 0.75], the autoconvolution support is in [0.5, 1.5] mod 1?
    # No, the autoconvolution of a function supported in [0.25, 0.75] is supported in [0.5, 1.5].
    # In the periodic domain [0, 1), this wraps around.
    # The peak is likely at t=0 (shifted) or t=0.5?
    # Let's just compute the circular autoconvolution and take the max absolute value.
    # The "upper bound" C3 is likely the L-infinity norm of the autoconvolution.
    
    fft_f = fft(f_grid)
    autoconv = ifft(fft_f * np.conj(fft_f)).real
    autoconv *= dx  # Scale by integration step
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_continuous(x):
    """
    Surrogate objective for continuous optimization.
    x is a vector of length N, values in [-1, 1].
    We use the sum of squares of the autoconvolution coefficients as a smooth proxy.
    This is related to the L2 norm of the autoconvolution, which is easier to optimize.
    Minimizing the L2 norm often correlates with minimizing the Linf norm for these types of signals.
    """
    # x is continuous
    # Construct f_grid
    M = 1 << (2 * N - 1).bit_length()
    dx = 1.0 / M
    start_idx = int(0.25 * M)
    block_size = M // (2 * N)
    
    f_grid = np.zeros(M)
    # Vectorized fill
    # Create indices for each block
    # This loop is N=400, which is fast.
    for i in range(N):
        s_idx = start_idx + i * block_size
        e_idx = s_idx + block_size
        f_grid[s_idx:e_idx] = x[i]
        
    fft_f = fft(f_grid)
    power_spectrum = np.abs(fft_f)**2
    # Parseval's theorem: sum |f|^2 = sum |F|^2 / M
    # The L2 norm of the autoconvolution is related to the L2 norm of f.
    # Actually, let's just compute the autoconvolution and return sum of squares.
    # This is differentiable.
    autoconv = ifft(fft_f * np.conj(fft_f)).real
    autoconv *= dx
    
    # Return sum of squares as a smooth objective
    return np.sum(autoconv**2)

def construct():
    start_time = time.time()
    
    # Initialize with all ones
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3_discrete(best_step)
    
    # 1. Continuous Optimization Phase
    # We try to find a good continuous vector x that minimizes the surrogate.
    # Then we round it to {-1, 1}.
    
    # Initial guess: random continuous vector in [-1, 1]
    np.random.seed(42)
    random.seed(42)
    
    x0 = np.random.uniform(-1, 1, N)
    
    # Define bounds
    bounds = [(-1, 1)] * N
    
    # Optimize
    try:
        if time.time() - start_time < DEADLINE - 10:
            res = minimize(_compute_c3_continuous, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-5})
            x_opt = res.x
        else:
            x_opt = x0
    except Exception:
        x_opt = x0
    
    # Round to discrete
    step_from_cont = np.sign(x_opt).astype(np.int8)
    # Handle zeros (should be rare)
    step_from_cont[step_from_cont == 0] = 1
    
    c3_from_cont = _compute_c3_discrete(step_from_cont)
    
    if c3_from_cont < best_c3:
        best_c3 = c3_from_cont
        best_step = step_from_cont.copy()
    
    # 2. Local Search Refinement Phase
    # Use the result from continuous optimization as the starting point for local search.
    # This is different from the seed which started with all ones.
    
    current_step = best_step.copy()
    current_c3 = best_c3
    
    # Pre-compute for speed?
    # We'll use a fast version of the discrete check.
    
    # Simulated Annealing / Hill Climbing
    # We have ~130s left.
    
    T_start = 0.05
    T_end = 1e-6
    alpha = 0.995
    
    steps = 50000
    for i in range(steps):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Multi-flip move: flip 1 or 2 bits
        if np.random.random() < 0.8:
            num_flips = 1
        else:
            num_flips = 2
            
        indices = np.random.choice(N, num_flips, replace=False)
        
        old_vals = current_step[indices].copy()
        current_step[indices] *= -1
        
        new_c3 = _compute_c3_discrete(current_step)
        
        delta = new_c3 - current_c3
        
        if delta < 0:
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_c3 = new_c3
                else:
                    current_step[indices] = old_vals
    
    return best_step
