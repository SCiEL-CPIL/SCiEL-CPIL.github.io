"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 110.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_energy(x):
    """Compute the autoconvolution upper bound for a signed step function.
    
    The autoconvolution c_k = sum_{i} x_i * x_{i+k} for k=0,...,N-1
    The upper bound C3 is max |c_k| / (sum |x_i|)^2 or similar metric.
    
    For a signed step function with heights x_i (each +/-1 typically),
    the autoconvolution at lag k is sum_{i=0}^{N-1-k} x_i * x_{i+k}.
    The score is the maximum absolute value of these correlations.
    """
    x = np.sign(x)  # ensure binary signs
    # Compute all autocorrelations
    max_corr = 0.0
    for k in range(N):
        corr = np.sum(x[:N-k] * x[k:])
        abs_corr = abs(corr)
        if abs_corr > max_corr:
            max_corr = abs_corr
    return max_corr

def _compute_energy_fast(x):
    """Faster computation using FFT for autoconvolution."""
    x = np.sign(x)
    # Zero-pad for linear convolution
    n = 1
    while n < 2 * N - 1:
        n *= 2
    X = np.fft.fft(x, n=n)
    auto = np.fft.ifft(X * np.conj(X)).real[:N]
    return np.max(np.abs(auto))

def _energy_grad(x, grad_out):
    """Energy function for scipy minimize (smooth approximation)."""
    x_sign = np.sign(x)
    if np.all(x_sign == 0):
        x_sign = np.ones(N)
    val = _compute_energy_fast(x_sign)
    return val

def _make_binary(x):
    """Convert continuous values to binary signs."""
    return np.where(x >= 0, 1.0, -1.0)

def construct():
    """Construct a signed step function with 400 intervals to minimize autoconvolution upper bound."""
    start_time = time.time()
    
    # Initialize with random signs
    np.random.seed(42)
    best_x = np.random.choice([-1.0, 1.0], size=N)
    best_score = _compute_energy_fast(best_x)
    
    # Simulated annealing with block flips
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    block_sizes = [2, 4, 8, 16, 32]
    
    iteration = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < TIME_LIMIT - 5.0:
        # Choose a random block size and position
        bs = np.random.choice(block_sizes)
        if bs >= N:
            bs = N // 2
        pos = np.random.randint(0, N - bs + 1)
        
        # Flip the block
        trial_x = best_x.copy()
        trial_x[pos:pos+bs] *= -1.0
        
        # Compute new energy
        trial_score = _compute_energy_fast(trial_x)
        
        # Accept if better or with probability exp(-(delta)/T)
        delta = trial_score - best_score
        if delta < 0:
            best_x = trial_x
            best_score = trial_score
            no_improve = 0
        else:
            if T > 0:
                accept_prob = np.exp(-delta / T)
                if np.random.random() < accept_prob:
                    best_x = trial_x
                    best_score = trial_score
            no_improve += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        iteration += 1
        
        # Reset temperature periodically to escape local minima
        if iteration % 20000 == 0 and no_improve > max_no_improve:
            T = max(T, 5.0)
            no_improve = 0
    
    # Final local optimization: try single-bit flips
    end_time = time.time()
    remaining = TIME_LIMIT - 5.0 - (end_time - start_time)
    if remaining > 1.0:
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT - 5.0:
            improved = False
            for i in range(N):
                trial_x = best_x.copy()
                trial_x[i] *= -1.0
                trial_score = _compute_energy_fast(trial_x)
                if trial_score < best_score - 1e-12:
                    best_x = trial_x
                    best_score = trial_score
                    improved = True
    
    # Return the best signed step function
    return best_x
# EVOLVE-BLOCK-END
