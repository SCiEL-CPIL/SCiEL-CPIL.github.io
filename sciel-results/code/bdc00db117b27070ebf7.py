"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(x):
    """Compute maximum absolute value of the autoconvolution of x."""
    # Use FFT for efficient convolution
    n = len(x)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, fft_size)
    conv = np.fft.irfft(X * X, fft_size)[:size]
    return np.max(np.abs(conv))

def _score(x):
    """Compute the score: max_abs_autoconv / (sum(x))^2"""
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    max_conv = _autoconv_max_abs(x)
    return max_conv / (integral * integral)

def _barker_like_init(n, seed):
    """Generate a deterministic pattern inspired by Barker/Golay constructions."""
    rng = np.random.RandomState(seed)
    # Start with a random binary pattern
    x = rng.choice([-1, 1], size=n)
    return x

def _gauss_seidel_flip(x, max_iters=50, deadline=None):
    """Iterative sign-flip optimization using Gauss-Seidel style."""
    x = x.copy().astype(np.float64)
    n = len(x)
    best_score = _score(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping sign at position i
            old_val = x[i]
            x[i] = -x[i]
            new_score = _score(x)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                x[i] = old_val
        if not improved:
            break
    return x, best_score

def _simulated_annealing(x, start_temp=1.0, end_temp=0.01, max_iters=100, deadline=None):
    """Simulated annealing for sign-flip optimization."""
    x = x.copy().astype(np.float64)
    n = len(x)
    best_x = x.copy()
    best_score = _score(x)
    current_score = best_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        for _ in range(n // 10):  # Some flips per iteration
            if deadline is not None and time.time() > deadline:
                break
            i = np.random.randint(n)
            old_val = x[i]
            x[i] = -x[i]
            new_score = _score(x)
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                x[i] = old_val
    return best_x, best_score

def construct():
    """Construct a signed step function minimizing max absolute autoconvolution."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Try multiple initial patterns with different seeds
    best_x = None
    best_score = float('inf')
    
    # Generate several diverse initial patterns
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Create initial pattern
        x_init = _barker_like_init(N, seed)
        
        # Phase 1: Gauss-Seidel local optimization
        x_gs, score_gs = _gauss_seidel_flip(x_init, max_iters=20, deadline=deadline)
        
        if score_gs < best_score:
            best_score = score_gs
            best_x = x_gs.copy()
        
        # Phase 2: Simulated annealing from best solution
        if time.time() < deadline - 5:
            x_sa, score_sa = _simulated_annealing(x_gs, start_temp=0.5, end_temp=0.01, 
                                                  max_iters=50, deadline=deadline)
            if score_sa < best_score:
                best_score = score_sa
                best_x = x_sa.copy()
    
    # Final polish with more iterations on the best solution
    if best_x is not None and time.time() < deadline - 2:
        x_final, score_final = _gauss_seidel_flip(best_x, max_iters=30, deadline=deadline)
        if score_final < best_score:
            best_x = x_final
            best_score = score_final
    
    # Convert to integer sequence (±1)
    if best_x is not None:
        result = np.where(best_x > 0, 1, -1).astype(np.int8)
    else:
        # Fallback: all ones
        result = np.ones(N, dtype=np.int8)
    
    return result
# EVOLVE-BLOCK-END
