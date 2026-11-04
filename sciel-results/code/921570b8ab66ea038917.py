"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

# Deadline guard: keep a safety margin before the 120s limit
_DEADLINE = 105.0  # seconds from start of construct()

def _compute_c3_upper_bound(f):
    """
    Compute the upper bound on C3 for a signed step function f with N intervals.
    
    The autoconvolution is (f * f)(x) = integral f(x-t) f(t) dt.
    For a step function on [-1/4, 1/4] with N equal intervals, we discretize.
    
    The 'upper bound on C3' is interpreted as the maximum absolute value of the
    autoconvolution over all lags (excluding the trivial self-lag which is sum(f^2)).
    
    Using FFT for efficiency:
    - Pad f to length 2*N to avoid circular convolution artifacts
    - Compute linear convolution via FFT
    - Return max |conv[k]| for k != 0 (excluding self-lag)
    """
    n = len(f)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real
    # conv[0] is the sum of squares (self-lag), which for ±1 is n
    # We want the max over non-zero lags
    # The convolution result has length size, but we only care about lags 0..n-1
    # Actually for linear convolution of two length-n sequences, result has length 2n-1
    # We take conv[0:n] which corresponds to lags 0 to n-1
    conv_lags = conv[:n]
    # Exclude lag 0 (self-convolution)
    if len(conv_lags) > 1:
        return np.max(np.abs(conv_lags[1:]))
    return 0.0


def _score(f):
    """Score is the upper bound on C3; lower is better."""
    return _compute_c3_upper_bound(f)


def _try_flip(f, i):
    """Try flipping position i and return (new_score, new_f)."""
    f_new = f.copy()
    f_new[i] *= -1.0
    return _score(f_new), f_new


def _local_search(f, max_iterations=5000, deadline=None):
    """
    Local search by flipping individual bits.
    Uses a greedy approach: repeatedly find the best single flip.
    """
    current_f = f.copy()
    current_score = _score(current_f)
    
    for _ in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        best_score = current_score
        best_idx = -1
        
        # Try all possible flips
        for i in range(N):
            f_new = current_f.copy()
            f_new[i] *= -1.0
            s = _score(f_new)
            if s < best_score:
                best_score = s
                best_idx = i
        
        if best_idx == -1:
            # No improvement found
            break
        
        current_f[best_idx] *= -1.0
        current_score = best_score
    
    return current_f, current_score


def _smooth_score(f, temperature=0.1):
    """
    Smooth surrogate for the max |conv| objective using log-sum-exp.
    This makes the objective differentiable for gradient-based optimization.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # Exclude lag 0
    conv_nonzero = conv[1:] if len(conv) > 1 else conv
    abs_conv = np.abs(conv_nonzero)
    # Log-sum-exp for smooth max
    max_val = np.max(abs_conv)
    if max_val < 1e-10:
        return 0.0
    # Soft max: sum(exp((x - max)/T)) * T + max
    scaled = (abs_conv - max_val) / temperature
    log_sum_exp = np.log(np.sum(np.exp(scaled))) + max_val / temperature
    return log_sum_exp * temperature


def _construct_initial():
    """Generate an initial signed step function."""
    # Start with alternating pattern or random
    # A good heuristic: try to make the autoconvolution small
    # Start with random signs
    rng = np.random.RandomState(42)
    f = rng.choice([-1.0, 1.0], size=N)
    return f


def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Start with an initial guess
    f = _construct_initial()
    best_f = f.copy()
    best_score = _score(f)
    
    # Phase 1: Local search with bit flips
    if time.time() < deadline - 30:
        f, score = _local_search(f, max_iterations=2000, deadline=deadline - 30)
        if score < best_score:
            best_f = f.copy()
            best_score = score
    
    # Phase 2: Try multiple random restarts
    rng = np.random.RandomState(123)
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random restart with some structure
        f_restart = rng.choice([-1.0, 1.0], size=N)
        # Add some structure: make blocks of same sign
        block_size = N // (4 + r)
        for i in range(0, N, block_size):
            sign = rng.choice([-1.0, 1.0])
            end = min(i + block_size, N)
            f_restart[i:end] = sign
        
        f_new, score_new = _local_search(f_restart, max_iterations=1000, deadline=deadline - 20)
        if score_new < best_score:
            best_f = f_new.copy()
            best_score = score_new
    
    # Phase 3: Try a smooth optimization approach
    if time.time() < deadline - 10:
        # Use continuous relaxation: f in [-1, 1]
        x0 = best_f.copy()
        
        def objective(x):
            if time.time() > deadline - 5:
                return best_score
            return _smooth_score(x, temperature=0.5)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            bounds=[(-1, 1)] * N,
                            options={'maxiter': 100, 'ftol': 1e-8})
            # Round to ±1
            f_smooth = np.sign(result.x)
            # Handle zeros
            f_smooth[f_smooth == 0] = 1.0
            score_smooth = _score(f_smooth)
            if score_smooth < best_score:
                best_f = f_smooth
                best_score = score_smooth
        except Exception:
            pass
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        f_final, score_final = _local_search(best_f, max_iterations=500, deadline=deadline - 2)
        if score_final < best_score:
            best_f = f_final.copy()
            best_score = score_final
    
    return best_f

# EVOLVE-BLOCK-END
