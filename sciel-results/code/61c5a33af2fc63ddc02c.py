"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_peak_sidelobe(s):
    """Compute the maximum absolute value of the autoconvolution sidelobes (lags 1..N-1)."""
    n = len(s)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    f = np.fft.rfft(s, n=size)
    acf = np.fft.irfft(f * np.conj(f), n=size)[:n]
    if n > 1:
        return float(np.max(np.abs(acf[1:])))
    return 0.0


def _continuous_objective(x):
    """
    Objective for continuous relaxation: minimize peak sidelobe of autocorrelation.
    x is a real-valued vector; we compute autocorrelation and return max |acf[1:]|.
    We add a small penalty to encourage values near ±1 to help rounding.
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    f = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(f * np.conj(f), n=size)[:n]
    peak_sidelobe = float(np.max(np.abs(acf[1:])))
    # Penalty to push values toward ±1
    penalty = float(np.mean((x**2 - 1.0)**2))
    return peak_sidelobe + 0.01 * penalty


def _grad_continuous_objective(x):
    """
    Numerical gradient of the continuous objective.
    Uses central differences with a small step.
    """
    n = len(x)
    eps = 1e-5
    grad = np.zeros(n)
    base = _continuous_objective(x)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (_continuous_objective(xp) - _continuous_objective(xm)) / (2 * eps)
    return grad


def _round_to_signs(x):
    """Round continuous values to ±1 signs."""
    return np.where(x >= 0, 1.0, -1.0)


def _local_search_binary(s, time_budget=10.0):
    """
    Local search on binary signs: try single-bit flips and small block flips.
    Returns the best sign sequence found.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    best_s = s.copy()
    best_val = _autoconv_peak_sidelobe(best_s)
    n = len(best_s)
    
    # Phase 1: single-bit flips
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            proposed = best_s.copy()
            proposed[i] *= -1.0
            val = _autoconv_peak_sidelobe(proposed)
            if val < best_val - 1e-12:
                best_s = proposed
                best_val = val
                improved = True
    
    # Phase 2: block flips of size 2-8
    if time.time() < deadline - 2.0:
        for block_size in [2, 3, 4, 5, 6, 8]:
            if time.time() > deadline - 2.0:
                break
            for start in range(n - block_size + 1):
                if time.time() > deadline - 2.0:
                    break
                proposed = best_s.copy()
                proposed[start:start+block_size] *= -1.0
                val = _autoconv_peak_sidelobe(proposed)
                if val < best_val - 1e-12:
                    best_s = proposed
                    best_val = val
    
    return best_s


def construct():
    """Construct a signed step function to minimize the autoconvolution peak sidelobe."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Phase 1: Continuous relaxation with gradient descent
    # Start from a random continuous vector
    x = np.random.randn(N) * 0.5
    
    # Gradient descent with adaptive step size
    lr = 0.1
    min_lr = 1e-5
    max_iter = 2000
    patience = 100
    best_obj = float('inf')
    best_x = x.copy()
    stall_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline - 30.0:  # Reserve time for local search
            break
        
        obj = _continuous_objective(x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        if stall_count > patience:
            break
        
        # Compute gradient (numerical)
        grad = _grad_continuous_objective(x)
        
        # Gradient step
        x = x - lr * grad
        
        # Adaptive learning rate
        if stall_count > patience // 2:
            lr *= 0.95
        
        # Clip to reasonable range to avoid numerical issues
        x = np.clip(x, -3.0, 3.0)
    
    # Round to binary
    s_binary = _round_to_signs(best_x)
    best_val = _autoconv_peak_sidelobe(s_binary)
    best_s = s_binary.copy()
    
    # Also try rounding from a few different continuous starting points
    for trial in range(3):
        if time.time() > deadline - 25.0:
            break
        
        # New random start
        x_trial = np.random.randn(N) * 0.5
        lr_trial = 0.1
        for it in range(1000):
            if time.time() > deadline - 25.0:
                break
            obj = _continuous_objective(x_trial)
            grad = _grad_continuous_objective(x_trial)
            x_trial = x_trial - lr_trial * grad
            x_trial = np.clip(x_trial, -3.0, 3.0)
            if it % 50 == 0:
                lr_trial *= 0.98
        
        s_trial = _round_to_signs(x_trial)
        val_trial = _autoconv_peak_sidelobe(s_trial)
        if val_trial < best_val:
            best_s = s_trial
            best_val = val_trial
    
    # Phase 2: Local search on the best binary result
    remaining_time = deadline - time.time() - 3.0
    if remaining_time > 5.0:
        best_s = _local_search_binary(best_s, time_budget=min(remaining_time, 30.0))
    
    # Final check
    if time.time() < deadline - 2.0:
        # One more round of single-bit flips if time permits
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            proposed = best_s.copy()
            proposed[i] *= -1.0
            val = _autoconv_peak_sidelobe(proposed)
            if val < _autoconv_peak_sidelobe(best_s) - 1e-12:
                best_s = proposed
    
    return best_s
# EVOLVE-BLOCK-END
