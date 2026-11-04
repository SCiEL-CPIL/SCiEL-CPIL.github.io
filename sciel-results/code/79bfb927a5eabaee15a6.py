"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _rule90_pattern(n_steps, n_cells):
    """Generate a binary pattern from Rule 90 CA run for n_steps from a single seed."""
    # Start with single 1 in the middle
    state = np.zeros(n_cells, dtype=np.uint8)
    state[n_cells // 2] = 1
    
    for _ in range(n_steps):
        # Rule 90: new[i] = left[i] XOR right[i]
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        state = left ^ right
    
    # Convert to signed: 0 -> -1, 1 -> +1
    return 2 * state.astype(np.float64) - 1.0


def _autoconv_bound(f):
    """Compute the C3 upper bound from the signed step function.
    
    For a step function with heights f[0..N-1], the autoconvolution at lag k
    is sum_i f[i]*f[i+k]. The C3 bound relates to the peak autoconvolution.
    We use the maximum absolute autoconvolution value as the metric.
    """
    # Compute autocorrelation via FFT for efficiency
    n = len(f)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    acf = np.fft.irfft(F * np.conj(F), n=size)[:n]
    # The autoconvolution upper bound C3 is related to max |acf|
    # For the inequality, we use the peak value
    return np.max(np.abs(acf))


def _local_search(f, max_iters=5000, time_budget=100.0):
    """Local search over sign flips to minimize the autoconvolution bound."""
    f = f.copy()
    current_val = _autoconv_bound(f)
    best_f = f.copy()
    best_val = current_val
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Try flipping a random position
        idx = rng.integers(0, N)
        old_sign = f[idx]
        f[idx] = -old_sign
        
        new_val = _autoconv_bound(f)
        
        if new_val < best_val:
            best_val = new_val
            best_f = f.copy()
        else:
            # Revert if not better (greedy local search)
            f[idx] = old_sign
    
    return best_f, best_val


def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Phase 1: Generate initial pattern from Rule 90 CA
    # Try different numbers of CA steps to get diverse patterns
    best_f = None
    best_val = np.inf
    
    ca_steps_options = [6, 7, 8, 9, 10, 11, 12]
    for steps in ca_steps_options:
        if time.time() > deadline:
            break
        f_init = _rule90_pattern(steps, N)
        val = _autoconv_bound(f_init)
        if val < best_val:
            best_val = val
            best_f = f_init.copy()
    
    # Also try a few random initializations
    rng = np.random.default_rng(123)
    for _ in range(5):
        if time.time() > deadline:
            break
        f_rand = rng.choice([-1.0, 1.0], size=N)
        val = _autoconv_bound(f_rand)
        if val < best_val:
            best_val = val
            best_f = f_rand.copy()
    
    # Phase 2: Local search from the best initial pattern
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_f is not None:
        best_f, best_val = _local_search(best_f, max_iters=10000, time_budget=remaining_time - 5.0)
    
    # Return the best signed step function found
    if best_f is None:
        return np.ones(N)
    return best_f
# EVOLVE-BLOCK-END
