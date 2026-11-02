"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_nonzero_max(x):
    """
    Compute the maximum absolute value of the linear auto-correlation 
    excluding lag 0.
    We assume the metric excludes lag 0 based on the inference that 
    all-ones is not optimal if lag 0 is included (since it would be 
    trivially optimal or all sequences equal).
    """
    # Use FFT for speed, but mask lag 0
    N = len(x)
    # Linear auto-correlation requires zero-padding to 2N
    x_padded = np.zeros(2 * N)
    x_padded[:N] = x
    X = np.fft.rfft(x_padded)
    acf = np.fft.irfft(X * np.conj(X), 2 * N)
    # The linear auto-correlation is in acf[0:N] and acf[N+1:2N] (symmetric)
    # Actually, irfft of X*conj(X) gives circular autocorr of padded signal.
    # For linear autocorr of x, we take the first N samples of the result?
    # Let's verify.
    # x = [a, b, c]
    # padded = [a, b, c, 0]
    # acf[0] = a^2+b^2+c^2
    # acf[1] = ab+bc
    # acf[2] = ac
    # acf[3] = 0
    # acf[4] = 0
    # acf[5] = 0 (wrap around? No, linear if padded enough)
    # With padding 2N, we get linear autocorr.
    # We want max of |acf[1:N]| (positive lags) and |acf[N+1:2N]| (negative lags).
    # Due to symmetry, max(positive) == max(negative).
    # So we just look at acf[1:N].
    
    if N <= 1:
        return 0.0
    
    # acf[0] is lag 0. We ignore it.
    # The relevant lags are 1 to N-1.
    side_lobs = acf[1:N]
    return np.max(np.abs(side_lobs))

def _compute_score(x):
    """
    Compute the score: max|autoconv_nonzero| / (integral)^2.
    Assuming integral is sum(x) or sum(|x|). For +/-1, sum(|x|)=N.
    If integral is sum(x), it can be small.
    Let's assume the metric is max|acf_nonzero| / (sum(x)^2) ?
    Or max|acf_nonzero| / N ?
    The prompt says "divided by squared integral".
    If we use sum(x)^2, we want large sum(x).
    If we use N (constant for +/-1), we just minimize max|acf_nonzero|.
    Let's assume the denominator is constant N for the purpose of optimization 
    (since we are constrained to +/-1, sum(|x|) is fixed). 
    If the denominator is (sum(x))^2, we have a trade-off.
    However, "signed step heights" usually implies the values are the steps.
    If the evaluator calculates the score, we must match its definition.
    Given the ambiguity, minimizing the peak side-lobe is the primary driver 
    for improvement over the "all ones" baseline (which has huge side-lobes).
    """
    peak = _autoconv_nonzero_max(x)
    # Assume denominator is N (sum of squares for +/-1) or similar constant.
    # If denominator is (sum(x))^2, we should also try to maximize |sum(x)|.
    # But for now, let's minimize the peak.
    return peak

def construct():
    """
    Strategy: Simulated Annealing / Hill Climbing on binary signs.
    Objective: Minimize the maximum absolute auto-correlation at non-zero lags.
    Method: 
    1. Initialize with random signs.
    2. Iteratively flip a single bit.
    3. Calculate the new peak.
    4. Accept if better, or with probability exp(-delta/T) if worse.
    5. Use fast O(N) update for auto-correlation? 
       Actually, N=400 is small. FFT is O(N log N) ~ 400*9 = 3600 ops.
       A flip changes acf by a linear term.
       acf_new[k] = acf_old[k] - 4 * x_i * x_{i-k} (for k>0) ... 
       This is O(N) per flip.
       We can do ~100,000 flips in 150s easily.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Compute initial peak
    current_peak = _autoconv_nonzero_max(x)
    best_x = x.copy()
    best_peak = current_peak
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    
    # Pre-compute indices for fast updates?
    # acf[k] = sum_{i=0}^{N-1-k} x[i] x[i+k]
    # If we flip x[j] -> -x[j]:
    # x_new[i] = x[i] except x_new[j] = -x[j]
    # acf_new[k] = sum x_new[i] x_new[i+k]
    # The terms involving j are:
    # 1. i=j, i+k=j+k -> term was x[j]x[j+k], becomes -x[j]x[j+k]
    # 2. i+k=j, i=j-k -> term was x[j-k]x[j], becomes -x[j-k]x[j]
    # So delta_acf[k] = -2 * x[j] * (x[j+k] + x[j-k]) (with bounds)
    # This is O(N) to update the whole acf array for one flip.
    
    # Let's implement a fast flip update
    def get_acf(x):
        N = len(x)
        x_padded = np.zeros(2 * N)
        x_padded[:N] = x
        X = np.fft.rfft(x_padded)
        acf = np.fft.irfft(X * np.conj(X), 2 * N)
        return acf[:N] # Only positive lags 0..N-1

    acf = get_acf(x)
    # We only care about acf[1:N]
    
    for iteration in range(200000): # Max iterations
        if time.time() > deadline:
            break
        
        # Pick a random index to flip
        j = rng.randint(N)
        
        # Calculate change in acf
        # old_val = x[j]
        # new_val = -old_val
        # delta = new - old = -2 * old_val
        delta = -2.0 * x[j]
        
        # Update acf
        # acf[k] = sum_{i} x[i] x[i+k]
        # Change in acf[k] = delta * x[j+k] + delta * x[j-k] (if indices valid)
        # Note: acf is symmetric, but we only stored 0..N-1.
        
        new_acf = acf.copy()
        
        # For each lag k in 1..N-1:
        # Contribution from index j:
        # If j+k < N: term x[j]x[j+k] changes by delta * x[j+k]
        # If j-k >= 0: term x[j-k]x[j] changes by x[j-k] * delta
        
        # Vectorized update:
        # k = 0..N-1
        # x[j+k] is valid if j+k < N -> k < N-j
        # x[j-k] is valid if j-k >= 0 -> k <= j
        
        # Let's just do it simply for N=400. It's fast enough.
        # Actually, let's use the FFT method for simplicity and robustness, 
        # N=400 is tiny.
        x[j] = -x[j]
        new_acf_full = get_acf(x)
        new_peak = np.max(np.abs(new_acf_full[1:N]))
        
        # Decide accept/reject
        delta_peak = new_peak - current_peak
        if delta_peak < 0 or rng.rand() < np.exp(-delta_peak / T):
            # Accept
            current_peak = new_peak
            acf = new_acf_full
            if current_peak < best_peak:
                best_peak = current_peak
                best_x = x.copy()
        else:
            # Reject
            x[j] = -x[j]
        
        T *= cooling_rate
    
    return best_x
# EVOLVE-BLOCK-END
