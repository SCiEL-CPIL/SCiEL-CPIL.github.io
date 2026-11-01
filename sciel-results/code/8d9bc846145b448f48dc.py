"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: must finish within 120s, use 110s as safety margin
DEADLINE = 110.0


def _autocorrelation_peak(s):
    """
    Compute the maximum absolute sidelobe value of the autocorrelation.
    s is a signed step function with values in {-1, +1} of length N.
    
    The autocorrelation is defined as:
      R[k] = sum_{i=0}^{N-1-k} s[i] * s[i+k]  for k >= 0
    By symmetry, R[-k] = R[k].
    
    The peak sidelobe is max(|R[k]|) for k != 0.
    R[0] = N (the DC term / energy).
    
    We want to minimize this peak sidelobe.
    """
    # Use FFT to compute autocorrelation efficiently
    # Zero-pad to avoid circular convolution issues
    n_fft = 2 * N
    S = np.fft.rfft(s, n=n_fft)
    R = np.fft.irfft(S * np.conj(S), n=n_fft)
    # R[0] is the DC term (N), R[k] for k=1..N-1 are the sidelobes
    # The circular autocorrelation via FFT gives us the linear autocorrelation
    # in the first N samples (for non-negative lags)
    # Actually, let's be more careful.
    
    # For a real signal s of length N, the linear autocorrelation is:
    # R[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k] for k=0,...,N-1
    # Using FFT with zero-padding:
    #   S = fft(s, n_fft)
    #   R_full = ifft(S * conj(S))
    #   R_full[0:N] gives R[0], R[1], ..., R[N-1] (positive lags)
    #   R_full[n_fft-N+1:n_fft] gives R[-(N-1)], ..., R[-1] (negative lags)
    
    # Since s is real, R_full is real and R_full[k] = R_full[n_fft-k]
    # So we only need to check k=1 to N-1
    
    sidelobes = np.abs(R[1:N])
    return np.max(sidelobes) if N > 1 else 0.0


def _continuous_objective(phases):
    """
    Objective for continuous phase optimization.
    phases: array of length N (real values representing phases in [-pi, pi])
    
    We construct a complex signal z = exp(1j * phases) and consider
    its real part or a mapping to signed values.
    
    Actually, let's think differently. The problem is about signed step functions
    with values in {-1, +1}. The autocorrelation peak sidelobe is what we minimize.
    
    For the continuous relaxation, we can let x be a real vector in [-1, 1]^N
    and minimize the peak of |autocorr(x)| for non-zero lags.
    
    But the peak is non-smooth. Let's use a smooth approximation:
    sum of squares of sidelobes, or a softmax-like peak approximation.
    
    Alternative: minimize the sum of squared sidelobes (energy-based), which
    is smoother and correlated with peak minimization.
    
    Actually, let's try minimizing the max sidelobe using a smooth approximation:
    max_k |R[k]| ≈ log(sum_k exp(t * |R[k]|)) / t  for large t
    
    For simplicity and speed, let's minimize the sum of squares of sidelobes,
    which is a well-known proxy for low autocorrelation.
    """
    x = np.tanh(phases)  # Map phases to [-1, 1]
    n_fft = 2 * N
    S = np.fft.rfft(x, n=n_fft)
    R = np.fft.irfft(S * np.conj(S), n=n_fft)
    sidelobes = R[1:N]
    return np.sum(sidelobes ** 2)


def _refine_discrete(s, max_iters=500, time_limit=10.0):
    """
    Local search refinement on the discrete signed sequence.
    Try flipping individual bits and accept if it improves the peak sidelobe.
    """
    s = s.copy()
    current_peak = _autocorrelation_peak(s)
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        best_improvement = 0
        best_idx = -1
        # Try flipping each bit
        for j in range(N):
            s[j] = -s[j]
            new_peak = _autocorrelation_peak(s)
            improvement = current_peak - new_peak
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = j
            s[j] = -s[j]  # restore
        
        if best_idx >= 0:
            s[best_idx] = -s[best_idx]
            current_peak -= best_improvement
        else:
            break
    
    return s, current_peak


def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4]
    to minimize the peak sidelobe of its autocorrelation.
    
    Strategy:
    1. Start with a random initialization or a structured initialization.
    2. Use continuous relaxation (L-BFGS-B) on a smooth surrogate to find a good basin.
    3. Map to discrete via hard thresholding.
    4. Refine with local search on the discrete sequence.
    5. Return the best sequence found.
    """
    global DEADLINE
    deadline = time.time() + DEADLINE
    
    best_s = None
    best_peak = np.inf
    
    # Try multiple random initializations
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize phases randomly
        phases = rng.uniform(-np.pi, np.pi, size=N)
        
        # Continuous optimization using L-BFGS-B
        try:
            result = minimize(
                _continuous_objective,
                phases,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            phases_opt = result.x
        except Exception:
            phases_opt = phases
        
        # Map to discrete: threshold at 0
        s = np.sign(phases_opt)
        s[s == 0] = 1  # Handle exact zeros
        s = s.astype(np.float64)
        
        # Refine with local search
        time_left = deadline - time.time()
        if time_left > 5:
            s, peak = _refine_discrete(s, max_iters=200, time_limit=min(10, time_left - 2))
        else:
            peak = _autocorrelation_peak(s)
        
        if peak < best_peak:
            best_peak = peak
            best_s = s.copy()
    
    # If no valid result, fall back to all ones
    if best_s is None:
        return np.ones(N)
    
    return best_s

# EVOLVE-BLOCK-END
