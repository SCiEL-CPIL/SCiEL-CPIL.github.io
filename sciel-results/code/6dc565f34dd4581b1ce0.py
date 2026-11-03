import time
import math
import numpy as np
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 115.0  # safety margin before 120s kill

def compute_autoconv(f):
    """
    Compute the autoconvolution of a step function with N intervals on [-1/4, 1/4].
    The autoconvolution at shift k (in interval units) is:
    C[k] = sum_i f[i] * f[i + k]  for k >= 0 (with appropriate wrapping or zero-padding)
    
    Actually, for a function on [-1/4, 1/4] with N intervals, the autoconvolution
    is (f * f)(t) = integral f(s) f(t-s) ds.
    
    For step functions on equal intervals, this becomes a discrete convolution.
    The interval width is h = (1/4 - (-1/4)) / N = (1/2) / N = 1/(2N).
    
    C[k] = h * sum_i f[i] * f[i+k] for the k-th shift in interval units.
    But we need to be careful about the domain. The function is defined on [-1/4, 1/4],
    so f[i] is zero outside that range. The autoconvolution will be defined on [-1/2, 1/2].
    
    For the discrete case:
    - f has N values, indexed 0 to N-1, representing intervals on [-1/4, 1/4]
    - The autoconvolution C has 2N-1 values (from shift -(N-1) to N-1)
    - C[k] for k >= 0: h * sum_{i=0}^{N-1-k} f[i] * f[i+k]
    - C[-k] = C[k] by symmetry
    
    The "normalized maximum" is max |C[k]| / C[0] where C[0] = h * sum_i f[i]^2.
    But C[0] is the peak at zero shift, so we normalize by that.
    """
    h = 1.0 / (2.0 * N)
    # Use FFT for efficient convolution
    # Zero-pad to avoid circular convolution
    nfft = 2 * N
    F = np.fft.rfft(f, n=nfft)
    autoconv = np.fft.irfft(F * np.conj(F), n=nfft)
    # The autoconvolution is centered at index N//2 in the FFT output
    # Actually, for irfft, the peak is at index 0
    # Let's think: if f is real and we compute f * f via FFT, the result is centered
    # at index 0 (or n//2 for negative shifts)
    # autoconv[0] = sum f[i]^2 * h (approximately, with FFT scaling)
    # autoconv[k] = sum f[i] * f[i+k] * h for positive k
    # autoconv[nfft-k] = sum f[i] * f[i-k] * h for positive k (negative shifts)
    
    # The full autoconvolution has peaks at both ends and center
    # We want the maximum over all shifts
    # Due to symmetry, we can look at autoconv[0:N] and take the max
    # But autoconv[0] is the zero-shift peak
    # The sidelobes are at autoconv[1:N] and autoconv[nfft-N+1:nfft]
    
    peak = autoconv[0]
    if peak <= 0:
        return 1.0  # degenerate case
    
    # The maximum sidelobe is max(autoconv[1:N], autoconv[nfft-N+1:nfft])
    # But due to symmetry, autoconv[k] = autoconv[nfft-k]
    # So we just need max(autoconv[1:N])
    max_sidelobe = np.max(autoconv[1:N]) if N > 1 else 0.0
    
    return max_sidelobe / peak

def objective_and_grad(x, f_current):
    """
    Objective: normalized max autoconvolution peak.
    This is not smooth, so we use a smooth approximation.
    We can use the sum of squares of sidelobes or a soft-max.
    """
    h = 1.0 / (2.0 * N)
    nfft = 2 * N
    F = np.fft.rfft(x, n=nfft)
    autoconv = np.fft.irfft(F * np.conj(F), n=nfft)
    
    peak = autoconv[0]
    if peak <= 0:
        return 1e10, np.zeros_like(x)
    
    # Soft approximation of max: use sum of squares of positive sidelobes
    # Or use a p-norm with large p
    sidelobes = autoconv[1:N]
    p = 20  # large p to approximate max
    norm = np.sum(np.maximum(sidelobes, 0.0) ** p) ** (1.0/p)
    val = norm / peak
    return val, None

def construct():
    start_time = time.time()
    
    # Start with a random binary sequence
    np.random.seed(42)
    best_f = np.random.randint(0, 2, size=N).astype(float)
    best_score = compute_autoconv(best_f)
    
    # Try to improve with local optimization
    # Use scipy.optimize.minimize with a smooth objective
    # But the objective is non-smooth, so we'll use Nelder-Mead
    
    # Alternative: use a greedy approach
    # Compute autoconv, find worst shift, try flipping bits to reduce it
    
    while time.time() - start_time < TIME_LIMIT - 5:
        # Compute autoconvolution
        h = 1.0 / (2.0 * N)
        nfft = 2 * N
        F = np.fft.rfft(best_f, n=nfft)
        autoconv = np.fft.irfft(F * np.conj(F), n=nfft)
        peak = autoconv[0]
        
        if peak <= 0:
            break
        
        # Find the shift with maximum sidelobe
        sidelobes = autoconv[1:N]
        worst_idx = np.argmax(sidelobes)
        worst_val = sidelobes[worst_idx]
        
        if worst_val / peak < 1e-6:
            break
        
        # Try flipping each bit and see if it reduces the worst sidelobe
        improved = False
        # Only try a subset of bits for efficiency
        indices = np.random.choice(N, size=min(N, 200), replace=False)
        
        for i in indices:
            # Flip bit i
            old_val = best_f[i]
            new_val = 1.0 - old_val
            best_f[i] = new_val
            
            # Compute new autoconv (only need the worst shift)
            F_new = np.fft.rfft(best_f, n=nfft)
            autoconv_new = np.fft.irfft(F_new * np.conj(F_new), n=nfft)
            peak_new = autoconv_new[0]
            
            if peak_new > 0:
                new_worst = autoconv_new[worst_idx + 1] / peak_new
            else:
                new_worst = 1.0
            
            if new_worst < worst_val / peak:
                improved = True
                break
            else:
                # Revert
                best_f[i] = old_val
        
        if not improved:
            # Try a random restart with a different seed
            np.random.seed(int(time.time() * 1000) % 1000000)
            best_f = np.random.randint(0, 2, size=N).astype(float)
            best_score = compute_autoconv(best_f)
        
        # Check if we're still improving
        current_score = compute_autoconv(best_f)
        if current_score < best_score:
            best_score = current_score
    
    # Ensure we return a valid nonnegative sequence
    return [float(x) for x in best_f]
