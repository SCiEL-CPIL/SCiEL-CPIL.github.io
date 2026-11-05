"""
Flattest plus-minus-one polynomials with 24 coefficients.
Strategy:
  1. Spectral synthesis initialization: inverse FFT of flattened target waveforms
     to create globally coherent starting points that bypass local minima traps.
  2. Fast FFT-based sup-norm screening on a dense grid.
  3. Golden-section refinement of the top few peaks for accurate evaluation.
  4. Multi-start local search: spectral-synthesis restarts + simulated annealing / 2-bit flips.
  5. Deadline guard: return best found before 140 s wall-clock.
"""
import numpy as np
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150 s kill

N = 24
SQRTN = math.sqrt(N)

# Precompute powers of z for evaluation at a given theta (vectorized)
_KS = np.arange(N)

def _time_left():
    return _DEADLINE - time.time()

def _eval_poly(coeffs, theta):
    """Evaluate |P(e^{i theta})| where P(z) = sum c_k z^k."""
    z = np.exp(1j * theta)
    # Use dot product with precomputed powers
    return float(np.abs(np.dot(coeffs, z ** _KS)))

def _refine_peak(coeffs, ang, iters=50):
    """
    Golden-section refine the maximum of |P(e^{i theta})| near ang.
    We search in a small window around ang.
    """
    # Window: half-width ~ pi/512 (since grid is 8192 points)
    half_w = math.pi / 4096.0
    a = ang - half_w
    b = ang + half_w
    
    # Golden ratio
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _eval_poly(coeffs, c)
    fd = _eval_poly(coeffs, d)
    
    for _ in range(iters):
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _eval_poly(coeffs, d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _eval_poly(coeffs, c)
    
    mid = (a + b) / 2.0
    return _eval_poly(coeffs, mid)

def _supnorm_precise(coeffs, n_grid=8192):
    """
    Compute sup-norm: FFT screening on dense grid, then refine top peaks.
    """
    # FFT screening
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    abs_vals = np.abs(vals)
    
    # Get top 10 peak indices
    top_idx = np.argsort(abs_vals)[-10:][::-1]
    
    best = 0.0
    for idx in top_idx:
        ang = 2.0 * math.pi * idx / n_grid
        val = _refine_peak(coeffs, ang, iters=40)
        if val > best:
            best = val
    
    return best

def _supnorm_fast(coeffs, n_grid=4096):
    """Fast sup-norm estimate via FFT only (no refinement)."""
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _local_search(coeffs, max_iters=2000):
    """
    2-bit-flip local search with simulated annealing.
    Returns best coeffs and best sup-norm found.
    """
    coeffs = list(coeffs)
    best_coeffs = list(coeffs)
    best_val = _supnorm_fast(coeffs)
    
    # Use fast screening for most iterations, precise for candidates
    temp = 0.5
    cool_rate = 0.999
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Pick two distinct indices to flip
        i, j = random.sample(range(N), 2)
        coeffs[i] = -coeffs[i]
        coeffs[j] = -coeffs[j]
        
        new_val = _supnorm_fast(coeffs)
        
        if new_val < best_val:
            best_val = new_val
            best_coeffs = list(coeffs)
        elif new_val < best_val + temp:
            # Accept with some probability (annealing)
            if random.random() < math.exp(-(new_val - best_val) / (temp + 1e-12)):
                pass  # keep the move
            else:
                # Reject: undo
                coeffs[i] = -coeffs[i]
                coeffs[j] = -coeffs[j]
        else:
            # Undo
            coeffs[i] = -coeffs[i]
            coeffs[j] = -coeffs[j]
        
        temp *= cool_rate
    
    # Refine the best found
    final_val = _supnorm_precise(best_coeffs)
    return best_coeffs, final_val

def _try_golay_rudin_shapiro():
    """Golay-Rudin-Shapiro recursion, truncated to 24 coefficients."""
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    return p[:24]

def _spectral_synthesis_init(seed_idx=0):
    """
    Generate a coherent ±1 sequence via inverse FFT of a flattened target waveform.
    
    The idea: we want the frequency response |P(e^{iθ})| to be as flat as possible.
    Instead of random ±1, we construct a complex signal whose real part (after
    appropriate scaling and thresholding) gives us a ±1 sequence with low peaks.
    
    Approach:
      1. Create a target spectrum with controlled smoothness (e.g., cosine taper).
      2. Take inverse FFT to get a time-domain signal.
      3. Threshold the real part to ±1.
      4. This yields a globally coherent starting point.
    
    Different seed_idx values produce different target waveforms for diversity.
    """
    rng = np.random.RandomState(seed_idx * 1000 + 42)
    
    # Choose a smoothing parameter: controls how "flat" the target spectrum is
    # alpha=0 gives flat spectrum, alpha=1 gives cosine-tapered (smoother)
    alpha = 0.3 + 0.5 * (seed_idx % 5) / 5.0  # varies with seed for diversity
    
    # Create target spectrum: start with flat, apply cosine taper
    n_spec = 64
    freqs = np.arange(n_spec)
    # Cosine taper: 0.5 * (1 + cos(pi * freqs / (n_spec-1)))
    taper = 0.5 * (1.0 + np.cos(np.pi * freqs / (n_spec - 1)))
    
    # Mix flat and tapered based on alpha
    target = (1 - alpha) * np.ones(n_spec) + alpha * taper
    target = target / np.max(target)  # normalize
    
    # Add small random phase perturbations for diversity
    phases = rng.uniform(0, 2 * np.pi, n_spec)
    target_complex = target * np.exp(1j * phases)
    
    # Inverse FFT to get time-domain signal
    time_signal = np.fft.ifft(target_complex)
    
    # Extract real part and threshold to ±1
    real_part = np.real(time_signal[:N])
    
    # Threshold: sign with small random tie-breaking
    coeffs = np.sign(real_part)
    # Handle zeros with random ±1
    zero_mask = (coeffs == 0)
    if np.any(zero_mask):
        coeffs[zero_mask] = rng.choice([1, -1], size=np.sum(zero_mask))
    
    # Convert to list of ints
    return [int(c) for c in coeffs]

def _spectral_synthesis_init_v2(seed_idx=0):
    """
    Alternative spectral synthesis: use a sum of cosines as the target waveform.
    This creates a more structured, low-peak spectrum.
    """
    rng = np.random.RandomState(seed_idx * 2000 + 123)
    
    # Create a target as sum of a few low-frequency cosines
    n_spec = 64
    freqs = np.arange(n_spec)
    theta = 2 * np.pi * freqs / n_spec
    
    # Random low-frequency components
    n_components = 2 + (seed_idx % 3)  # 2, 3, or 4 components
    target = np.ones(n_spec)  # DC component
    
    for k in range(n_components):
        # Random frequency in the low range
        f = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.1, 0.5)
        target += amp * np.cos(2 * np.pi * f * freqs / n_spec + phase)
    
    # Ensure positivity and normalize
    target = np.abs(target)
    target = target / np.max(target)
    
    # Inverse FFT
    time_signal = np.fft.ifft(target)
    real_part = np.real(time_signal[:N])
    
    # Threshold to ±1
    coeffs = np.sign(real_part)
    zero_mask = (coeffs == 0)
    if np.any(zero_mask):
        coeffs[zero_mask] = rng.choice([1, -1], size=np.sum(zero_mask))
    
    return [int(c) for c in coeffs]

def _spectral_synthesis_init_v3(seed_idx=0):
    """
    Third variant: use the magnitude of a random polynomial's spectrum as target,
    then smooth it heavily to create a flat-ish target.
    """
    rng = np.random.RandomState(seed_idx * 3000 + 456)
    
    # Start with a random ±1 sequence
    rand_coeffs = rng.choice([1, -1], size=N)
    
    # Compute its spectrum
    n_spec = 128
    padded = np.zeros(n_spec, dtype=float)
    padded[:N] = rand_coeffs
    spectrum = np.abs(np.fft.fft(padded))
    
    # Smooth the spectrum heavily using a moving average
    kernel_size = 5 + (seed_idx % 4) * 2  # 5, 7, 9, or 11
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(spectrum, kernel, mode='same')
    
    # Normalize
    smoothed = smoothed / np.max(smoothed)
    
    # Use the smoothed spectrum as target for inverse FFT
    # Take only the first N/2 + 1 frequencies (real signal)
    n_half = n_spec // 2 + 1
    target_half = smoothed[:n_half]
    
    # Reconstruct full spectrum (Hermitian symmetry)
    target_full = np.zeros(n_spec, dtype=complex)
    target_full[:n_half] = target_half
    for k in range(1, n_spec // 2):
        target_full[n_spec - k] = np.conj(target_full[k])
    
    # Inverse FFT
    time_signal = np.fft.ifft(target_full)
    real_part = np.real(time_signal[:N])
    
    # Threshold to ±1
    coeffs = np.sign(real_part)
    zero_mask = (coeffs == 0)
    if np.any(zero_mask):
        coeffs[zero_mask] = rng.choice([1, -1], size=np.sum(zero_mask))
    
    return [int(c) for c in coeffs]

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    random.seed(42)
    np.random.seed(42)
    
    # Start with Golay-Rudin-Shapiro as initial best
    best_coeffs = _try_golay_rudin_shapiro()
    best_val = _supnorm_precise(best_coeffs)
    
    # Generate spectral synthesis initial points
    spectral_inits = []
    for s in range(20):
        spectral_inits.append(_spectral_synthesis_init(s))
        spectral_inits.append(_spectral_synthesis_init_v2(s))
        spectral_inits.append(_spectral_synthesis_init_v3(s))
    
    # Multi-start local search: first try spectral synthesis inits, then random
    n_restarts = 500
    for r in range(n_restarts):
        if _time_left() < 10.0:
            break
        
        if r < len(spectral_inits):
            # Use spectral synthesis initialization
            coeffs = spectral_inits[r]
        else:
            # Random initial polynomial
            coeffs = [random.choice([1, -1]) for _ in range(N)]
        
        # Local search from this start
        new_coeffs, new_val = _local_search(coeffs, max_iters=1500)
        
        if new_val < best_val:
            best_val = new_val
            best_coeffs = new_coeffs
    
    # Final precise evaluation of the best
    final_val = _supnorm_precise(best_coeffs)
    
    return best_coeffs

# Allow module-level access if needed
if __name__ == "__main__":
    result = construct()
    print(result)
