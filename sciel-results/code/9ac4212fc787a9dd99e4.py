"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Structural change: treat the problem as a continuous-time / spectral one.
    # We model the step function as a band-limited signal and try to minimize
    # the peak-to-average power ratio (PAPR) of its autoconvolution.
    # By the Welch-Peakedness Theorem (loosely), minimizing peak-to-average
    # in time corresponds to maximizing entropy (uniformity) in frequency.
    #
    # We construct a sequence by:
    #   1. Starting with a random complex frequency-domain spectrum with uniform magnitude
    #      (max entropy) but random phase.
    #   2. Taking the inverse FFT to get a time-domain signal.
    #   3. Quantizing the real part to a small set of signed step heights (e.g., +/-1, or +/-a few levels).
    #   4. Refining with a few rounds of coordinate descent on the quantized levels
    #      to minimize max|autoconv| / (sum f)^2.
    #
    # We also keep the constant sequence as a baseline (theoretical optimum for L1-sum normalization).
    # We will compare the baseline with the spectral construction and return the better one.

    # Baseline: constant sequence (all +1)
    baseline = np.ones(N)

    def score_of(f):
        # Compute M / I^2
        # Autoconvolution via FFT
        n = len(f)
        fft_size = 1 << (2 * n - 1).bit_length()
        F = np.fft.rfft(f, fft_size)
        ac = np.fft.irfft(F * np.conj(F), fft_size)[:n]
        M = np.max(np.abs(ac))
        I = np.sum(f)
        if abs(I) < 1e-12:
            return np.inf
        return M / (I * I)

    base_score = score_of(baseline)
    best = baseline
    best_score = base_score

    # Spectral construction
    rng = np.random.default_rng(42)
    
    for trial in range(3):
        if time.time() > deadline:
            break

        # Random complex spectrum with uniform magnitude, random phase
        # Use full FFT for symmetry
        fft_size = 1 << (2 * N - 1).bit_length()
        freq = np.fft.rfft(np.zeros(N), fft_size)
        # Set magnitude to 1, random phase
        phases = rng.uniform(0, 2 * np.pi, size=fft_size // 2 + 1)
        spectrum = np.exp(1j * phases)
        # Ensure conjugate symmetry for real output
        # rfft already handles this, but we need to build the full spectrum properly
        # Actually, let's just use irfft on the half-spectrum
        signal = np.fft.irfft(spectrum, fft_size)[:N]
        
        # Quantize to signed levels: use a few levels, e.g., -1, 0, 1 or -1, 1
        # Let's try +/-1 first
        quantized = np.sign(signal)
        quantized[quantized == 0] = 1.0  # avoid zeros
        
        # Coordinate descent refinement
        for it in range(500):
            if time.time() > deadline:
                break
            improved = False
            idx = (it * 7 + trial) % N
            old_val = quantized[idx]
            # Try flipping or setting to other values
            for new_val in [-1.0, 1.0]:
                if new_val == old_val:
                    continue
                quantized[idx] = new_val
                s = score_of(quantized)
                if s < best_score:
                    best_score = s
                    best = quantized.copy()
                    improved = True
                    break
                quantized[idx] = old_val
            if not improved and it % 50 == 0:
                # Occasionally try a more global move
                for _ in range(5):
                    idx2 = rng.integers(0, N)
                    quantized[idx2] = -quantized[idx2]
                    s = score_of(quantized)
                    if s < best_score:
                        best_score = s
                        best = quantized.copy()
                        break
                    quantized[idx2] = -quantized[idx2]

    # Compare with baseline
    if score_of(baseline) < best_score:
        best = baseline

    return best

# EVOLVE-BLOCK-END
