"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_sidelobes(a):
    """
    Compute the maximum absolute value of the autocorrelation 
    for non-zero lags.
    """
    n = len(a)
    size = 2 * n
    fa = np.fft.rfft(a, size)
    power = fa * np.conj(fa)
    c = np.fft.irfft(power, size)
    sidelobes = c[1:n]
    return np.max(np.abs(sidelobes))

def _score_val(a):
    """
    Calculate the raw value: max_abs_sidelobe / (sum a^2)^2
    Assuming sum a^2 = N for binary sequences.
    """
    max_side = _autocorr_sidelobes(a)
    integral_sq = (np.sum(a**2))**2
    return max_side / integral_sq

def _psd_to_binary(psd, rng):
    """
    Given a Power Spectral Density (PSD) of length N (real, non-negative),
    reconstruct a binary sequence.
    1. Take square root to get magnitude.
    2. Assign random phases.
    3. Inverse FFT to get time domain signal.
    4. Threshold to +/- 1.
    5. Refine via local search.
    """
    # PSD is defined on the positive frequencies for rfft
    # We need to construct the full complex FFT.
    # rfft of real signal x of length N returns N//2 + 1 values.
    # Let's assume psd is the magnitude squared of the rfft coefficients.
    
    # Actually, let's work with the full complex spectrum for easier phase manipulation.
    # Or simpler: generate magnitude from PSD, random phase, ifft, threshold.
    
    # psd input: array of length N//2 + 1 (rfft size)
    # Magnitude = sqrt(psd)
    mags = np.sqrt(psd + 1e-12)
    
    # Random phases
    phases = rng.uniform(0, 2*np.pi, size=len(mags))
    # DC and Nyquist (if N even) should have real values (phase 0 or pi) for real signal?
    # Actually, if we want a real signal, the spectrum must be conjugate symmetric.
    # rfft handles this implicitly if we ifft it correctly?
    # np.fft.irfft expects the rfft format.
    # If we just do irfft(mags * exp(i*phases)), will it be real?
    # Yes, irfft assumes the input is the first half of a conjugate symmetric spectrum.
    # However, the phases of the DC and Nyquist bins must be 0 or pi for the result to be real?
    # No, irfft just reconstructs the real signal corresponding to that half-spectrum.
    # The imaginary parts cancel out in the reconstruction.
    
    complex_spec = mags * np.exp(1j * phases)
    # Ensure DC and Nyquist are real?
    # For a real signal, X[0] and X[N/2] (if N even) must be real.
    # If we assign random phase to DC, the signal might have a non-zero mean offset?
    # Actually, if X[0] is complex, the resulting time domain signal is still real?
    # No. The inverse DFT of a non-conjugate-symmetric spectrum is complex.
    # irfft assumes the input IS the conjugate-symmetric half.
    # So if I provide a complex number for DC, irfft treats it as the DC component.
    # The resulting signal will be real.
    # But wait, if X[0] = A * e^{i phi}, then the contribution to x[n] is A * e^{i phi}.
    # For x[n] to be real, the sum of all contributions must be real.
    # The conjugate symmetry is built into irfft.
    # So x[n] = sum_{k=0}^{N/2} ... 
    # It works.
    
    a_real = np.fft.irfft(complex_spec, n=N)
    
    # Threshold to +/- 1
    a_bin = np.sign(a_real)
    # Handle zeros (unlikely with random phase)
    a_bin[a_bin == 0] = 1
    
    # Local refinement: Flip bits to reduce max sidelobe
    # This is crucial because the thresholding step introduces errors.
    # We use a simple greedy local search.
    
    # Precompute autocorrelation for speed?
    # _autocorr_sidelobes is O(N log N).
    # Let's do a few iterations of flipping the worst lag's contributing bits?
    # Or just random flips with acceptance.
    
    current_val = _score_val(a_bin)
    
    # Simple local search
    no_improve = 0
    max_iters = 500 # Keep it fast per candidate
    
    for _ in range(max_iters):
        if no_improve > 50:
            break
            
        # Pick random index
        idx = rng.integers(0, N)
        old_val = a_bin[idx]
        a_bin[idx] = -old_val
        
        new_val = _score_val(a_bin)
        
        if new_val < current_val:
            current_val = new_val
            no_improve = 0
        else:
            a_bin[idx] = old_val
            no_improve += 1
            
    return a_bin, current_val

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    rng = np.random.default_rng(42)
    
    best_a = None
    best_val = float('inf')
    
    # Strategy:
    # 1. Start with a "flat" PSD (white noise) or slightly shaped PSD.
    # 2. Optimize the PSD to minimize the peak of the inverse FFT (autoconvolution).
    #    The autoconvolution is the inverse FFT of the PSD.
    #    We want to minimize max |c[k]| for k!=0.
    #    This is equivalent to minimizing the peak of the sidelobes of the sinc-like function.
    #    We can use gradient descent on the log-PSD.
    
    # Initialize PSD
    # Uniform PSD corresponds to white noise, which has sinc autocorrelation.
    # We want to "flatten" the autocorrelation sidelobes.
    # This means making the PSD more "peaked" at DC? Or "flat"?
    # Actually, for Barker sequences, the PSD is relatively flat.
    # Let's start with a flat PSD and see if we can shape it.
    
    # Number of frequency bins for rfft
    n_bins = N // 2 + 1
    
    # Initial PSD: Uniform (White Noise)
    # Or maybe a raised cosine to reduce edge effects?
    psd = np.ones(n_bins)
    
    # Gradient Descent on PSD
    # Objective: max |irfft(psd)[1:N]|
    # This is non-smooth due to max and abs.
    # We can approximate by minimizing the sum of squares of sidelobes, or just the max.
    # Let's minimize the peak sidelobe.
    
    # Since we can't easily differentiate max, we can use a heuristic:
    # Find the lag with the largest sidelobe.
    # Adjust the PSD to reduce the contribution to that lag.
    # c[k] = sum psd[m] * exp(-i 2 pi k m / N) ... roughly.
    # Actually c[k] is the inverse DFT of PSD.
    # To reduce c[k], we want to change PSD such that its projection onto the k-th harmonic is reduced.
    
    # Alternative: Just use random restarts of the PSD shape + Phase Retrieval.
    # Given the time limit, a robust search over PSD shapes might be better than gradient descent.
    
    # Let's try a few PSD shapes:
    # 1. Flat (White)
    # 2. Raised Cosine
    # 3. Gaussian
    # 4. Random Smooth
    
    shapes = []
    
    # 1. Flat
    shapes.append(np.ones(n_bins))
    
    # 2. Raised Cosine
    freqs = np.linspace(0, 1, n_bins)
    rc = 0.5 * (1 + np.cos(2 * np.pi * freqs))
    shapes.append(rc)
    
    # 3. Gaussian (centered at 0)
    gauss = np.exp(-10 * freqs**2)
    shapes.append(gauss)
    
    # 4. Random Smooth
    for _ in range(5):
        # Generate random low-pass filtered noise
        rand_spec = rng.standard_normal(n_bins)
        # Smooth it
        kernel = np.hanning(n_bins)
        rand_spec_smooth = np.convolve(rand_spec, kernel, mode='same')
        # Ensure positive
        rand_spec_smooth = np.abs(rand_spec_smooth) + 0.1
        # Normalize
        rand_spec_smooth /= np.sum(rand_spec_smooth)
        shapes.append(rand_spec_smooth)
        
    # 5. "Barker-like" PSD? 
    # Actually, let's just try optimizing the PSD via simple gradient steps on the peak.
    
    # Let's combine: Use the shapes as initializations for a local search on the PSD.
    # For each shape, run a few iterations of:
    # 1. Compute c = irfft(psd)
    # 2. Find k_max with max |c[k]|
    # 3. Update psd to reduce c[k_max]
    #    Gradient of c[k] w.r.t psd[m] is exp(-i 2 pi k m / N)
    #    We want to decrease |c[k]|.
    #    If c[k] is real (it is, because psd is real and symmetric? No, c is real because psd is real and even? 
    #    Wait, psd is the magnitude squared. The autocorrelation c is real and even.
    #    c[k] = sum psd[m] cos(2 pi k m / N) (roughly)
    #    So gradient is cos(2 pi k m / N).
    #    We want to move psd in the direction that reduces c[k_max].
    #    If c[k_max] > 0, decrease psd where cos > 0, increase where cos < 0.
    #    If c[k_max] < 0, increase psd where cos > 0, decrease where cos < 0.
    
    for shape in shapes:
        if time.time() > deadline - 20:
            break
            
        psd = shape.copy()
        # Normalize PSD? The scale doesn't matter for the binary sequence, only the shape.
        # But for gradient step size, let's keep it normalized.
        
        # Iterate
        for it in range(50):
            if time.time() > deadline - 10:
                break
                
            # Compute autocorrelation from PSD
            # c = irfft(psd)
            # Note: irfft expects the rfft format.
            # psd is the power spectrum.
            # c[k] = sum_{m} psd[m] * exp(i 2 pi k m / N) ?
            # Let's verify.
            # If x is time domain, X = fft(x). PSD = |X|^2.
            # Autocorr c = ifft(PSD).
            # Yes.
            
            c = np.fft.irfft(psd, n=N)
            
            # Find max sidelobe
            # c[0] is the peak. We care about c[1:N]
            sidelobes = c[1:N]
            k_max = np.argmax(np.abs(sidelobes)) + 1
            val_max = sidelobes[k_max-1]
            
            # If val_max is small, we might be converged
            if abs(val_max) < 1e-6:
                break
                
            # Compute gradient
            # c[k] = sum_{m=0}^{N/2} psd[m] * 2 * cos(2 pi k m / N) (approx, with DC and Nyquist handling)
            # Actually, let's just use the complex form for gradient direction.
            # c[k] = ifft(psd)[k]
            # d c[k] / d psd[m] = exp(-i 2 pi k m / N) ? 
            # Let's check.
            # c = ifft(psd). 
            # c[k] = (1/N) sum_{j} psd[j] exp(i 2 pi k j / N)
            # So gradient is exp(i 2 pi k j / N).
            
            # We want to change psd to reduce |c[k_max]|.
            # Let g = exp(i 2 pi k_max * m / N) for m in 0..N/2.
            # We want to update psd -= eta * Re(conj(c[k_max]) * g) ?
            # Actually, c[k_max] is real.
            # We want to minimize c[k_max]^2.
            # d (c^2) / d psd = 2 c * dc/dpsd.
            # So step is - c * dc/dpsd.
            
            m_indices = np.arange(n_bins)
            # Phase term
            phase_terms = np.exp(1j * 2 * np.pi * k_max * m_indices / N)
            
            # Gradient of c[k_max] w.r.t psd
            # Since c is real, and psd is real, the effective gradient is the real part of the phase term?
            # Let's just use the real part of the complex gradient, scaled by the sign of c[k_max].
            
            grad = np.real(phase_terms)
            
            # Step direction: opposite to sign of c[k_max] * grad
            # If c[k_max] > 0, we want to decrease c[k_max], so move psd in direction -grad.
            # If c[k_max] < 0, we want to increase c[k_max] (make it less negative), so move psd in direction +grad.
            # So direction = -sign(c[k_max]) * grad
            
            direction = -np.sign(val_max) * grad
            
            # Normalize direction to avoid huge steps
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction /= direction_norm
                
            # Learning rate
            eta = 0.1
            
            psd_new = psd + eta * direction
            
            # Ensure positivity
            psd_new = np.maximum(psd_new, 1e-6)
            
            # Optional: Normalize to keep scale consistent
            psd_new /= np.sum(psd_new)
            
            psd = psd_new
            
        # After PSD optimization, reconstruct binary sequence
        a_candidate, val_candidate = _psd_to_binary(psd, rng)
        
        if val_candidate < best_val:
            best_val = val_candidate
            best_a = a_candidate.copy()
            
    # Fallback to random if nothing found (should not happen)
    if best_a is None:
        best_a = rng.choice([-1, 1], size=N)
        best_val = _score_val(best_a)
        
    return best_a
# EVOLVE-BLOCK-END
