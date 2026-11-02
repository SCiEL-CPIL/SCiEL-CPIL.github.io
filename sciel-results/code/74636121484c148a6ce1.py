"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def _fast_autocorr_max_abs(x):
    """
    Compute the maximum absolute value of the linear autocorrelation of x.
    Uses FFT for O(N log N) complexity.
    """
    # To compute linear autocorrelation via FFT, we zero-pad to avoid circular convolution artifacts.
    # The length of linear autocorrelation is 2*N - 1.
    # We need a transform size M >= 2*N - 1.
    # A power of 2 is efficient for FFT.
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    X = np.fft.fft(x, M)
    # Power spectrum
    P = X * np.conj(X)
    # Inverse FFT
    ac = np.fft.ifft(P).real
    
    # The first N values of ac correspond to lags 0 to N-1.
    # Due to symmetry, ac[k] for k >= 0 is the correlation.
    # We want max absolute value.
    return np.max(np.abs(ac[:N]))

def _spectral_init(rng):
    """
    Generate an initial signal using spectral phase retrieval principles.
    We fix the magnitude spectrum to a raised-cosine (Hann-like) window to
    minimize the peak-to-sidelobe ratio in the time domain autoconvolution,
    and assign random phases.
    """
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    # Create a raised-cosine magnitude spectrum
    # We want a smooth spectrum to avoid sharp transitions in time domain
    freqs = np.fft.fftfreq(M)
    # Normalize frequency to [-0.5, 0.5]
    freqs_norm = freqs * M
    
    # Create a window in frequency domain
    # We want to keep low frequencies and taper high frequencies
    # A simple raised cosine window centered at 0
    w = np.ones(M)
    # Apply a smooth taper
    # Let's use a sinc-like or just a smooth polynomial decay
    # Actually, a flat-top window in frequency corresponds to a specific time-domain shape.
    # But we want to minimize autoconvolution peaks.
    # A Gaussian spectrum gives a Gaussian time-domain signal, which has good autocorr properties.
    # Let's use a Gaussian magnitude spectrum.
    sigma = M / 8.0  # Bandwidth control
    mag = np.exp(-0.5 * (freqs_norm / sigma) ** 2)
    
    # Ensure symmetry for real signal
    # mag[k] should equal mag[M-k]
    # The Gaussian is symmetric.
    
    # Random phase
    phase = rng.uniform(0, 2 * np.pi, M)
    # Ensure phase symmetry for real signal: phase[k] = -phase[M-k]
    # phase[0] and phase[M/2] (if M even) should be 0 or pi
    phase[0] = 0
    if M % 2 == 0:
        phase[M // 2] = 0
        
    # Construct spectrum
    spectrum = mag * np.exp(1j * phase)
    
    # Enforce Hermitian symmetry
    # spectrum[k] = conj(spectrum[M-k])
    # This is tricky to enforce directly with random phases.
    # Better approach: generate random real signal in time, FFT, keep magnitude, randomize phase of FFT, IFFT.
    
    # Let's restart the init logic properly.
    # 1. Start with a random time-domain signal? No, we want spectral control.
    # 2. Generate a random phase spectrum that is Hermitian.
    
    # Generate random phases for positive frequencies
    half_M = M // 2
    phases = rng.uniform(0, 2 * np.pi, half_M)
    
    # Construct full spectrum
    spec = np.zeros(M, dtype=np.complex128)
    spec[0] = mag[0]  # DC component, real
    if M % 2 == 0:
        spec[half_M] = mag[half_M]  # Nyquist, real
    
    for k in range(1, half_M):
        spec[k] = mag[k] * np.exp(1j * phases[k-1])
        spec[M-k] = np.conj(spec[k])
        
    # Inverse FFT to get time domain signal
    x = np.fft.ifft(spec).real
    
    # Normalize to unit norm or similar scale?
    # The task says "signed step heights". Usually implies +/-1.
    # But the optimization might work better in continuous domain first, then quantize?
    # The seed program returns int8.
    # Let's optimize in continuous domain, then round to +/-1.
    # However, the autocorrelation of a continuous signal is different from the quantized one.
    # The task asks for a sequence of signed step heights.
    # If we optimize continuous x, the max abs autocorr of x might be low, but after rounding to +/-1, it might jump.
    # Strategy: Optimize continuous x to minimize max abs autocorr. Then map to +/-1.
    # But the evaluator measures the RAW value of the returned sequence.
    # So we must evaluate the +/-1 sequence.
    
    # Alternative: Optimize the +/-1 sequence directly using gradient-free or discrete methods?
    # The inspiration suggests spectral init enables L-BFGS to converge to better local minima.
    # This implies optimizing a continuous relaxation.
    
    # Let's try:
    # 1. Initialize x from spectral method.
    # 2. Optimize x (continuous) to minimize max abs autocorr.
    # 3. Threshold x to +/-1.
    # 4. Refine the +/-1 sequence using local search (flips) to fix any degradation from thresholding.
    
    return x

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_val = float('inf')
    best_x = np.zeros(N, dtype=np.float64)
    
    # Generate multiple diverse initializations
    num_inits = 10
    inits = []
    for _ in range(num_inits):
        x_init = _spectral_init(rng)
        # Normalize to unit norm to keep values in a reasonable range for optimization
        norm = np.linalg.norm(x_init)
        if norm > 0:
            x_init = x_init / norm
        inits.append(x_init)
        
    # For each initialization, run L-BFGS
    for x0 in inits:
        if time.time() > deadline - 5.0:
            break
            
        # Objective function: max absolute autoconvolution
        # Note: L-BFGS minimizes a scalar. We use the max abs autocorr.
        # This function is non-smooth (max of absolutes).
        # We can use a soft-max or just rely on L-BFGS's robustness?
        # L-BFGS expects smooth functions. The max function is piecewise linear (non-differentiable at 0).
        # A common trick is to use a log-sum-exp or similar smooth approximation, 
        # but for max abs, we can just use the raw max and hope L-BFGS handles the kinks,
        # or use a penalty method.
        # Given the time constraint, let's try the raw max first.
        
        def objective(x):
            return _fast_autocorr_max_abs(x)
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
            
            x_opt = res.x
            val_opt = objective(x_opt)
            
            # Threshold to +/-1
            # We want to map the continuous signal to a signed step function.
            # Simple threshold: sign(x)
            x_quant = np.sign(x_opt)
            # Handle zeros (unlikely with random init, but possible)
            x_quant[x_quant == 0] = 1
            
            # Evaluate the quantized signal
            val_quant = _fast_autocorr_max_abs(x_quant)
            
            if val_quant < best_val:
                best_val = val_quant
                best_x = x_quant.copy()
                
        except Exception as e:
            # If optimization fails, try the initial quantized version
            x_quant = np.sign(x0)
            x_quant[x_quant == 0] = 1
            val_quant = _fast_autocorr_max_abs(x_quant)
            if val_quant < best_val:
                best_val = val_quant
                best_x = x_quant.copy()

    # Final local search refinement on the best quantized solution
    # Use a simple random restart hill climbing on the +/-1 sequence
    current_x = best_x.copy()
    current_val = best_val
    
    # Time remaining for local search
    time_left = deadline - time.time()
    if time_left > 5.0:
        # Simple random flips
        steps = int(time_left * 50)  # Estimate steps
        for _ in range(steps):
            if time.time() > deadline - 1.0:
                break
                
            i = rng.integers(0, N)
            current_x[i] *= -1
            new_val = _fast_autocorr_max_abs(current_x)
            
            if new_val < current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x.copy()
            else:
                current_x[i] *= -1  # Revert

    return best_x.astype(np.int8)
# EVOLVE-BLOCK-END
