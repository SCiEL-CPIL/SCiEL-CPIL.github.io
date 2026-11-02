"""
Third autocorrelation inequality: choose a signed step function with 400 equal intervals on [-1/4, 1/4]
to improve an autoconvolution upper bound.

We model the step function as a binary sign vector derived from the zero crossings
of a frequency-modulated sine wave (chirp), optimizing over center frequency
and chirp rate to minimize the peak side-lobe of the autoconvolution.
"""

import numpy as np

N = 400

def _compute_peak_side_lobe(signs):
    """
    Compute the peak side-lobe magnitude of the aperiodic autocorrelation.
    This is a robust proxy for minimizing the autoconvolution upper bound,
    as it directly controls the worst-case correlation deviation.
    """
    signs = np.asarray(signs, dtype=np.float64)
    n = len(signs)
    
    # Compute aperiodic autocorrelation using FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    f = np.fft.rfft(signs, n=size)
    autocorr = np.fft.irfft(f * np.conj(f), n=size)[:2*n-1]
    
    # Exclude lag 0 (autocorr[0] and autocorr[2n-2] are the same, autocorr[n] is lag 0 in full circle? No)
    # autocorr[0] is R(0) = N
    # autocorr[1..n-1] are R(1)..R(n-1)
    # autocorr[n..2n-2] are R(-1)..R(-(n-1))
    # We want max abs of R(k) for k != 0.
    # Due to symmetry, we can just look at 1..n-1.
    if n > 1:
        side_lobes = np.abs(autocorr[1:n])
        peak = np.max(side_lobes)
    else:
        peak = 0.0
        
    return peak

def _local_optimize(signs, max_iters=5000, patience=100):
    """
    Local search by flipping individual bits to reduce peak side-lobe.
    Uses randomized order to escape local minima.
    """
    signs = np.array(signs, dtype=np.float64)
    n = len(signs)
    current_score = _compute_peak_side_lobe(signs)
    best_score = current_score
    best_signs = signs.copy()
    
    no_improvement_count = 0
    rng = np.random.RandomState(12345)
    
    for iter_num in range(max_iters):
        improved = False
        # Try positions in randomized order
        order = rng.permutation(n)
        for i in order:
            signs[i] = -signs[i]
            new_score = _compute_peak_side_lobe(signs)
            if new_score < best_score - 1e-9:
                best_score = new_score
                best_signs = signs.copy()
                improved = True
                no_improvement_count = 0
            else:
                signs[i] = -signs[i]  # revert
            
            if improved:
                break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > patience:
                break
        else:
            signs = best_signs.copy()
            
    return best_signs, best_score

def construct():
    """
    Construct a signed step function with N equal intervals.
    Returns a sequence of signed step heights (values in {-1, +1}).
    
    Strategy: Use low-frequency sine waves to create smooth sign blocks.
    We search over frequency and phase, then refine with local bit-flipping.
    """
    x = np.arange(N, dtype=np.float64)
    t = x / N
    
    best_signs = np.ones(N, dtype=np.float64)
    best_score = _compute_peak_side_lobe(best_signs)
    
    rng = np.random.RandomState(42)
    
    # Search over low-frequency sine waves
    # Focus on frequencies that create smooth blocks
    for _ in range(100):
        f = rng.uniform(0.5, 8.0)
        p = rng.uniform(0, 2 * np.pi)
        sine_vals = np.sin(2 * np.pi * (f * t + p / (2 * np.pi)))
        signs = np.where(sine_vals >= 0, 1.0, -1.0)
        
        score = _compute_peak_side_lobe(signs)
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()

    # Search over linear chirps
    # s(t) = sign(sin(2*pi*(f0*t + 0.5*f1*t^2)))
    for _ in range(50):
        f0 = rng.uniform(0.5, 4.0)
        f1 = rng.uniform(-5.0, 5.0)
        chirp_vals = np.sin(2 * np.pi * (f0 * t + 0.5 * f1 * t * t))
        signs = np.where(chirp_vals >= 0, 1.0, -1.0)
        
        score = _compute_peak_side_lobe(signs)
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()

    # Refine the best candidate found with local search
    if best_score < 1000:  # Avoid refining if it's terrible
        refined_signs, refined_score = _local_optimize(best_signs)
        if refined_score < best_score:
            best_signs = refined_signs
            best_score = refined_score
            
    return best_signs
