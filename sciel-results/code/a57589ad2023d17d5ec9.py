"""
Spectral shaping approach for minimizing the normalized maximum of autoconvolution.
Instead of optimizing 600 time-domain heights directly, we parameterize the function
by its frequency domain power spectrum. The step heights are the inverse DFT of the
square root of the power spectrum (since autoconvolution peak relates to spectral energy).

We use a parametric family for the power spectrum, such as a raised cosine or Gaussian
in frequency, which allows us to control the "flatness" of the spectrum and thus the
peak of the autoconvolution.
"""

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
dx = 1.0 / 1200.0

def _compute_autoconv_max(heights, dx):
    """
    Compute the maximum of the autoconvolution of the step function.
    We normalize the function such that the integral is 1.
    Objective: max_x (f*f)(x)
    """
    heights = np.asarray(heights, dtype=float)
    # Normalize so that integral f = 1
    total_area = np.sum(heights) * dx
    if total_area < 1e-12:
        return 1e10
    h_norm = heights / total_area
    
    # Compute autoconvolution via FFT
    # The autoconvolution of a discrete signal h is the IFFT of |FFT(h)|^2
    # But we need the continuous approximation.
    # For a step function, the autoconvolution at lag k (in samples) is:
    # c[k] = sum_i h[i] h[i+k] dx
    # We can compute this efficiently using FFT.
    
    n_fft = 2 * N
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h_norm
    
    H = np.fft.fft(h_padded, n=n_fft)
    # Autoconvolution in frequency domain is H * conj(H) = |H|^2
    # Inverse FFT gives the circular autoconvolution.
    # Since we padded to 2N, and N is the support, the linear autoconvolution
    # is contained in the first 2N-1 samples (or centered).
    # Actually, for autoconvolution of a real signal, the result is real and symmetric.
    # The peak is at lag 0.
    conv = np.fft.ifft(H * H.conjugate()).real
    # The autoconvolution values are at lags 0, 1, ..., N-1 and -(N-1), ..., -1.
    # In the FFT output, index 0 is lag 0, index 1 is lag 1, ..., index N-1 is lag N-1.
    # Index N is lag -N+1 (due to circular shift), etc.
    # We only care about the valid lags where both functions overlap.
    # The max is typically at lag 0 for non-negative functions.
    # Let's take the max over the valid range [0, N-1].
    return np.max(conv[:N]) * dx  # Multiply by dx to get the integral scale

def _objective_spectral(params, dx):
    """
    Objective function for spectral parameters.
    params: [width, center, amplitude] or similar.
    We model the power spectrum S(k) = A * exp(-(k-center)^2 / width^2)
    The heights h are the inverse DFT of sqrt(S(k)).
    """
    width, center, amplitude = params
    if width < 0.1 or amplitude < 0:
        return 1e10
        
    # Frequency bins for N-point DFT
    # We use the first N/2 bins and mirror for symmetry to ensure real heights.
    k = np.arange(N)
    # Shift center to be in the middle of the spectrum
    # The DFT frequency axis for real signals is symmetric around 0.
    # Let's define the spectrum on the full DFT grid.
    # To ensure real heights, S(k) must be symmetric: S(k) = S(N-k).
    # Let's define S(k) for k in [0, N/2] and mirror.
    
    # Simplified: Use a raised cosine spectrum which is symmetric and non-negative.
    # S(k) = amplitude * (0.5 + 0.5 * cos(2 * pi * width * k / N))
    # This ensures non-negativity and symmetry.
    # The parameter 'width' controls the bandwidth.
    
    # Let's try a Gaussian in frequency, centered at 0 (DC) and N/2 (Nyquist) for symmetry?
    # Actually, for a real time-domain signal, the spectrum is symmetric.
    # Let's just define the magnitude spectrum |H(k)| directly.
    # |H(k)| = amplitude * exp(-(k - N/2)^2 / (2 * width^2)) + amplitude * exp(-(k)^2 / (2 * width^2))
    # This creates two peaks at DC and Nyquist, ensuring symmetry and real IFFT.
    
    k = np.arange(N)
    # Gaussian centered at 0 and N (which is 0) and N/2 (Nyquist)
    # Actually, to ensure real output, we can just take the IFFT of a symmetric magnitude.
    # Let's use a simple parametric form:
    # |H(k)| = amplitude * (1 + beta * cos(2 * pi * k / N))
    # This is a "flat-top" or "raised cosine" spectrum.
    # It has 3 parameters: amplitude, beta, and maybe a shift.
    # Let's stick to 2 parameters: amplitude and beta (bandwidth control).
    
    # Let's use a more flexible form:
    # |H(k)| = amplitude * exp(-alpha * (1 - cos(2 * pi * k / N)))
    # This is a von Mises distribution in frequency, which is symmetric and non-negative.
    # alpha controls the peakiness. alpha=0 is flat (maximally spread in time, min peak in autoconv? No, flat spectrum -> white noise -> high peak? No, flat spectrum -> random phases -> but here we fix phase to 0 for simplicity? No, we need to optimize phase too?
    # The task asks for non-negative step heights. This is a strong constraint.
    # If we use IFFT of a real, non-negative spectrum, the heights are not necessarily non-negative.
    # We must project or constrain.
    
    # Alternative Strategy:
    # The "Two-Block" inspiration suggests that simple structures work well.
    # The current program uses generic optimization.
    # Let's try a **Genetic Algorithm** or **Differential Evolution** on a reduced set of parameters
    # that define a piecewise constant function with, say, 10 segments.
    # This is a different *design family* (piecewise constant with few segments) vs (600 independent variables).
    
    # Let's implement a piecewise constant function with M segments.
    # Parameters: [heights of M segments, widths of M segments]
    # Normalize widths to sum to N.
    # This reduces the dimensionality from 600 to ~20, allowing global optimization methods to work better.
    
    M = 10  # Number of segments
    # params: [h_0, ..., h_{M-1}, w_0, ..., w_{M-1}]
    # We normalize widths so they sum to 1.
    # We normalize heights so the integral is 1.
    
    h_seg = params[:M]
    w_seg = params[M:]
    
    # Ensure widths are positive
    w_seg = np.abs(w_seg) + 1e-6
    w_seg = w_seg / np.sum(w_seg)
    
    # Ensure heights are non-negative
    h_seg = np.maximum(h_seg, 0.0)
    
    # Expand to N steps
    heights = np.zeros(N)
    idx = 0
    for i in range(M):
        w_count = int(round(w_seg[i] * N))
        if idx + w_count > N:
            w_count = N - idx
        if w_count > 0:
            heights[idx:idx+w_count] = h_seg[i]
            idx += w_count
        if idx >= N:
            break
    
    # If we didn't fill all, fill remaining with last height
    if idx < N:
        heights[idx:] = h_seg[-1]
        
    return _compute_autoconv_max(heights, dx)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    M = 12  # Number of segments for piecewise constant approximation
    
    # Initial guess: uniform heights, uniform widths
    h0 = np.ones(M)
    w0 = np.ones(M)
    x0 = np.concatenate([h0, w0])
    
    best_x = x0.copy()
    best_val = _objective_spectral(best_x, dx)
    
    # Use Differential Evolution for global search
    # Bounds: heights [0, 5], widths [0.01, 1]
    bounds = [(0, 5)] * M + [(0.01, 1.0)] * M
    
    def wrapper(params):
        if time.time() > deadline:
            return best_val
        return _objective_spectral(params, dx)
    
    try:
        res = differential_evolution(
            wrapper,
            bounds,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            seed=42,
            polish=True
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Local refinement with Nelder-Mead on the best solution
    if time.time() < deadline - 10:
        try:
            res2 = minimize(
                wrapper,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception as e:
            print(f"NM failed: {e}")
            
    # Reconstruct the full 600-step heights
    h_seg = best_x[:M]
    w_seg = best_x[M:]
    
    w_seg = np.abs(w_seg) + 1e-6
    w_seg = w_seg / np.sum(w_seg)
    
    heights = np.zeros(N)
    idx = 0
    for i in range(M):
        w_count = int(round(w_seg[i] * N))
        if idx + w_count > N:
            w_count = N - idx
        if w_count > 0:
            heights[idx:idx+w_count] = h_seg[i]
            idx += w_count
        if idx >= N:
            break
            
    if idx < N:
        heights[idx:] = h_seg[-1]
        
    # Final normalization to ensure integral is 1 (as per _compute_autoconv_max logic)
    total_area = np.sum(heights) * dx
    if total_area > 1e-12:
        heights = heights / total_area
        
    return list(heights)
