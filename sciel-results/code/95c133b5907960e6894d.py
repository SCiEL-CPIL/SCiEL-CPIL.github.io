"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600

# EVOLVE-BLOCK-START

def _autoconv_max_normalized(heights):
    """
    Compute the normalized maximum of the autoconvolution of the step function.
    
    The step function has N equal intervals on [-1/4, 1/4].
    Interval width: dx = (1/2) / N = 1/(2N)
    
    Autoconvolution at shift k (in units of dx):
      (f * f)(k*dx) = sum_j f_j * f_{j+k} * dx^2
    
    But we need to be careful about the definition. The autoconvolution of a step function
    defined on [-1/4, 1/4] with N intervals:
      f(x) = h_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*dx, dx = 1/(2N)
    
    The autoconvolution (f*f)(y) = integral f(x) f(y-x) dx
    
    For a step function, this is a piecewise linear function. But we're asked to minimize
    the "normalized maximum of its autoconvolution".
    
    Let's think about what "normalized" means. The score is an upper bound on C1.
    The first autocorrelation inequality relates to C1.
    
    Actually, let me reconsider. The autoconvolution of a step function with N steps
    on [-1/4, 1/4]:
    
    Let f be the step function. (f*f)(t) = ∫ f(x) f(t-x) dx.
    
    For a step function with values h_0, ..., h_{N-1} on intervals of width dx,
    the autoconvolution is a piecewise linear function that can be computed efficiently.
    
    However, for the purpose of optimization, we can approximate the maximum of the
    autoconvolution by evaluating it at a grid of points, or we can compute it exactly.
    
    The autoconvolution of two step functions (same function here) with N steps each
    on a domain of length L = 1/2 will have support on [-L, L] = [-1/2, 1/2].
    
    The autoconvolution at shift m*dx (where m is an integer from -N to N) can be computed.
    
    Actually, let's use FFT for efficiency. The autoconvolution can be computed via:
    1. Zero-pad the signal to length 2N
    2. FFT, square the magnitude, inverse FFT
    3. The result gives the discrete autoconvolution
    
    But we need to be careful about the continuous vs discrete distinction.
    
    Let dx = 1/(2N). The step function has values h_i on [x_i, x_{i+1}).
    
    The autoconvolution (f*f)(t) for t = m*dx:
    This is the integral over x of f(x) f(m*dx - x) dx.
    
    For a step function, this integral can be computed by summing over pairs of intervals.
    
    Let's use the FFT approach. If we treat the step function as a sequence h_0, ..., h_{N-1},
    the discrete autoconvolution c_k = sum_j h_j h_{j+k} (with appropriate handling of boundaries)
    approximates (1/dx) * (f*f)(k*dx) ... actually, let me think more carefully.
    
    (f*f)(t) = ∫ f(x) f(t-x) dx
    
    If f is a step function with f(x) = h_i for x in [x_i, x_{i+1}), then:
    (f*f)(t) = sum_{i,j} h_i h_j * measure({x : x in [x_i, x_{i+1}) and t-x in [x_j, x_{j+1})})
    
    The measure is the length of the intersection of [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j).
    
    This is complex to compute directly. Let's use FFT instead.
    
    For a step function on N intervals of width dx, the autoconvolution at integer multiples
    of dx can be approximated by the discrete autoconvolution of the sequence h.
    
    Specifically, (f*f)(k*dx) ≈ dx * sum_j h_j h_{j+k} for k = 0, 1, ..., N-1
    (with h_j = 0 outside the range)
    
    Wait, let me be more precise. If we define the discrete sequence h_j for j=0,...,N-1,
    then the discrete autoconvolution is:
    c_k = sum_j h_j h_{j+k} for k >= 0 (with h_j = 0 if j < 0 or j >= N)
    
    The continuous autoconvolution at t = k*dx is approximately:
    (f*f)(k*dx) ≈ dx * c_k
    
    But actually, the continuous autoconvolution is a piecewise linear function, and its
    maximum may occur at non-integer multiples of dx. However, for optimization purposes,
    we can use the discrete autoconvolution as a proxy, or we can compute the exact maximum.
    
    Let me use FFT to compute the discrete autoconvolution efficiently.
    
    The "normalized" maximum - I think this means we normalize by something. Let's assume
    the normalization is such that the integral of f is 1, or the maximum is 1, or something.
    
    Actually, looking at the task again: "minimize the normalized maximum of its autoconvolution".
    
    Let's assume the normalization is: max((f*f)(t)) / (integral f)^2 or something similar.
    
    Actually, for the first autocorrelation inequality, C1 is related to the maximum of the
    autocorrelation. Let me just compute the maximum of the autoconvolution and see.
    
    Let's use the FFT approach:
    1. Create a zero-padded array of length 2N
    2. Compute FFT
    3. Square the magnitude
    4. Inverse FFT
    5. The result is the circular autoconvolution, but with zero-padding, it's the linear autoconvolution
    
    The linear autoconvolution of h with itself has length 2N-1.
    c_k = sum_j h_j h_{j+k} for k = -(N-1), ..., N-1
    
    Using FFT:
    - Pad h to length 2N
    - FFT, square, inverse FFT
    - The result gives the circular convolution, which equals the linear convolution due to zero-padding
    
    Let me implement this.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    # Zero-pad to length 2n for linear convolution via FFT
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    # FFT
    F = np.fft.fft(padded)
    # Autoconvolution: |F|^2
    G = F * np.conj(F)
    # Inverse FFT
    conv = np.fft.ifft(G).real
    # The autoconvolution is conv[0:n] + conv[n:2n-1] reversed... 
    # Actually, the linear autoconvolution of length n with itself has length 2n-1.
    # The FFT gives us the circular convolution of length 2n.
    # The first n elements are conv[0], conv[1], ..., conv[n-1]
    # The next n-1 elements are conv[n], conv[n+1], ..., conv[2n-2]
    # But due to circularity, conv[2n-1] = 0 (since we zero-padded)
    
    # The linear autoconvolution c_k for k=0 to 2n-2 is:
    # c_k = conv[k] for k=0 to n-1
    # c_k = conv[k] for k=n to 2n-2 (but these correspond to negative lags)
    
    # Actually, the autoconvolution is symmetric: c_k = c_{-k}
    # The maximum will be at k=0 or near it.
    
    # Let's just take the maximum of the entire conv array (excluding the wrap-around)
    # The linear convolution is in conv[0:2n-1]
    max_conv = np.max(conv[:2*n-1])
    
    # Normalize: I'll assume the normalization is by the square of the sum (integral)
    total = np.sum(h)
    if total == 0:
        return 0.0
    # The integral of f is total * dx where dx = 1/(2n)
    # The autoconvolution at 0 is approximately dx * c_0
    # Normalized max = max((f*f)(t)) / (integral f)^2
    # = max_conv * dx / (total * dx)^2
    # = max_conv / (total^2 * dx)
    
    dx = 1.0 / (2 * n)
    normalized_max = max_conv / (total * total * dx)
    
    return normalized_max

def _objective(params):
    """
    Objective function for differential evolution.
    
    params: array of parameters
    - params[0]: envelope amplitude
    - params[1:]: modulation phases and amplitudes
    
    We represent the step function as:
    h_i = envelope(i) * (1 + sum_k a_k cos(2*pi*k*i/N + phi_k))
    
    where envelope is a raised cosine:
    envelope(i) = 0.5 * (1 + cos(pi * i / (N-1)))
    
    This ensures non-negativity if the modulation is bounded.
    """
    n = N
    t = np.arange(n, dtype=np.float64)
    
    # Raised cosine envelope
    envelope = 0.5 * (1.0 + np.cos(np.pi * t / (n - 1)))
    
    # Modulation: sum of a few harmonics
    # params[0] = base amplitude
    # params[1:4] = amplitude and phase for harmonic 1
    # params[4:7] = amplitude and phase for harmonic 2
    # params[7:10] = amplitude and phase for harmonic 3
    
    base = params[0]
    
    # Harmonic 1
    a1 = params[1]
    phi1 = params[2]
    # Harmonic 2
    a2 = params[3]
    phi2 = params[4]
    # Harmonic 3
    a3 = params[5]
    phi3 = params[6]
    
    mod = 1.0 + a1 * np.cos(2 * np.pi * t / n + phi1) \
              + a2 * np.cos(4 * np.pi * t / n + phi2) \
              + a3 * np.cos(6 * np.pi * t / n + phi3)
    
    # Ensure non-negativity
    h = envelope * base * mod
    h = np.maximum(h, 0.0)
    
    return _autoconv_max_normalized(h)

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    # Initial best: uniform
    best_heights = [1.0] * N
    best_score = _autoconv_max_normalized(best_heights)
    
    # Try a few simple analytic shapes first
    t = np.arange(N, dtype=np.float64)
    
    # Raised cosine
    h_cos = 0.5 * (1.0 + np.cos(np.pi * t / (N - 1)))
    score_cos = _autoconv_max_normalized(h_cos)
    if score_cos < best_score:
        best_score = score_cos
        best_heights = list(h_cos)
    
    # Gaussian-like
    center = N / 2.0
    sigma = N / 4.0
    h_gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    score_gauss = _autoconv_max_normalized(h_gauss)
    if score_gauss < best_score:
        best_score = score_gauss
        best_heights = list(h_gauss)
    
    # Check time
    if time.time() > deadline - 30.0:
        return best_heights
    
    # Use differential evolution to optimize the modulation parameters
    # Parameters: base, a1, phi1, a2, phi2, a3, phi3
    # Bounds:
    # base: [0, 2]
    # a1, a2, a3: [0, 0.5] (to ensure non-negativity with the envelope)
    # phi1, phi2, phi3: [0, 2*pi]
    
    bounds = [
        (0.0, 2.0),   # base
        (0.0, 0.5),   # a1
        (0.0, 2 * np.pi),  # phi1
        (0.0, 0.5),   # a2
        (0.0, 2 * np.pi),  # phi2
        (0.0, 0.5),   # a3
        (0.0, 2 * np.pi),  # phi3
    ]
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.success or result.fun < best_score:
            params = result.x
            n = N
            t = np.arange(n, dtype=np.float64)
            envelope = 0.5 * (1.0 + np.cos(np.pi * t / (n - 1)))
            base = params[0]
            a1 = params[1]
            phi1 = params[2]
            a2 = params[3]
            phi2 = params[4]
            a3 = params[5]
            phi3 = params[6]
            mod = 1.0 + a1 * np.cos(2 * np.pi * t / n + phi1) \
                      + a2 * np.cos(4 * np.pi * t / n + phi2) \
                      + a3 * np.cos(6 * np.pi * t / n + phi3)
            h = envelope * base * mod
            h = np.maximum(h, 0.0)
            score = _autoconv_max_normalized(h)
            if score < best_score:
                best_score = score
                best_heights = list(h)
    except Exception:
        pass
    
    return best_heights

# EVOLVE-BLOCK-END
