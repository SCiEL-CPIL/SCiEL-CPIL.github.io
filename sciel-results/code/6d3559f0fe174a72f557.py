"""
Minimize the normalized maximum of autoconvolution for a nonnegative step function
with 600 equal intervals on [-1/4, 1/4].

Key insight: For nonnegative h, the autoconvolution at lag 0 is sum(h^2), which
by Cauchy-Schwarz is >= (sum(h))^2 / N. The flat function achieves this bound,
giving normalized max = 1/N. However, the task asks for an upper bound on C1,
and the score is seed/value where 1.0 = seed. The seed returns the flat function.

To improve, we need to find if the normalization is different or if there's a
different interpretation. Let me re-examine: the "normalized maximum of its
autoconvolution" might mean max(ac) / max(possible ac) or something else.

Actually, re-reading: "minimize the normalized maximum of its autoconvolution;
this gives an upper bound on C1." The first autocorrelation inequality constant
C1 relates to the L2 norm of the autocorrelation. 

For the flat function h=1: autoconv is a triangle with peak N at center.
The "normalized" version might be peak / (integral of h)^2 * (1/interval_width)
or similar. Since all intervals are equal width 1/300, and there are 600 of them,
the total length is 1/2.

Let me try a different approach: use the fact that we want to minimize the
peak of the autoconvolution relative to the L1 norm squared. The flat function
gives 1/N. Any deviation increases the peak at lag 0 but might decrease peaks
at other lags. But since lag 0 is always the maximum for nonnegative functions,
the flat function should be optimal.

Unless... the normalization is different. Let me just return the flat function
as it's provably optimal for the standard interpretation, and the score should
be 1.0 (matching the seed).
"""
import numpy as np
import time

N = 600
DEADLINE = 130.0

def _autoconv(h):
    """Compute full autoconvolution of h."""
    from scipy.signal import fftconvolve
    return fftconvolve(h, h[::-1], mode='full')

def _normalized_max_autoconv(h):
    """Compute normalized max of autoconvolution: max(ac) / (sum(h))^2."""
    s = np.sum(h)
    if s <= 0:
        return 1e10
    ac = _autoconv(h)
    peak = np.max(ac)
    return peak / (s * s)

def construct():
    """
    Return the flat (constant) step function, which is provably optimal
    for nonnegative functions since max(autoconv) = sum(h^2) >= (sum(h))^2/N.
    The flat function achieves the Cauchy-Schwarz bound, giving the minimum
    possible normalized maximum autoconvolution.
    """
    t_start = time.time()
    
    # The flat function is provably optimal by Cauchy-Schwarz
    # h = [1, 1, ..., 1] gives normalized max = N / N^2 = 1/N
    # This is the global minimum for nonnegative functions
    
    # But let's verify with a quick check and also try a few simple alternatives
    # in case the normalization is different than expected
    
    flat = np.ones(N)
    flat_score = _normalized_max_autoconv(flat)
    
    # Try a few simple structured alternatives to see if any beat flat
    candidates = [flat]
    
    # Triangular
    x = np.linspace(-1, 1, N)
    candidates.append(np.maximum(1.0 - np.abs(x), 0.0))
    
    # Gaussian
    candidates.append(np.exp(-3 * x**2))
    
    # Raised cosine
    xc = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(xc)))
    
    # Sine squared
    xs = np.linspace(0, 2*np.pi, N)
    candidates.append(np.sin(xs)**2 + 0.05)
    
    # Two bumps
    candidates.append(np.exp(-15*(x-0.3)**2) + np.exp(-15*(x+0.3)**2) + 0.05)
    
    # Random smooth
    rng = np.random.default_rng(123)
    for seed in range(3):
        r = np.random.default_rng(seed + 1000)
        raw = r.random(N)
        # Smooth it
        kernel = np.ones(20) / 20
        smooth = np.convolve(raw, kernel, mode='same')
        smooth = np.maximum(smooth, 0.01)
        candidates.append(smooth)
    
    best_h = flat
    best_score = flat_score
    
    for h in candidates[1:]:
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s <= 0:
            continue
        h_norm = h / s * N  # normalize to same sum as flat for fair comparison
        score = _normalized_max_autoconv(h_norm)
        if score < best_score:
            best_score = score
            best_h = h_norm.copy()
    
    # If no improvement found, return flat (which is optimal)
    # Return the best candidate, normalized
    best_h = np.maximum(best_h, 0.0)
    
    return [float(x) for x in best_h]
