"""
Minimize the normalized maximum of the autoconvolution of a nonnegative step
function with 600 equal intervals on [-1/4, 1/4].

Design: use a symmetric "double-box" (hollow) structure:
  h(x) = A * 1_{|x| in [a, b]} + B * 1_{|x| in [c, d]}
with 0 <= a < b < c < d <= 1/4.
The gap near the origin and/or between the two annuli causes the central peak
of the autoconvolution to be suppressed relative to the support of the
function, lowering the normalized max.

We optimize the geometric parameters (a,b,c,d) and amplitudes (A,B) via
grid + local refinement, with a smooth max (log-sum-exp) surrogate for the
nonsmooth max, plus a final exact evaluation.

Score = normalized max = max_k (conv_k) / (sum h)  ... but the task says
"normalized maximum of its autoconvolution". The seed program returns [1.0]*600
and scores 1.0 (the seed value). The score reported is seed/value, so the raw
value for the seed is 1.0. We need to figure out the normalization.

For h = 1 on [-1/4,1/4] with 600 intervals, each interval has width w = (1/2)/600.
Autoconvolution: (h*h)(t) = integral h(t-s)h(s) ds. For the constant function
on the full interval, the autoconvolution is a triangle with peak at t=0 of
height = length = 1/2. The "normalized maximum" likely is max(conv) / (integral h)
or max(conv) / max(h) / something. Let's think: if the raw value for the
constant-1 function is 1.0, and the autoconvolution peak is 1/2, then the
normalization must be such that (1/2) / norm = 1.0, so norm = 1/2.

Alternatively, the normalization might be max(conv) / (max(h) * width * N) or
similar. Let me just compute: for h=1 everywhere, conv at 0 = sum of h_i * w for
all i where the overlap is full... Actually for a step function with N intervals
of width w, the autoconvolution at lag 0 is sum_i h_i^2 * w = sum_i h_i * w (since
h_i=1) = N * w = 1/2. At other lags it's less.

So normalized max = max(conv) / (1/2) = 1.0 for the constant function. That makes
sense: normalize by the autoconvolution value at lag 0 of the constant function,
or equivalently by the L1 norm or the integral.

Actually, a common normalization: normalized max = max_t (h*h)(t) / (∫h)^2 / (1/width)
or something. Let me just use: the evaluator computes the autoconvolution of the
step function (treating it as a piecewise-constant function on [-1/4,1/4]),
finds its maximum, and divides by some normalization that gives 1.0 for the
all-ones function.

The most natural: normalized_max = max(conv) / conv_0 where conv_0 is the
autoconvolution at 0... but that would always be >= 1 and for constant it's 1.
Wait, for a non-constant function, conv_0 = sum h_i^2 * w, while max(conv) could
be at a different lag. Hmm.

Let me just go with: the score is max(conv) / (sum(h) * w) where w is interval
width... no. Let me think differently.

For the all-ones: sum(h) = 600, w = 1/1200. sum(h)*w = 600/1200 = 1/2. max(conv) = 1/2.
So max(conv)/(sum(h)*w) = 1.0. That works!

So: normalized_max = max(conv) / (sum(h_i) * w) where w = (1/2)/600.

Let me verify: for a function with some zeros, sum(h)*w is smaller, and max(conv)
is also smaller. The ratio is what we minimize.

I'll implement the autoconvolution computation and the optimization.
"""

import numpy as np
import time

N = 600
W = (0.5) / N  # interval width = 1/1200
DEADLINE = 150.0  # seconds
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _autoconv_max_normalized(h):
    """Compute max(autoconvolution(h)) / (sum(h)*W).
    h is an array of N nonnegative step heights.
    The autoconvolution is computed on a grid of lags.
    """
    # h is piecewise constant on N intervals of width W.
    # autoconv at lag k (in units of W): sum_i h[i] * h[i+k] * W for valid i
    # We need to find the max over all real lags, but for step functions the
    # autoconvolution is piecewise linear, so the max occurs at a lag that is
    # an integer multiple of W (i.e., at the interval boundaries).
    # Actually, the autoconvolution of two step functions is a piecewise linear
    # function that is convex on each interval of lags, so the max is at a
    # breakpoint, which is at integer multiples of W.
    
    h = np.asarray(h, dtype=np.float64)
    # Compute autoconvolution at lags 0, W, 2W, ..., (N-1)W and also at the
    # "negative" lags (symmetric). By symmetry, we only need lags 0..(N-1)W.
    # conv[k] = W * sum_{i=0}^{N-1-k} h[i] * h[i+k]
    
    # Use FFT for efficiency
    # Pad to at least 2N-1
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    H = np.fft.rfft(h, n=nfft)
    conv = np.fft.irfft(H * H, n=nfft)
    # conv[k] for k=0..N-1 gives W * sum h[i]*h[i+k]... 
    # Actually FFT gives circular convolution. We need linear convolution.
    # Since we padded to 2N, the first N values are the linear autoconvolution
    # (with the W factor).
    # conv[k] = sum_{i} h[i] * h[(i+k) mod N] ... no, with zero padding it's linear.
    # With zero padding to 2N, conv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k] for k=0..N-1
    # and the actual autoconvolution value at lag k*W is conv[k] * W.
    
    conv_vals = conv[:N] * W
    max_conv = np.max(conv_vals)
    
    total = np.sum(h) * W
    if total < 1e-30:
        return float('inf')
    return max_conv / total

def _make_double_box(a, b, c, d, A, B):
    """Create a step function with double-box structure.
    h(x) = A for |x| in [a, b], B for |x| in [c, d], 0 otherwise.
    x is in [-1/4, 1/4], discretized into N intervals.
    """
    h = np.zeros(N)
    x_edges = np.linspace(-0.25, 0.25, N + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    
    # Inner annulus: a <= |x| <= b
    mask_inner = (np.abs(x_centers) >= a) & (np.abs(x_centers) <= b)
    h[mask_inner] = A
    
    # Outer annulus: c <= |x| <= d
    mask_outer = (np.abs(x_centers) >= c) & (np.abs(x_centers) <= d)
    h[mask_outer] = B
    
    return h

def _optimize_double_box():
    """Optimize the double-box parameters."""
    best_h = None
    best_score = float('inf')
    
    # Parameter ranges (in units of 1/4, i.e., x in [-0.25, 0.25])
    # a in [0, 0.25), b in (a, 0.25), c in (b, 0.25), d in (c, 0.25]
    # A, B > 0
    
    # Coarse grid
    a_vals = np.linspace(0.01, 0.12, 12)
    b_vals = np.linspace(0.03, 0.18, 12)
    c_vals = np.linspace(0.05, 0.22, 12)
    d_vals = np.linspace(0.08, 0.25, 12)
    A_vals = np.linspace(0.5, 2.0, 6)
    B_vals = np.linspace(0.5, 2.0, 6)
    
    # For each geometry, optimize amplitudes analytically or by small grid
    # Since the objective is homogeneous in (A, B) in a certain sense... 
    # Actually, normalized_max = max(conv) / (sum(h)*W). 
    # conv is quadratic in (A,B), sum is linear in (A,B).
    # So the ratio is homogeneous of degree 1 in (A,B). We can fix one and
    # optimize the other, or fix the ratio.
    
    # Let's fix A = 1 and optimize B, then scale.
    # Or just do a joint grid.
    
    count = 0
    for a in a_vals:
        for b in b_vals:
            if b <= a:
                continue
            for c in c_vals:
                if c <= b:
                    continue
                for d in d_vals:
                    if d <= c:
                        continue
                    # For this geometry, try a few amplitude pairs
                    for A in A_vals:
                        for B in B_vals:
                            h = _make_double_box(a, b, c, d, A, B)
                            if np.all(h == 0):
                                continue
                            score = _autoconv_max_normalized(h)
                            if score < best_score:
                                best_score = score
                                best_h = h.copy()
                            count += 1
                            if _elapsed() > DEADLINE - 10:
                                return best_h, best_score
                    if _elapsed() > DEADLINE - 10:
                        return best_h, best_score
                if _elapsed() > DEADLINE - 10:
                    return best_h, best_score
            if _elapsed() > DEADLINE - 10:
                return best_h, best_score
        if _elapsed() > DEADLINE - 10:
            return best_h, best_score
    
    return best_h, best_score

def _refine(h, max_iter=50):
    """Local refinement using coordinate descent on the step heights."""
    h = h.copy().astype(np.float64)
    best_score = _autoconv_max_normalized(h)
    
    for it in range(max_iter):
        if _elapsed() > DEADLINE - 5:
            break
        improved = False
        # Try small perturbations on random indices
        n_try = 20
        for _ in range(n_try):
            idx = np.random.randint(0, N)
            old_val = h[idx]
            # Try a few new values
            for new_val in [0.0, old_val * 0.5, old_val * 1.5, 1.0]:
                if new_val < 0:
                    continue
                h[idx] = new_val
                score = _autoconv_max_normalized(h)
                if score < best_score - 1e-12:
                    best_score = score
                    improved = True
                else:
                    h[idx] = old_val
                    break
            if _elapsed() > DEADLINE - 5:
                return h, best_score
        if not improved:
            break
    
    return h, best_score

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Start with the double-box optimization
    h, score = _optimize_double_box()
    
    if h is None:
        # Fallback: constant function
        return [1.0] * N
    
    # Local refinement
    h, score = _refine(h)
    
    # Ensure nonnegative
    h = np.maximum(h, 0.0)
    
    # Final check
    final_score = _autoconv_max_normalized(h)
    
    return [float(x) for x in h]

# Precompute for speed
np.random.seed(42)
