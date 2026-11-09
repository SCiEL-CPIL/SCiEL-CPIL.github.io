"""
Optimized program for the first autocorrelation inequality task.

Strategy:
1. We want to minimize the maximum of the autoconvolution of a nonnegative step function
   with 600 equal intervals on [-1/4, 1/4].
2. The autoconvolution in the time domain corresponds to the squared magnitude of the FFT
   in the frequency domain (Parseval's theorem / Wiener-Khinchin).
3. Specifically, if f is the step function (heights h_i), then the autoconvolution c_k = sum_i h_i h_{i-k}.
   The maximum of c_k is what we want to minimize (normalized).
4. The FFT of the autoconvolution is |FFT(h)|^2. So minimizing the peak of the time-domain
   autoconvolution is related to making the spectral density |FFT(h)|^2 as flat as possible
   (white noise spectrum) while keeping h_i >= 0.

Approach:
- Parameterize h via a non-negative spectral amplitude.
- Use gradient descent on the spectral amplitudes to minimize the peak autoconvolution.
- Enforce non-negativity of h via softplus/ReLU in the time domain after inverse FFT.

However, a simpler and effective approach for this specific problem:
- The optimal function for minimizing the peak of the autoconvolution (while being non-negative)
  is known to be related to a "bang-bang" or thresholded version of a low-pass filter.
- Alternatively, we can use iterative projection: start with a candidate, compute autoconvolution,
  and adjust heights to reduce the peak.

Given the 120s time limit, we'll use a gradient-based approach in the frequency domain:
- Let H = FFT(h). We want to choose H such that |H|^2 is as flat as possible (to minimize peak
  autoconvolution) while h = IFFT(H) >= 0.
- We parameterize H = A * exp(i*theta) where A >= 0 is the amplitude.
- We minimize max_k (IFFT(|H|^2))_k subject to IFFT(H) >= 0.

This is complex. A simpler heuristic that works well:
- The uniform function (all 1s) has autoconvolution that is a triangle with peak N at center.
- Normalized peak = 1 (since sum h_i^2 = N, and peak autoconv = N, so normalized = N/N = 1).
- To do better, we need a function whose autoconvolution is flatter.
- A random non-negative function will have autoconvolution that is roughly constant away from
  the peak, but the peak will be higher than average.
- The optimal is known to be close to a thresholded Gaussian or a specific low-pass shape.

Let's use a practical approach:
1. Start with a candidate that is a smoothed version of a low-pass filter.
2. Use projected gradient descent: minimize peak autoconvolution, project onto h >= 0.

But given time constraints, let's try a simpler idea:
- The normalized maximum autoconvolution is max_k c_k / sum_i h_i^2.
- For the uniform function, this is 1.
- We can try to find a function where the autoconvolution is more uniform.
- One known result: the "flat-top" window or similar has a flatter autoconvolution.

Let's implement a gradient descent in the time domain:
- Objective: minimize max_k c_k where c = autoconv(h)
- Constraint: h_i >= 0
- Use subgradient method for the max, and projection for non-negativity.

Since N=600, FFT-based autoconvolution is O(N log N).
We'll do ~1000 iterations of gradient descent.
"""

import numpy as np

N = 600

def autoconvolve(h):
    """Compute autoconvolution of h using FFT."""
    n = len(h)
    # Pad to avoid circular convolution
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H.conj(), n=size)
    return C[:2*n-1]  # Linear autoconvolution

def construct():
    # Initialize with a smooth, non-negative function
    # Use a raised cosine or similar to start
    x = np.linspace(-0.5, 0.5, N, endpoint=False)
    h = 1.0 + 0.5 * np.cos(2 * np.pi * x)  # Non-negative, smooth
    
    # Normalize so that sum h_i^2 = 1 (for consistent scaling)
    h = h / np.sqrt(np.sum(h**2))
    
    # Gradient descent to minimize peak autoconvolution
    # We'll use a simple subgradient approach
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros(N)
    
    best_h = h.copy()
    best_score = np.inf
    
    for it in range(500):
        # Compute autoconvolution
        c = autoconvolve(h)
        peak = np.max(c)
        
        # Normalize: score = peak / sum(h^2)
        norm = np.sum(h**2)
        score = peak / norm if norm > 0 else np.inf
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Subgradient of peak autoconvolution w.r.t. h
        # The peak is at index k*. The subgradient is the derivative of c_{k*} w.r.t. h.
        k_star = np.argmax(c)
        
        # c_k = sum_i h_i h_{i-k}
        # d(c_k)/d(h_j) = h_{j-k} + h_{j+k} (with appropriate bounds)
        # This is the k-th element of the autoconvolution kernel
        
        # More efficiently: the gradient of c_k w.r.t. h is the shifted h
        # Specifically, if c = conv(h, h), then d(c_k)/d(h_j) = h_{j-k} + h_{k-j}
        
        # We can compute this as: grad = shift(h, -k_star) + shift(h, k_star)
        # But we need to be careful with boundaries
        
        # Simpler: use the fact that the gradient of the autoconvolution at lag k
        # is h shifted by k, plus h shifted by -k
        
        # Let's compute the gradient numerically or use the FFT relationship
        # Actually, for the max, we only need the subgradient at the argmax
        
        # The subgradient of max(c) is the gradient of c_{k*}
        # c_{k*} = sum_i h_i * h_{i-k*}
        # d(c_{k*})/d(h_j) = h_{j-k*} + h_{j+k*} (with h_out_of_bounds = 0)
        
        grad = np.zeros(N)
        for j in range(N):
            val = 0.0
            if j - k_star >= 0 and j - k_star < N:
                val += h[j - k_star]
            if j + k_star >= 0 and j + k_star < N:
                val += h[j + k_star]
            grad[j] = val
        
        # Momentum update
        v = momentum * v - lr * grad
        h_new = h + v
        
        # Project onto non-negative orthant
        h_new = np.maximum(h_new, 0.0)
        
        # Renormalize to keep sum(h^2) constant (optional, but helps stability)
        norm_new = np.sum(h_new**2)
        if norm_new > 0:
            h_new = h_new * np.sqrt(norm / norm_new) if norm > 0 else h_new
        
        h = h_new
    
    # Scale back: the score is invariant to scaling, so we can return as is
    # But the problem says "nonnegative step heights", so any scale is fine
    # The normalized max autoconvolution is scale-invariant
    
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
