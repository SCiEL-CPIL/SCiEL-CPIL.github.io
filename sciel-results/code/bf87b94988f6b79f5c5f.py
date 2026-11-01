"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolve_step(heights, n=N):
    """Compute autoconvolution of a step function with n equal intervals."""
    # Each interval has width w = (1/4 - (-1/4)) / n = (1/2)/n
    w = 0.5 / n
    # heights[i] is the value on interval [i*w - 0.25, (i+1)*w - 0.25)
    # (f*f)(x) = integral of f(t) f(x-t) dt
    # For step functions, we can compute this by convolving the height array
    # The autoconvolution at discrete points can be computed via FFT or direct
    # For a step function, (f*f) at the midpoint of interval k is:
    # sum over i,j where the overlap of interval i and (interval j shifted by x_k) is nonzero
    # Simpler: use np.convolve on the heights with themselves, scaled by w^2
    # Actually, for piecewise constant f, (f*f)(x) can be computed by:
    # The convolution of two step functions. Let's use the fact that
    # if f = sum_i h_i * 1_{[a_i, a_i+w)}, then f*f = sum_{i,j} h_i h_j * (1_{[a_i,a_i+w)} * 1_{[a_j, a_j+w)})
    # Each pair gives a trapezoid. But for simplicity, let's sample at midpoints.
    
    # Use FFT-based convolution for efficiency
    # Pad heights to avoid circular convolution
    size = 2 * n
    h = np.zeros(size)
    h[:n] = heights
    # Convolve h with h
    conv = np.convolve(h, h, mode='full')
    # The result has length 2*n - 1 + ... actually np.convolve gives length 2*n-1 for two arrays of length n
    # But we padded to size, so conv has length 2*size - 1 = 4*n - 1
    # The autoconvolution (f*f)(x) for x in [-1/2, 1/2] corresponds to the middle part
    # Actually, let me think more carefully.
    
    # f(t) is defined on [-1/4, 1/4]. Let's shift to [0, 1/2] for convolution.
    # g(u) = f(u - 1/4) for u in [0, 1/2]
    # Then (f*f)(x) = (g*g)(x + 1/2) where g*g is convolution on [0, 1/2]
    # g is piecewise constant with n intervals of width w = 1/(2n)
    # g*g at point y is sum_i sum_j h_i h_j * length of intersection of [i*w, (i+1)*w] and [y - j*w, y - (j+1)*w]
    
    # For discrete approximation, sample (f*f) at the midpoints of the 2n-1 intervals of the convolution
    # The convolution of two functions on [0, L] gives a function on [0, 2L]
    # Here L = 1/2, so f*f is on [-1/2, 1/2]
    
    # Let's just use np.convolve on the original heights
    conv_direct = np.convolve(heights, heights, mode='full')
    # conv_direct[k] approximates (f*f) at position (k - (n-1)) * w for k=0,...,2n-2
    # Actually, conv_direct[k] = sum_i h[i]*h[k-i] which is the discrete convolution
    # For continuous step functions, (f*f)(x_k) where x_k = (k - (n-1)) * w
    # is approximately w * conv_direct[k] ... no, it's w^2 * sum but with the trapezoidal shape
    
    # For a proper continuous autoconvolution of step functions, the result is piecewise linear
    # (trapezoidal). The value at the midpoint between two sample points can be computed.
    # For simplicity and speed, let's use the discrete convolution scaled by w
    # as an approximation, and take the max absolute value.
    
    # The continuous autoconvolution at x = (k - (n-1)) * w for integer k is:
    # w * conv_direct[k] if we're being careful, but actually for step functions
    # the autoconvolution at the "center" points is w * sum of products.
    
    # Let me just compute it properly for the midpoint samples
    # (f*f)(x) for x in [-1/2, 1/2]
    # Sample at x_j = (j - n + 0.5) * w for j = 0, ..., 2n-1? No, the range is [-1/2, 1/2]
    # which is 2n intervals of width w, so 2n+1 sample points if we include endpoints.
    
    # Simpler approach: the autoconvolution of two step functions with n intervals each
    # is a piecewise linear function with at most 2n-1 "peaks" (trapezoids).
    # The maximum occurs at one of the sample points x = (k - (n-1)) * w for k = 0,...,2n-2
    # or at the midpoints.
    
    # For step functions, (f*f) is actually piecewise linear, and its extrema occur at
    # x = i*w for integer i. So sampling at x_k = (k - (n-1)) * w should capture the max.
    
    # The value at x_k is: w * conv_direct[k] is NOT correct. Let me derive it.
    # f(t) = h_i for t in [(i-1)*w - 1/4, i*w - 1/4) for i=1,...,n
    # Let's index from 0: f(t) = h_i for t in [i*w - 1/4, (i+1)*w - 1/4)
    # (f*f)(x) = ∫ f(t) f(x-t) dt
    # For x = (k - (n-1)) * w, this is a sum of products of heights times overlap lengths.
    # The overlap of interval i for f(t) and interval j for f(x-t) is a trapezoid.
    
    # This is getting complicated. Let me use a numerical integration approach
    # by evaluating on a fine grid.
    
    # Actually, for the purpose of this task, let me use a simpler metric that's
    # computationally cheap. The L2 norm of the autoconvolution, or the max.
    
    # Let me use FFT for a quick approximation
    from scipy import fft as spfft
    # Pad to avoid circular convolution
    nfft = 2 * n
    H = spfft.fft(heights, nfft)
    conv_fft = spfft.ifft(H * H).real
    # The first n elements of conv_fft correspond to the autoconvolution
    # (f*f)(x_k) for x_k = k * w, k = 0, ..., n-1... no this isn't right either.
    
    # Let me just do it directly for now since n=400 is small
    # (f*f)(x) for x in [-1/2, 1/2], sample at x_k = (k - n) * w for k = 0,...,2n
    # At x_k, the autoconvolution is:
    # sum_{i=0}^{n-1} h_i * h_j * overlap_length where j is determined by x_k - t being in interval j
    
    # For a step function, the autoconvolution at x = m*w (m integer) is:
    # w * sum_{i=max(0, m-n+1)}^{min(n-1, m)} h_i * h_{m-i}
    # Wait, this is just w * conv_direct[m] for m in range(2n-1)
    
    # I think the correct formula is:
    # (f*f)(m*w) = w * sum_{i} h[i] * h[m-i] for valid i, m-i
    # which is w * conv_direct[m]
    
    return w * conv_direct


def _compute_c3(heights):
    """Compute the C3 upper bound for a given step function."""
    ac = _autoconvolve_step(heights)
    # C3 is likely the max absolute value of the autoconvolution
    # or some related quantity
    return np.max(np.abs(ac))


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try several structured patterns
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Antisymmetric: -1 on first half, +1 on second half
    antisym = np.zeros(N)
    antisym[:N//2] = -1
    antisym[N//2:] = 1
    candidates.append(antisym)
    
    # 3. Alternating blocks of size 10
    block_size = 10
    nblocks = N // block_size
    alt = np.zeros(N)
    for b in range(nblocks):
        val = 1 if b % 2 == 0 else -1
        alt[b*block_size:(b+1)*block_size] = val
    candidates.append(alt)
    
    # 4. Alternating blocks of size 50
    block_size = 50
    nblocks = N // block_size
    alt2 = np.zeros(N)
    for b in range(nblocks):
        val = 1 if b % 2 == 0 else -1
        alt2[b*block_size:(b+1)*block_size] = val
    candidates.append(alt2)
    
    # 5. Smooth transition: -1 on edges, +1 in middle
    smooth = np.zeros(N)
    for i in range(N):
        t = i / N  # 0 to 1
        # -1 near 0 and 1, +1 in middle
        if t < 0.25 or t > 0.75:
            smooth[i] = -1
        else:
            smooth[i] = 1
    candidates.append(smooth)
    
    # 6. Two transitions: -1, +1, -1
    tri = np.zeros(N)
    tri[:N//3] = -1
    tri[N//3:2*N//3] = 1
    tri[2*N//3:] = -1
    candidates.append(tri)
    
    # Evaluate all candidates
    best_val = np.inf
    best_candidate = candidates[0]
    
    for cand in candidates:
        val = _compute_c3(cand)
        if val < best_val:
            best_val = val
            best_candidate = cand
    
    # Check time
    if time.time() > deadline:
        return best_candidate
    
    # Try some random perturbations of the best candidate
    rng = np.random.default_rng(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        perturbed = best_candidate.copy()
        # Flip a random block
        start = rng.integers(0, N - 50)
        length = rng.integers(10, 100)
        perturbed[start:start+length] *= -1
        val = _compute_c3(perturbed)
        if val < best_val:
            best_val = val
            best_candidate = perturbed
    
    return best_candidate
# EVOLVE-BLOCK-END
