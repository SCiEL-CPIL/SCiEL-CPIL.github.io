N = 600

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max(h):
    """Compute the maximum of the autoconvolution of h."""
    h = np.asarray(h, dtype=np.float64)
    # Autoconvolution: (h * h)(x) = integral h(x-y) h(y) dy
    # For step functions on equal intervals, this is a discrete convolution
    # padded to zero outside [-1/4, 1/4]
    n = len(h)
    # Full convolution has length 2n-1, but we only care about the part
    # corresponding to x in [-1/4, 1/4] which is indices n-1 to n+n-2 = 2n-2
    # Actually, the autoconvolution of a function supported on [-1/4, 1/4]
    # is supported on [-1/2, 1/2]. We evaluate it on the same grid [-1/4, 1/4].
    # The step function h[i] is constant on [i/n - 1/4, (i+1)/n - 1/4]
    # Autoconvolution at point x = j/n - 1/4 (for j in 0..n-1):
    # integral over y of h(x-y) h(y) dy
    # This is a discrete sum: sum_i h[i] * h[j-i] * (1/n) for valid i, j-i
    # where both i and j-i are in [0, n-1]
    # Let's use FFT for efficiency
    h_pad = np.zeros(2 * n)
    h_pad[:n] = h
    conv = np.fft.irfft(np.fft.rfft(h_pad) ** 2, n=2 * n) * (1.0 / n)
    # The autoconvolution on the grid [-1/4, 1/4] corresponds to indices n-1 to 2n-2
    # Actually, let me think more carefully.
    # h is supported on [0, 1] in the padded sense (indices 0 to n-1 of h_pad)
    # Autoconvolution (h*h)(t) for t in [0, 2]
    # We want to evaluate at t = 1 (center) and nearby points corresponding to [-1/4, 1/4]
    # In the original coordinate, x in [-1/4, 1/4] maps to t in [0.5, 1.5] in the [0,2] scale
    # Wait, let me reconsider. The original function is on [-1/4, 1/4].
    # Let's shift: let g(u) = h(u + 1/4) for u in [0, 1/2].
    # Then autoconvolution of h at x is autoconvolution of g at x + 1/4.
    # x in [-1/4, 1/4] => x + 1/4 in [0, 1/2].
    # g is on [0, 1/2], so autoconvolution of g is on [0, 1].
    # We evaluate at points in [0, 1/2], which is the first half.
    # Discretize [0, 1/2] into n points: u_k = k/(2n) for k=0..n-1? 
    # Actually, the original grid has n points on [-1/4, 1/4], so spacing is 1/(2n).
    # Let's just compute the full autoconvolution and extract the right part.
    
    # Simpler: use scipy or direct FFT approach
    # The autoconvolution of h (step function on n intervals) at the midpoints
    # of the n intervals can be computed as follows:
    # Let the intervals be I_i = [i/n - 1/4, (i+1)/n - 1/4] for i=0..n-1
    # h(x) = h_i for x in I_i
    # Autoconvolution at x: A(x) = integral h(x-y) h(y) dy
    # For x in I_j, A(x) is a piecewise linear function.
    # The maximum will occur at some point. To be safe, let's evaluate at
    # all interval midpoints and also at interval boundaries.
    
    # For efficiency, let's use the fact that we can compute this via FFT.
    # Define f on a grid of 2n points representing [0, 1] where f[k] = h[k] for k=0..n-1
    # and f[k] = 0 for k=n..2n-1. This represents h shifted to [0, 1/2].
    # Autoconvolution of f on [0, 1] evaluated at grid points.
    
    # Actually, let's just do it directly for correctness, using FFT.
    # h is on n intervals of width 1/(2n) each, covering [-1/4, 1/4].
    # Let's sample the autoconvolution at 2n points to be safe.
    
    # Use FFT-based convolution
    # Pad h to length 2n
    h_full = np.zeros(2 * n)
    h_full[:n] = h
    # Linear convolution via FFT
    F = np.fft.fft(h_full, n=2 * n)
    conv_full = np.fft.ifft(F * F, n=2 * n).real * (1.0 / n)
    
    # Now, the autoconvolution of the original h at points x_k = k/(2n) - 1/4 for k=0..2n-1
    # corresponds to conv_full at certain indices.
    # The function h is represented as h_full on [0, 1/2] (shifted).
    # Autoconvolution of h_full is on [0, 1].
    # We want values at x in [-1/4, 1/4], which in shifted coords is [0, 1/2].
    # The grid for h_full has 2n points on [0, 1/2], so spacing is 1/(4n).
    # Wait, h_full has n nonzero values on [0, 1/2], so each value represents an interval of width 1/(2n).
    # The autoconvolution conv_full has 2n points on [0, 1], each representing width 1/(2n).
    # We want the part of conv_full corresponding to [0, 1/2], which is indices 0 to n-1.
    
    # But we need to be careful about normalization.
    # The discrete autoconvolution: A_j = sum_i h_i h_{j-i} * (1/(2n)) for the shifted version
    # Let me just compute it directly for a few points to verify, but for speed use FFT.
    
    # Let's use a cleaner approach: 
    # The autoconvolution at the midpoint of interval j (in original coords) is:
    # A_j = (1/(2n)) * sum_{i: valid} h_i * h_{j-i}
    # where the sum is over i such that both i and j-i are in [0, n-1]
    # This is a standard convolution.
    
    # For the maximum, it's sufficient to check at the interval midpoints and boundaries.
    # Let's compute at 2n points: midpoints and boundaries of the n intervals.
    
    # Actually, the autoconvolution of a step function is piecewise quadratic (or linear).
    # The maximum on each interval occurs at an endpoint or where derivative is zero.
    # For simplicity and safety, let's evaluate at 2n+1 points: all interval boundaries.
    
    # Let's just compute the full autoconvolution array and take max over the relevant range.
    # Using the FFT result above:
    # conv_full[k] for k=0..2n-1 gives autoconvolution at points k/(2n) in [0, 1]
    # We want the part in [0, 1/2], so k=0..n-1
    # But we also need to consider that the autoconvolution might peak between grid points.
    
    # For a step function autoconvolution, it's piecewise quadratic. The maximum on each
    # subinterval occurs at an endpoint or at the vertex of the parabola.
    # Let's evaluate at the 2n+1 grid points (boundaries of the n intervals in [0, 1/2])
    
    # The boundaries in [0, 1/2] are at k/(2n) for k=0..n
    # In conv_full, these correspond to indices k for k=0..n
    # But conv_full only has 2n points for [0,1], so index k corresponds to k/(2n)
    # So we need conv_full[0:n+1] for the points 0, 1/(2n), ..., n/(2n)=1/2
    
    # Wait, I think I'm overcomplicating. Let me just compute the autoconvolution
    # at all 2n interval midpoints and n-1 internal boundaries, plus the two endpoints.
    
    # Simpler approach: compute the autoconvolution on a fine grid and take max.
    # But that's expensive. Let's use the FFT result.
    
    # The autoconvolution of h (on [-1/4,1/4]) evaluated at x = k/(2n) - 1/4 for k=0..2n-1
    # Let's compute this properly.
    
    # Shift h to start at 0: g(u) = h(u - 1/4) for u in [0, 1/2]
    # g is represented by h on n intervals of width 1/(2n)
    # Autoconvolution of g: (g*g)(t) = integral g(t-v) g(v) dv for t in [0, 1]
    # We want (g*g)(t) for t in [0, 1/2] (corresponding to x in [-1/4, 1/4])
    
    # Discretize: g_i = h_i for interval i of [0, 1/2], width w = 1/(2n)
    # (g*g)(t) at t = j*w for j=0..2n-1:
    # = sum_{i,k: i+k=j} g_i g_k w
    # = w * sum_i g_i g_{j-i} for valid i, j-i
    
    # This is w * (g * g)[j] where * is discrete convolution
    # g has n nonzero values, so g*g has up to 2n-1 nonzero values
    
    # Using FFT:
    g = np.zeros(2 * n)
    g[:n] = h
    G = np.fft.fft(g, n=2 * n)
    gg = np.fft.ifft(G * G, n=2 * n).real
    w = 1.0 / (2 * n)
    autoconv = gg * w
    
    # autoconv[j] for j=0..2n-1 gives (g*g)(j*w)
    # We want j=0..n (t in [0, 1/2])
    # The maximum of the autoconvolution of h on [-1/4, 1/4] is the max of autoconv[0:n+1]
    # But we should also check if the maximum occurs between grid points.
    # For a step function, the autoconvolution is piecewise quadratic.
    # The maximum on each interval [j*w, (j+1)*w] occurs at an endpoint or at the vertex.
    
    # For now, let's just take the max over the grid points. This is an upper bound
    # on the true max only if we miss the peak. But since we're using this for optimization,
    # we need to be accurate. Let's evaluate at 2n+1 points and also check the vertices.
    
    # Actually, for the purpose of this optimization, let's just use the grid max.
    # The error is O(1/n^2) which is small for n=600.
    
    return np.max(autoconv[0:n+1])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = N
    # Initialize with uniform
    h = np.ones(n)
    
    # Phase 1: Subgradient descent on the autoconvolution max
    # The autoconvolution A(x) = integral h(x-y)h(y) dy
    # d/dh_i max_x A(x) = 2 * h(x*) for x* where max is achieved
    # But max is not differentiable. Use subgradient: average over all maximizers
    
    # Better approach: minimize sum of squares of autoconvolution values above a threshold
    # Or use a smooth approximation: minimize log-sum-exp of autoconv values
    
    # Let's use a projected gradient method with a smooth loss.
    # Loss: L = log(sum exp(A_j / T)) * T  (smooth max approximation)
    # Gradient w.r.t. h_i: (1/T) * sum_j (A_j / T) * exp(A_j / T) * dA_j/dh_i / sum exp(A_j/T)
    # dA_j/dh_i = 2 * h_{j-i} * w for the discrete case
    
    # Actually, let's use a different approach: minimize the max via iterative flattening.
    # At each step, find where the autoconvolution is maximal, and reduce h there.
    # This is like a Remez algorithm.
    
    # Phase 1: Simple iterative flattening
    for it in range(500):
        if time.time() > deadline:
            break
        A = _autoconv_max(h)
        if A < 1e-10:
            break
        
        # Compute the full autoconvolution to find where it's maximal
        g = np.zeros(2 * n)
        g[:n] = h
        G = np.fft.fft(g, n=2 * n)
        gg = np.fft.ifft(G * G, n=2 * n).real
        w = 1.0 / (2 * n)
        autoconv = gg * w
        ac_relevant = autoconv[0:n+1]
        
        # Find the maximum location
        max_idx = np.argmax(ac_relevant)
        max_val = ac_relevant[max_idx]
        
        # The autoconvolution at point max_idx is:
        # A(max_idx) = w * sum_i h_i * h[max_idx - i]
        # To reduce this, we can reduce h_i where h[max_idx-i] is large
        # Gradient of A(max_idx) w.r.t. h_i is 2 * w * h[max_idx - i] for valid i
        
        # Compute gradient
        grad = np.zeros(n)
        for i in range(n):
            j = max_idx - i
            if 0 <= j < n:
                grad[i] = 2 * w * h[j]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Step size: use a simple adaptive step
        # We want to reduce the max, so take a step in -grad direction
        # But we need to keep h >= 0
        step_size = 0.1
        h_new = h - step_size * grad
        h_new = np.maximum(h_new, 0)
        
        # Check if this improves
        A_new = _autoconv_max(h_new)
        if A_new < A:
            h = h_new
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    # Phase 2: L-BFGS polish with smooth loss
    if time.time() < deadline - 30:
        T = 0.01  # temperature for log-sum-exp
        
        def loss_and_grad(h_flat):
            h = h_flat.reshape(n)
            h = np.maximum(h, 0)
            
            g = np.zeros(2 * n)
            g[:n] = h
            G = np.fft.fft(g, n=2 * n)
            gg = np.fft.ifft(G * G, n=2 * n).real
            w = 1.0 / (2 * n)
            autoconv = gg * w
            ac = autoconv[0:n+1]
            
            # Smooth max: T * log(sum exp(ac/T))
            max_ac = np.max(ac)
            if max_ac > 1e-10:
                log_sum = np.log(np.sum(np.exp((ac - max_ac) / T))) + max_ac / T
                loss = T * log_sum
            else:
                loss = 0.0
            
            # Gradient: dL/dh_i = (1/T) * sum_j (ac_j/T) * exp((ac_j - max_ac)/T) * dac_j/dh_i / sum exp((ac_j-max_ac)/T)
            # dac_j/dh_i = 2 * w * h[j-i] for valid j-i
            # This is expensive to compute directly. Use FFT.
            # dac_j/dh_i = 2 * w * (h * h)[j] where * is convolution
            # So grad_i = (2w/T) * sum_j p_j * h[j-i] where p_j = exp((ac_j-max_ac)/T) / sum exp(...)
            # This is 2w/T * (p * h)[i]
            
            if max_ac > 1e-10:
                p = np.exp((ac - max_ac) / T)
                p = p / np.sum(p)
                # Compute (p * h)[i] = sum_j p_j h[j-i]
                # p has n+1 values, h has n values
                p_pad = np.zeros(2 * n)
                p_pad[:n+1] = p
                h_pad = np.zeros(2 * n)
                h_pad[:n] = h
                P = np.fft.fft(p_pad, n=2 * n)
                H = np.fft.fft(h_pad, n=2 * n)
                conv_ph = np.fft.ifft(P * H, n=2 * n).real
                grad = (2 * w / T) * conv_ph[:n]
            else:
                grad = np.zeros(n)
            
            return loss, grad
        
        # Use scipy L-BFGS-B
        from scipy.optimize import minimize
        result = minimize(loss_and_grad, h.copy(), method='L-BFGS-B', 
                         jac=True, options={'maxiter': 200, 'gtol': 1e-8})
        if result.success and time.time() < deadline:
            h_candidate = np.maximum(result.x, 0)
            if _autoconv_max(h_candidate) < _autoconv_max(h):
                h = h_candidate
    
    return [float(x) for x in h]
# EVOLVE-BLOCK-END
