"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
W = 1.0 / 1200.0

def compute_ratio_fast(h):
    """
    Compute max(autoconv) / l2_sq for a step function defined on N equal intervals.
    Uses FFT for the autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    # The step function f(x) is constant h[i] on interval [x_i, x_{i+1})
    # x_i = -1/4 + i * W
    # Autoconvolution g(t) = integral f(x) f(x-t) dx
    # Since f is a step function, we can compute the convolution efficiently.
    # The convolution of two step functions with N steps is a piecewise constant
    # function with 2N-1 steps (if we consider the discrete sum approximation).
    
    # However, the "normalized maximum" likely refers to:
    # ||f * f||_inf / ||f||_2^2
    
    # Let's compute ||f||_2^2 first.
    # ||f||_2^2 = sum_i h[i]^2 * W
    
    l2_sq = np.sum(h**2) * W
    if l2_sq < 1e-15:
        return float('inf')
    
    # Now compute the autoconvolution.
    # g(t) = sum_i h[i] h[j] * overlap_length(t, i, j)
    # This is effectively the discrete convolution of the sequence h with itself,
    # scaled appropriately?
    
    # Let's verify.
    # f(x) = sum_i h[i] * 1_{[x_i, x_{i+1})}(x)
    # f*f (t) = integral f(x) f(x-t) dx
    # = integral (sum_i h[i] 1_{x in [x_i, x_{i+1})}) (sum_j h[j] 1_{x-t in [x_j, x_{j+1})}) dx
    # = sum_i sum_j h[i] h[j] integral 1_{x in [x_i, x_{i+1})} 1_{x in [x_j+t, x_{j+1}+t]} dx
    # The integral is the length of the intersection of [x_i, x_{i+1}) and [x_j+t, x_{j+1}+t].
    
    # This is the convolution of the step function with itself.
    # We can compute this using FFT.
    # Pad h to length 2*N to avoid circular convolution artifacts.
    
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    
    # FFT convolution
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h_pad, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    
    # The discrete convolution result `conv` approximates the integral?
    # Let's check the scaling.
    # The standard FFT convolution computes sum_k a[k] b[n-k].
    # Here a and b are the step heights.
    # The actual integral is sum_i sum_j h[i] h[j] * overlap(i, j, t).
    # The overlap length is W if the intervals are fully overlapping (which happens for specific t).
    # Actually, the convolution of two rectangular pulses of width W and height H is a triangle.
    # The discrete convolution of the heights [H, 0, ...] with itself gives [H^2, 0, ...].
    # This doesn't seem to capture the geometry directly without scaling.
    
    # Let's use the property that the L2 norm of the convolution is related to the L2 norm of the function.
    # ||f*f||_1 = ||f||_1^2.
    # ||f*f||_inf <= ||f||_1^2.
    
    # Let's re-evaluate the "normalized maximum".
    # Usually, inequalities of the form ||f*f||_inf <= C ||f||_2^2 are studied.
    # For the uniform function on [-1/4, 1/4] (height 1, width 1/2):
    # ||f||_2^2 = 1 * (1/2) = 0.5.
    # f*f is a triangle with base 1/2 and height 1/2.
    # ||f*f||_inf = 1/2.
    # Ratio = (1/2) / 0.5 = 1.
    
    # Let's check the FFT result for uniform h.
    # h = [1, 1, ..., 1] (N times).
    # conv[k] for k=0..N-1 should be proportional to k+1.
    # conv[N-1] should be N.
    # The actual integral at t=0 is ||f||_2^2 = N * W.
    # The FFT sum is sum h[i]^2 = N.
    # So the FFT sum is ||f||_2^2 / W.
    # Therefore, the actual autoconvolution value is conv_fft * W.
    
    max_conv = np.max(conv) * W
    ratio = max_conv / l2_sq
    
    return ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Strategy:
    # 1. Generate diverse initial candidates (analytical shapes).
    # 2. Evaluate them.
    # 3. Use local refinement (gradient descent / coordinate descent) on the best candidate.
    
    i = np.arange(N)
    t = i / N  # 0 to 1
    
    candidates = []
    
    # 1. Uniform (baseline, score 1.0)
    candidates.append(np.ones(N))
    
    # 2. Cosine bell (smooth, compact support)
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    
    # 3. Triangular peak
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))
    
    # 4. Gaussian
    sigma = 0.2
    candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))
    
    # 5. Spiky (concentrated in center)
    candidates.append(np.exp(-10 * (t - 0.5) ** 2))
    
    # 6. Disjoint pulses (Inspiration 2 idea)
    # Alternating 1, 0, 1, 0...
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # 7. Disjoint pulses (4-wide blocks)
    h_block = np.zeros(N)
    h_block[::4] = 1.0
    candidates.append(h_block)
    
    # Evaluate initial candidates
    best_h = candidates[0].copy()
    best_ratio = compute_ratio_fast(best_h)
    
    for h in candidates[1:]:
        h = np.clip(h, 0.0, None)
        r = compute_ratio_fast(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Local Refinement
    # We want to minimize ratio(h).
    # Use a simple coordinate descent or gradient step.
    # Given the time budget, coordinate descent on a subset of coordinates is safer.
    
    h = best_h.copy()
    lr = 0.05
    eps = 0.01
    
    # We will perform iterations until deadline.
    # Each iteration: pick a random subset of coordinates, compute gradient (finite diff), update.
    
    # To speed up gradient computation, we can reuse FFTs if we change only a few points?
    # Not easily with standard FFT.
    # However, compute_ratio_fast is quite fast (~1ms for N=600).
    # 150s allows ~100,000 evaluations.
    # If we do 10 coords per step, that's 10 evaluations.
    # So we can do ~10,000 steps.
    
    step_count = 0
    while time.time() < deadline:
        step_count += 1
        
        # Pick random coordinates to update
        n_coords = 10
        idxs = np.random.choice(N, size=n_coords, replace=False)
        
        grad = np.zeros(N)
        for idx in idxs:
            h_old = h[idx]
            
            # Central difference
            h[idx] = h_old + eps
            r_plus = compute_ratio_fast(h)
            
            h[idx] = h_old - eps
            r_minus = compute_ratio_fast(h)
            
            h[idx] = h_old
            
            grad[idx] = (r_plus - r_minus) / (2 * eps)
        
        # Update
        h -= lr * grad
        h = np.clip(h, 0.0, None)
        
        # Evaluate new state
        r_new = compute_ratio_fast(h)
        if r_new < best_ratio:
            best_ratio = r_new
            best_h = h.copy()
        else:
            # If it got worse, revert or adjust? 
            # For simplicity, if it's significantly worse, revert.
            if r_new > best_ratio * 1.1:
                h = best_h.copy()
            # Otherwise, keep h to allow exploration? 
            # Gradient descent usually keeps the state if it's a step, but here we check improvement.
            # Let's always revert if not improved to stay on the manifold of "good" solutions?
            # No, that's hill climbing. Let's just keep the best found so far.
            # But we need to continue from a good state.
            # Let's revert if not improved.
            h = best_h.copy()
            
        # Adaptive learning rate?
        # If we haven't improved in a while, reduce lr?
        # For now, keep it simple.
        
    return best_h.tolist()
