"""
Program for minimizing the normalized maximum of autoconvolution of a
nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Approach:
- We use Gradient Descent with a Softmax approximation of the maximum.
- The objective is to minimize the soft-max of the autoconvolution values.
- This allows the gradient to be influenced by all significant peaks,
  not just the single global maximum, leading to a flatter autoconvolution
  profile and potentially a lower peak value.
- We use additive updates followed by projection onto the non-negative
  orthant and L2 normalization.
"""

import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Start with a uniform distribution
    h = np.ones(N)
    h = h / np.sqrt(np.sum(h ** 2))
    
    best_h = h.copy()
    best_obj = np.max(np.convolve(h, h, mode='full'))
    
    # Hyperparameters
    eta = 0.05  # Learning rate
    beta = 10.0 # Temperature for softmax (higher = closer to hard max)
    max_iters = 3000
    patience = 150
    no_improve_count = 0
    
    # Precompute indices for gradient calculation
    # We need to compute grad[j] = sum_m w[m] * (h[m-j] + h[j-m])
    # This is a convolution of w with h, plus h with w?
    # Let c = h * h.
    # grad = w * h + w * h (reversed)?
    # (w * h)[j] = sum_m w[m] h[j-m]
    # (h * w)[j] = sum_m h[m] w[j-m]
    # Since w is symmetric (autoconvolution is symmetric, softmax of symmetric is symmetric),
    # w * h = h * w.
    # So grad = 2 * (w * h).
    
    for it in range(max_iters):
        # Compute autoconvolution
        ac = np.convolve(h, h, mode='full')
        current_obj = np.max(ac)
        
        if current_obj < best_obj:
            best_obj = current_obj
            best_h = h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                break
        
        # Compute softmax weights
        # Shift ac to center to avoid numerical issues with large exponents
        # ac is symmetric, max is at center (index N-1)
        # We only care about relative values
        max_ac = np.max(ac)
        # Use a stable softmax
        exp_ac = np.exp(beta * (ac - max_ac))
        w = exp_ac / np.sum(exp_ac)
        
        # Compute gradient
        # grad = 2 * convolve(w, h, mode='same') ?
        # np.convolve(w, h, 'full') has length 2N-1.
        # We need the part corresponding to indices 0..N-1 of h.
        # The convolution result y[k] = sum_i w[i] h[k-i].
        # We want grad[j] = sum_m w[m] h[j-m] + sum_m w[m] h[m-j].
        # Since w is symmetric, w[m] = w[-m] (conceptually).
        # Let's just compute the full convolution and slice.
        # conv_wh = np.convolve(w, h, mode='full')
        # The term sum_m w[m] h[j-m] is conv_wh[j] for j=0..N-1?
        # conv_wh[k] = sum_{i=0}^{N-1} w[i] h[k-i].
        # If we set k=j, we get sum_i w[i] h[j-i]. This matches the second term if we swap indices.
        # The first term is sum_m w[m] h[m-j]. Let i = m-j => m = i+j.
        # sum_i w[i+j] h[i]. This is a cross-correlation.
        # Actually, since h and w are real, grad = 2 * (w * h) if w is symmetric and centered?
        # Let's verify.
        # c[m] = sum_i h[i] h[m-i].
        # d c[m] / d h[j] = h[m-j] + h[j-m].
        # grad[j] = sum_m w[m] (h[m-j] + h[j-m]).
        # Term 1: sum_m w[m] h[m-j]. Let k = m-j => m = k+j.
        # sum_k w[k+j] h[k]. This is the cross-correlation of w and h at lag j?
        # Term 2: sum_m w[m] h[j-m]. Let k = j-m => m = j-k.
        # sum_k w[j-k] h[k]. This is the convolution (w * h)[j].
        # Since w is symmetric (w[m] = w[N-1-m] roughly, actually w is defined on 0..2N-2 and symmetric around N-1),
        # and h is not necessarily symmetric.
        # However, w is derived from h*h which is symmetric.
        # So w is symmetric.
        # If w is symmetric, w[k+j] is not simply related to w[j-k] unless w is even around 0.
        # But w is defined on the convolution support.
        
        # Simpler approach: Compute the gradient directly using the precomputed structure
        # or just use the fact that grad is the derivative of the quadratic form.
        # J = w^T c. c = h * h.
        # grad = 2 * (w * h) if w is symmetric and we treat the indices correctly.
        
        # Let's use the property: grad = 2 * np.convolve(w, h, mode='same') is NOT quite right due to shifts.
        # Let's compute it explicitly for safety, though N=600 is small enough for O(N^2) if needed,
        # but we can do it in O(N log N) or O(N) with conv.
        
        # conv_wh = np.convolve(w, h, mode='full')
        # The gradient component for h[j] involves w shifted.
        # Let's just use the explicit loop for the first few iterations to verify, 
        # but for speed, let's use the convolution trick.
        
        # Actually, the gradient of sum_m w_m (h*h)_m with respect to h_j
        # is sum_m w_m (h_{m-j} + h_{j-m}).
        # This is exactly 2 * (w * h)_j IF w is symmetric and centered at 0.
        # Our w is indexed 0..2N-2. The center is N-1.
        # Let's shift w to be centered at 0.
        # w_centered[k] for k in -(N-1)..(N-1).
        
        # Instead of complex shifting, let's observe that for small N, we can just do:
        # grad = np.zeros(N)
        # for j in range(N):
        #    # sum_m w[m] h[m-j] + h[j-m]
        #    # This is O(N^2) per iteration. 600^2 * 3000 = 1e9. Too slow.
        
        # We need O(N log N) or O(N).
        # Let's use FFT.
        # H = fft(h, 2*N)
        # W = fft(w, 2*N)
        # C = ifft(H*W)
        # The gradient is related to the real part of the inverse FFT of W*H?
        # grad_j = sum_m w_m h_{m-j} + sum_m w_m h_{j-m}
        # Let's define w_ext and h_ext.
        
        # Actually, there is a simpler way.
        # The gradient of the peak is the sum of the "influence" of each peak.
        # For the soft max, the gradient is the weighted sum of the gradients of the individual peaks.
        # grad_j = sum_m w_m (h_{m-j} + h_{j-m})
        
        # Let's compute conv_w_h = np.convolve(w, h, mode='full')
        # conv_w_h[k] = sum_i w[i] h[k-i]
        # The second term is sum_m w[m] h[j-m]. Let i = j-m => m = j-i.
        # sum_i w[j-i] h[i]. This is the cross-correlation of w and h at lag j.
        # The first term is sum_m w[m] h[m-j]. Let i = m-j => m = i+j.
        # sum_i w[i+j] h[i]. This is the cross-correlation of w and h at lag -j?
        
        # If w is symmetric (w[i] = w[2N-2-i]), then:
        # w[i+j] = w[2N-2-(i+j)] = w[2N-2-i-j].
        # This doesn't immediately simplify to the convolution.
        
        # Let's stick to the explicit calculation but optimized.
        # Since N=600, O(N^2) is 360,000 operations.
        # 3000 iterations * 360,000 = 1.08e9 operations.
        # In Python, this might be too slow (120s limit).
        # We need to vectorize or use FFT.
        
        # Using FFT:
        # Let's pad h and w to length 2N.
        # h_pad = np.zeros(2*N)
        # h_pad[:N] = h
        # w_pad = np.zeros(2*N)
        # w_pad[:2*N-1] = w  # w has length 2N-1
        
        # H = np.fft.rfft(h_pad)
        # W = np.fft.rfft(w_pad)
        
        # The term sum_m w[m] h[j-m] is the linear convolution (w * h)[j].
        # The term sum_m w[m] h[m-j] is the cross-correlation.
        # Note that (w * h)[j] = sum_i w[i] h[j-i].
        # If we reverse w, let w_rev[i] = w[2N-2-i]. Since w is symmetric, w_rev = w (shifted).
        # Actually, w is symmetric around N-1.
        # Let's just compute the two terms separately using FFT.
        
        # Term 2: Convolution of w and h.
        # conv_wh = np.fft.irfft(np.fft.rfft(w, 2*N) * np.fft.rfft(h, 2*N), 2*N)
        # We need conv_wh[0:N].
        
        # Term 1: Cross-correlation.
        # corr_wh[j] = sum_m w[m] h[m-j].
        # This is equivalent to the convolution of w with the reversed h?
        # Let h_rev[i] = h[N-1-i].
        # No, let's use the property: corr_w_h[j] = (w * h_rev)[N-1-j]?
        
        # Simpler:
        # grad = np.zeros(N)
        # We can compute the gradient by:
        # 1. Compute C = np.convolve(w, h, mode='full')
        #    C[k] = sum_i w[i] h[k-i]
        #    This gives the second term for index j if we look at C[j]?
        #    C[j] = sum_i w[i] h[j-i]. Let m=i. sum_m w[m] h[j-m]. Yes.
        #    So Term2[j] = C[j].
        
        # 2. Compute the first term: sum_m w[m] h[m-j].
        #    Let's define h_shifted such that h_shifted[m] = h[m-j].
        #    This is equivalent to summing w[m] * h[m-j].
        #    This is the cross-correlation of w and h.
        #    corr[j] = sum_m w[m] h[m-j].
        #    Note that corr[j] = C[2N-2-j]?
        #    Let's check.
        #    C[k] = sum_i w[i] h[k-i].
        #    Let k = 2N-2-j.
        #    C[2N-2-j] = sum_i w[i] h[2N-2-j-i].
        #    Let m = 2N-2-i. Then i = 2N-2-m.
        #    Sum over i=0..N-1 corresponds to m=2N-2..N-1.
        #    h[2N-2-j-(2N-2-m)] = h[m-j].
        #    w[2N-2-m] = w[m] (by symmetry of w).
        #    So C[2N-2-j] = sum_{m=N-1}^{2N-2} w[m] h[m-j].
        #    Since h[k] = 0 for k outside 0..N-1, and w[m]=0 for m outside 0..2N-2,
        #    the sum effectively covers all valid m.
        #    So Term1[j] = C[2N-2-j].
        
        # Therefore, grad[j] = C[j] + C[2N-2-j].
        
        C = np.convolve(w, h, mode='full')
        # C has length 2N-1.
        # We need C[0:N] and C[2N-2 : 2N-2-N : -1] which is C[2N-2-j] for j=0..N-1.
        # C[2N-2-j] for j=0 is C[2N-2].
        # C[2N-2-j] for j=N-1 is C[N-1].
        # So we take the last N elements of C in reverse order.
        
        term1 = C[2*N-2 : N-2 : -1] # Length N
        term2 = C[0:N]               # Length N
        
        grad = term1 + term2
        
        # Gradient descent step
        h_new = h - eta * grad
        
        # Projection: clip to 0
        h_new = np.clip(h_new, 0, None)
        
        # Normalize
        norm = np.sqrt(np.sum(h_new ** 2))
        if norm < 1e-12:
            h = np.ones(N)
            h = h / np.sqrt(np.sum(h ** 2))
            continue
            
        h = h_new / norm
        
        # Adaptive parameters
        # Increase beta to focus more on the peak as we converge
        beta *= 1.005
        # Decrease eta slightly to stabilize
        eta *= 0.999

    return list(best_h)
# EVOLVE-BLOCK-END
