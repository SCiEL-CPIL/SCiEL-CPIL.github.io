import numpy as np
import time
import random
from scipy.optimize import minimize

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv(h):
        """Compute full autoconvolution of h (linear)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    def smooth_objective_and_grad(h, tau=1.0):
        """
        Compute smooth approximation of max(autoconv) and its gradient.
        F(h) = tau * log(sum(exp(c_k / tau)))
        grad F = (1/sum(exp(c_k/tau))) * sum(exp(c_k/tau) * grad(c_k))
        """
        c = compute_autoconv(h)
        # Shift to avoid overflow in exp
        c_max = np.max(c)
        exp_c = np.exp((c - c_max) / tau)
        sum_exp = np.sum(exp_c)
        
        obj = tau * (np.log(sum_exp) + c_max / tau)
        
        # Gradient of sum(exp(c_k/tau)) is sum(exp(c_k/tau) * grad(c_k)/tau)
        # We need grad(c_k) for each k.
        # c_k = sum_i h_i h_{k-i}
        # grad(c_k) is a vector v_k where v_k[l] = 2 h_{k-l} (if 0<=k-l<N) else 0
        # Actually, if k-l == l, it's 2 h_l. If not, it's h_{k-l} + h_{k-l} = 2 h_{k-l}?
        # Let's re-verify.
        # c_k = sum_{i=0}^{N-1} h_i h_{k-i}
        # d c_k / d h_l = sum_i (delta_{i,l} h_{k-i} + h_i delta_{k-i,l})
        # = h_{k-l} + h_{k-l} = 2 h_{k-l} (if k-l is in range)
        # Wait, if i=l, term is h_{k-l}. If k-i=l => i=k-l, term is h_{k-l}.
        # So yes, 2 h_{k-l}.
        
        # We can compute the weighted sum of gradients efficiently.
        # grad_F[l] = (1/tau) * (1/sum_exp) * sum_k exp_c[k] * 2 h_{k-l}
        # This is a convolution of exp_c with h?
        # sum_k exp_c[k] h_{k-l} is the cross-correlation of exp_c and h.
        
        # Let's construct the gradient vector.
        # grad_obj = (1/sum_exp) * sum_k exp_c[k] * grad(c_k)
        # Note: d/dh (tau log sum exp(c/tau)) = (1/sum exp) sum exp(c/tau) grad c
        # The 1/tau cancels?
        # d/dh (tau log S) = tau * (1/S) dS/dh = tau * (1/S) sum (exp(c_k/tau)/tau) grad c_k = (1/S) sum exp(c_k/tau) grad c_k.
        
        # So we need G[l] = sum_k w_k * 2 h_{k-l} where w_k = exp_c[k].
        # This is 2 * (w * h)_l where * is cross-correlation?
        # (w * h)_l = sum_k w_k h_{k-l}.
        # Yes.
        
        # We can compute this via FFT.
        size = 2 * N
        w_pad = np.zeros(size)
        w_pad[:size] = w_k # w_k is exp_c, length 2N-1?
        # c has length 2N-1.
        # h has length N.
        # We need correlation of w (len 2N-1) and h (len N).
        # Result length 3N-2?
        # We only need indices 0..N-1 for the gradient of h.
        
        # Let's just do it directly if N is small? N=600. 600*1200 = 720k ops. Fast.
        # Or FFT.
        
        # FFT approach for correlation:
        # corr = ifft(fft(w) * conj(fft(h)))
        # Need to pad to appropriate size.
        
        L = 2 * N
        w_fft = np.fft.fft(w_k, n=L)
        h_fft = np.fft.fft(h, n=L)
        # Correlation: ifft(fft(w) * conj(fft(h)))
        # The result is circular correlation.
        # We need linear correlation.
        # Pad to 2*L?
        
        # Simpler:
        # G = 2 * np.correlate(w_k, h, mode='full')
        # We need the part corresponding to l=0..N-1.
        # correlate(w, h, 'full')[k] = sum_i w_i h_{i-k}?
        # numpy.correlate(a, v, mode)
        # a is w, v is h.
        # c_k = sum_n a[n+k] conj(v[n])
        # We want sum_k w_k h_{k-l}.
        # This is cross-correlation.
        
        # Let's stick to the direct sum for simplicity and correctness, it's fast enough.
        # N=600, len(c)=1199.
        # 600 * 1200 = 720,000 multiplications.
        # In Python loop this is slow. In NumPy vectorized?
        
        # Vectorized:
        # We want G[l] = sum_k w_k h_{k-l}
        # Indices k-l must be in [0, N-1].
        # So k must be in [l, l+N-1].
        
        # We can use FFT for the convolution/correlation.
        # Let's use scipy.signal.correlate or FFT.
        
        # FFT size 2048 (power of 2 > 3N)
        fft_size = 2048
        W = np.fft.fft(w_k, n=fft_size)
        H = np.fft.fft(h, n=fft_size)
        # Correlation of w and h: ifft(W * conj(H))
        # This gives circular correlation.
        # Linear correlation is the first N + len(w) - 1 terms?
        # w has len 2N-1. h has len N.
        # Correlation length 3N-2.
        # We need indices 0 to N-1 of the linear correlation.
        
        corr = np.fft.ifft(W * np.conj(H)).real
        # The linear correlation is embedded in the circular one if fft_size is large enough.
        # The first N elements of corr should be the linear correlation for l=0..N-1?
        # Let's check.
        # corr[m] = sum_n w[n] h[n-m] (circular).
        # If no wrap-around, corr[m] = sum_n w[n] h[n-m].
        # We want sum_k w_k h_{k-l}. Let n=k. sum_n w_n h_{n-l}.
        # This matches corr[l] if we align properly.
        
        # Actually, np.correlate(w, h, 'full') gives:
        # c_k = sum_n w[n+k] h[n]
        # We want sum_k w_k h_{k-l}.
        # Let's just use the gradient of the max directly for the subgradient part, 
        # and for the smooth part, maybe just use the subgradient of the top few?
        
        # Given the complexity, let's use the Subgradient Descent with Top-K averaging as the primary method.
        # It is robust and fast.
        
        return obj, None # Placeholder

    # We will implement Subgradient Descent with Top-K averaging.
    
    def subgradient_step(h, k_max, top_k=10):
        """
        Compute average subgradient of max(autoconv) using top_k maxima.
        """
        c = compute_autoconv(h)
        # Find top k indices
        # Use argpartition for speed
        if top_k < len(c):
            top_indices = np.argpartition(c, -top_k)[-top_k:]
        else:
            top_indices = np.arange(len(c))
        
        # Weights for each top index (optional, e.g. softmax of c)
        # Simple average
        grad = np.zeros(N)
        
        for k in top_indices:
            # grad(c_k) = 2 h_{k-l}
            # For a given k, the gradient vector g_l is 2 h_{k-l} if 0 <= k-l < N
            # We need to add this to grad.
            
            # Efficiently:
            # g is a shifted version of h.
            # Indices l such that 0 <= k-l < N  =>  k-N+1 <= l <= k
            # Also 0 <= l < N.
            # So l in [max(0, k-N+1), min(N-1, k)]
            
            l_min = max(0, k - N + 1)
            l_max = min(N - 1, k)
            
            if l_min <= l_max:
                # h indices are k - l
                # l = k - h_idx => h_idx = k - l
                # As l goes from l_min to l_max, h_idx goes from k-l_min to k-l_max
                # k - l_min = k - max(0, k-N+1)
                # k - l_max = k - min(N-1, k)
                
                h_start = k - l_max
                h_end = k - l_min + 1
                
                # grad[l] += 2 * h[h_idx]
                # We need to map h_idx to l.
                # l = k - h_idx.
                # So grad[k - h_idx] += 2 * h[h_idx]
                
                # Vectorized:
                # h_vals = h[h_start:h_end]
                # l_indices = k - np.arange(h_start, h_end)
                # grad[l_indices] += 2 * h_vals
                
                h_vals = h[h_start:h_end]
                l_indices = k - np.arange(h_start, h_end)
                grad[l_indices] += 2 * h_vals
        
        # Normalize by top_k
        grad /= top_k
        return grad, c

    # Optimization Loop
    inits = []
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal
    for sigma in [N/10, N/8, N/6]:
        for alpha in [0.1, 0.2, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
            
    # W-shape
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))

    best_h = None
    best_val = np.inf
    
    for init in inits:
        if time_left() < 10:
            break
        h = init / np.sum(init)
        c = compute_autoconv(h)
        val = np.max(c)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = np.max(compute_autoconv(best_h))
        
    # Subgradient Descent
    current_h = best_h.copy()
    current_val = best_val
    
    # Adaptive step size
    alpha = 0.01
    min_alpha = 1e-5
    max_alpha = 0.1
    
    n_iter = 0
    while time_left() > 10:
        grad, c = subgradient_step(current_h, None, top_k=20)
        
        # Check if gradient is zero
        if np.allclose(grad, 0):
            break
            
        # Line search or fixed step with backtracking
        # Try alpha
        new_h = current_h - alpha * grad
        new_h = np.maximum(new_h, 0)
        s = np.sum(new_h)
        if s < 1e-10:
            alpha *= 0.5
            continue
        new_h /= s
        
        new_c = compute_autoconv(new_h)
        new_val = np.max(new_c)
        
        if new_val < current_val:
            current_h = new_h
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
            # Increase alpha slightly
            alpha = min(alpha * 1.1, max_alpha)
        else:
            # Decrease alpha
            alpha *= 0.8
            if alpha < min_alpha:
                # Try smaller top_k or break
                break
                
        n_iter += 1
        if n_iter % 100 == 0:
            # Restart or re-normalize
            pass

    return best_h.tolist()

construct()
