import numpy as np
import time

N = 100

def compute_max_overlap(h):
    """Compute max_k C(k) where C(k) = sum_i h_i * (1 - h_{(i+k) mod N})."""
    h_comp = 1.0 - h
    # Use FFT for circular cross-correlation
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    # C(k) = sum_i h_i * h_comp_{(i+k) mod N} = circular cross-correlation
    # This is the inverse FFT of H * conj(Hc)
    c = np.fft.ifft(H * np.conj(Hc)).real
    return np.max(c)

def construct():
    """
    Construct step heights h to minimize the maximum overlap of h with shifted 1-h.
    Uses gradient descent on a Log-Sum-Exp smooth approximation of the max overlap.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish within 120s, keep safety margin
    
    # Start from uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = compute_max_overlap(best_h)
    
    # Log-Sum-Exp parameters
    beta = 100.0  # temperature for smooth max approximation
    lr = 0.01
    max_iter = 5000
    
    def logsumexp_max(x, b):
        """Smooth approximation of max(x) using log-sum-exp."""
        m = np.max(x)
        if m > 0:
            return m + np.log(np.sum(np.exp(b * (x - m)))) / b
        else:
            return np.log(np.sum(np.exp(b * x))) / b
    
    def gradient_logsumexp_max(x, b):
        """Gradient of log-sum-exp approximation of max."""
        m = np.max(x)
        if m > 0:
            weights = np.exp(b * (x - m))
            weights /= np.sum(weights)
        else:
            weights = np.exp(b * x)
            weights /= np.sum(weights)
        return weights
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Compute overlaps
        h_comp = 1.0 - h
        H = np.fft.fft(h)
        Hc = np.fft.fft(h_comp)
        c = np.fft.ifft(H * np.conj(Hc)).real
        
        # Smooth max and its gradient
        c_max = logsumexp_max(c, beta)
        grad_c = gradient_logsumexp_max(c, beta)
        
        # Gradient of c with respect to h
        # c_k = sum_i h_i * (1 - h_{(i+k) mod N})
        # dc_k/dh_j = (1 - h_{(j+k) mod N}) - h_j * (1 if j==i... )
        # Actually: c_k = sum_i h_i * h_comp_{(i+k) mod N}
        # dc_k/dh_j = h_comp_{(j+k) mod N} + sum_i h_i * dh_comp_{(i+k) mod N}/dh_j
        # dh_comp_m/dh_j = -delta_{m,j}
        # dc_k/dh_j = h_comp_{(j+k) mod N} - h_{(j-k) mod N}
        
        # So grad_c_k * dc_k/dh_j, summed over k
        # grad_h_j = sum_k grad_c_k * (h_comp_{(j+k) mod N} - h_{(j-k) mod N})
        
        # This is a convolution-like operation
        # term1_j = sum_k grad_c_k * h_comp_{(j+k) mod N} = circular cross-correlation of grad_c and h_comp
        # term2_j = sum_k grad_c_k * h_{(j-k) mod N} = circular convolution of grad_c and h
        
        # Use FFT
        G = np.fft.fft(grad_c)
        Hc_fft = np.fft.fft(h_comp)
        H_fft = np.fft.fft(h)
        
        # term1: sum_k grad_c_k * h_comp_{(j+k) mod N}
        # This is the circular cross-correlation: ifft(conj(G) * Hc_fft) ... 
        # Actually, let's think: sum_k a_k * b_{(j+k) mod N}
        # = ifft(fft(a) * conj(fft(b))) ... no
        # Let's use: sum_k a_k * b_{(j+k)} = circular correlation
        # In FFT: ifft(conj(A) * B) gives correlation
        # where A = fft(a), B = fft(b)
        # Actually: corr[j] = sum_k a[k] * b[(j+k) mod N]
        # FFT: corr = ifft(conj(fft(a)) * fft(b))
        
        term1 = np.fft.ifft(np.conj(G) * Hc_fft).real
        # term2: sum_k grad_c_k * h_{(j-k) mod N}
        # = sum_k a_k * b_{(j-k) mod N} = circular convolution
        # conv[j] = sum_k a[k] * b[(j-k) mod N]
        # FFT: conv = ifft(fft(a) * fft(b))
        
        term2 = np.fft.ifft(G * H_fft).real
        
        grad_h = term1 - term2
        
        # Gradient step
        h = h - lr * grad_h
        
        # Project back to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        # Adjust sum to be exactly N/2
        current_sum = np.sum(h)
        target_sum = N / 2.0
        diff = target_sum - current_sum
        h += diff / N
        
        # Re-clip after sum adjustment
        h = np.clip(h, 0.0, 1.0)
        
        # Check score
        score = compute_max_overlap(h)
        if score < best_score - 1e-10:
            best_score = score
            best_h = h.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    # Final projection to ensure constraints
    h = np.clip(best_h, 0.0, 1.0)
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # Final check: ensure sum is as close to N/2 as possible
    # If clipping broke the sum, do a more careful projection
    if abs(np.sum(h) - N/2.0) > 1e-10:
        # Simple iterative correction
        for _ in range(100):
            diff = N/2.0 - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
    
    return h
