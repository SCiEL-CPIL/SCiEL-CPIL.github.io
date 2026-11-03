import numpy as np
import time

N = 100

def compute_max_overlap(h):
    """Compute max_k C(k) where C(k) = sum_i h_i * (1 - h_{(i+k) mod N})."""
    # C(k) = N/2 - autocorr(k), where autocorr(k) = sum_i h_i * h_{(i-k) mod N}
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h_i * h_{(i-k) mod N}
    # C(k) = sum_i h_i * (1 - h_{(i+k) mod N}) = N/2 - sum_i h_i * h_{(i+k) mod N}
    # Note: autocorr[k] = sum_i h_i * h_{(i-k) mod N} = sum_i h_{i+k} * h_i = sum_i h_i * h_{(i-k) mod N}
    # So C(k) = N/2 - autocorr[k]
    max_overlap = N / 2.0 - np.min(autocorr)
    return max_overlap

def construct():
    """
    Construct step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Combines alternating pattern initialization with FFT-based gradient descent
    using log-sum-exp smoothing for the max function.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Start from alternating pattern (slightly better than uniform in practice)
    a = (N - 2) / (2 * (N - 1))
    h = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            h[i] = a
        else:
            h[i] = 1 - a
    
    best_h = h.copy()
    best_score = compute_max_overlap(best_h)
    
    # Also try uniform as baseline
    h_uniform = np.full(N, 0.5)
    uniform_score = compute_max_overlap(h_uniform)
    if uniform_score < best_score:
        best_h = h_uniform.copy()
        best_score = uniform_score
    
    # Log-Sum-Exp parameters for smooth max approximation
    beta = 50.0
    lr = 0.01
    max_iter = 3000
    
    def logsumexp_min(x, b):
        """Smooth approximation of min(x) using log-sum-exp."""
        m = np.min(x)
        shifted = -b * (x - m)  # Negative because we want min
        shifted = np.clip(shifted, -700, 700)
        return m - np.log(np.sum(np.exp(shifted))) / b
    
    def gradient_logsumexp_min(x, b):
        """Gradient (softmax weights) of log-sum-exp approximation of min."""
        m = np.min(x)
        shifted = -b * (x - m)
        shifted = np.clip(shifted, -700, 700)
        weights = np.exp(shifted)
        s = np.sum(weights)
        if s > 0:
            weights /= s
        else:
            weights = np.full_like(x, 1.0 / len(x))
        return weights
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Compute autocorrelation via FFT
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        
        # We want to maximize min(autocorr), which is equivalent to minimizing -min(autocorr)
        # Using log-sum-exp to approximate -min(autocorr)
        neg_autocorr = -autocorr
        c_min_smooth = logsumexp_min(autocorr, beta)
        grad_weights = gradient_logsumexp_min(autocorr, beta)
        
        # Gradient of autocorr[k] with respect to h_j:
        # autocorr[k] = sum_i h_i * h_{(i-k) mod N}
        # d(autocorr[k])/d(h_j) = h_{(j+k) mod N} + h_{(j-k) mod N}
        # 
        # We want to maximize min(autocorr), so gradient of -min(autocorr) is:
        # grad_h_j = -sum_k grad_weights[k] * [h_{(j+k) mod N} + h_{(j-k) mod N}]
        
        G = np.fft.fft(grad_weights)
        H_fft = np.fft.fft(h)
        
        # term1: sum_k grad_weights[k] * h_{(j+k) mod N} = circular cross-correlation
        # corr[j] = sum_k grad_weights[k] * h[(j+k) mod N]
        # FFT: ifft(conj(fft(grad_weights)) * fft(h))
        term1 = np.fft.ifft(np.conj(G) * H_fft).real
        
        # term2: sum_k grad_weights[k] * h_{(j-k) mod N} = circular convolution
        # conv[j] = sum_k grad_weights[k] * h[(j-k) mod N]
        # FFT: ifft(fft(grad_weights) * fft(h))
        term2 = np.fft.ifft(G * H_fft).real
        
        # Gradient of -min(autocorr) with respect to h_j
        grad_h = -(term1 + term2)
        
        # Gradient ascent step (we want to maximize min autocorr)
        h_new = h + lr * grad_h
        
        # Project to [0, 1] and adjust sum to N/2
        h_new = np.clip(h_new, 0.0, 1.0)
        current_sum = np.sum(h_new)
        target_sum = N / 2.0
        if abs(current_sum - target_sum) > 1e-12:
            diff = target_sum - current_sum
            if diff > 0:
                room = 1.0 - h_new
                if np.sum(room) > 1e-12:
                    h_new += diff * room / np.sum(room)
            else:
                if np.sum(h_new) > 1e-12:
                    h_new += diff * h_new / np.sum(h_new)
            h_new = np.clip(h_new, 0.0, 1.0)
        
        # Check if improvement
        score = compute_max_overlap(h_new)
        if score < best_score - 1e-12:
            best_score = score
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate decay
        if it % 200 == 0 and it > 0:
            lr *= 0.92
    
    # Final projection to ensure constraints are satisfied
    h_final = np.clip(best_h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h_final)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = 1.0 - h_final
            if np.sum(room) > 1e-12:
                h_final += diff * room / np.sum(room)
            else:
                h_final += diff / N
        else:
            if np.sum(h_final) > 1e-12:
                h_final += diff * h_final / np.sum(h_final)
            else:
                h_final += diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
    
    # Verify the final result
    final_sum = np.sum(h_final)
    if abs(final_sum - N / 2.0) > 1e-9:
        # Fallback to uniform if projection failed
        h_final = np.full(N, 0.5)
    
    return h_final
