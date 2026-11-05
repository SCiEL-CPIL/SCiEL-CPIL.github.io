import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial guess: Random with zero mean perturbation
    # Start with a spread-out distribution to avoid local minima near spikes
    h = np.random.randn(N)
    h -= np.mean(h)
    h += 1.0 / N  # Ensure sum is 1
    
    tau = 1.0
    lr = 0.01
    
    # Adaptive learning rate / tau schedule
    # We want to minimize max(c).
    # Smooth max approximation: tau * log(sum(exp(c/tau)))
    
    best_h = h.copy()
    best_val = np.inf
    
    def compute_value(h):
        c = np.convolve(h, h)
        s = np.sum(h)
        if s == 0:
            return np.inf
        return abs(2 * N * np.max(c) / (s ** 2))
    
    current_val = compute_value(h)
    if current_val < best_val:
        best_val = current_val
        best_h = h.copy()
        
    # Gradient Descent
    # We will use a simple loop
    # To speed up, we can use FFT for convolution and gradient
    # But N=400 is small. np.convolve is C-optimized.
    
    steps = 0
    while time.time() < deadline:
        # Compute autoconvolution
        c = np.convolve(h, h)
        
        # Compute weights for smooth max
        # To avoid overflow, subtract max
        c_max = np.max(c)
        w = np.exp((c - c_max) / tau)
        w_sum = np.sum(w)
        w = w / w_sum
        
        # Gradient of smooth max
        # d/dh_i sum_k w_k c_k
        # = sum_k w_k (h_{k-i} + h_{i-k})
        # This is convolve(w, h) + convolve(flip(w), h)
        # Note: np.convolve(w, h) gives full convolution. We need the part corresponding to h indices.
        # The gradient vector has length N.
        # g = np.convolve(w, h) + np.convolve(w[::-1], h)
        # The result of convolve(w, h) has length 2N-1.
        # The index i in h corresponds to index i in the first half?
        # Let's check indices.
        # c_k = sum_j h_j h_{k-j}.
        # d c_k / d h_i = h_{k-i} + h_{i-k}.
        # Sum_k w_k h_{k-i}. Let m = k-i. Sum_m w_{m+i} h_m.
        # This is the cross-correlation or convolution?
        # (w * h)_i = sum_j w_j h_{i-j}.
        # Let j = i-m. (w * h)_i = sum_m w_{i-m} h_m.
        # We have sum_m w_{m+i} h_m. This is like w reversed?
        # Let v = w[::-1]. (v * h)_i = sum_j v_j h_{i-j} = sum_m w_{N-1-(i-m)} h_m ... complicated.
        
        # Let's use FFT for the gradient step to be safe and fast.
        # F_h = fft(h, n=2*N)
        # F_w = fft(w, n=2*N)
        # grad = ifft(F_w * F_h + F_w_flip * F_h)
        # Where F_w_flip is fft of reversed w?
        # Actually, the gradient of x*x is 2x.
        # Here we have weighted sum.
        # Let's just use the explicit loop for N=400 if it's fast enough.
        # 400*400 = 160k ops per gradient.
        # 10k steps = 1.6e9 ops. In Python, this is too slow.
        # Must use FFT or vectorized numpy.
        
        # Vectorized gradient:
        # g[i] = sum_k w[k] * (h[k-i] + h[i-k])
        # This can be computed as:
        # term1 = np.convolve(w, h)
        # term2 = np.convolve(w, h) # with appropriate slicing
        
        # Let's use FFT.
        n_fft = 2 * N
        F_h = np.fft.fft(h, n=n_fft)
        F_w = np.fft.fft(w, n=n_fft)
        
        # The term sum_k w_k h_{k-i} corresponds to ifft(F_w * F_h) but shifted?
        # Convolution theorem: ifft(fft(a)*fft(b)) = a*b (circular).
        # We are doing linear convolution effectively by zero padding.
        # c = ifft(fft(h)*fft(h)).
        # grad = ifft(fft(w)*fft(h) + fft(w_rev)*fft(h))?
        # Let's verify.
        # d/dh_i (h*h) = h * 1_i + 1_i * h = 2 h (if unweighted).
        # With weights w: sum_k w_k (h_{k-i} + h_{i-k}).
        # Let A = w.
        # sum_k A_k h_{k-i} = (A * h)_{-i}? No.
        # Standard conv: (A*h)_n = sum_k A_k h_{n-k}.
        # We have sum_k A_k h_{k-i}. Let j = k-i. sum_j A_{j+i} h_j.
        # This is (A_shifted * h) where A is shifted?
        
        # Simpler:
        # g = np.convolve(w, h) + np.convolve(w, h) # NO
        # Let's construct the gradient matrix? No, too big.
        
        # Let's use the property:
        # grad = np.fft.ifft(np.fft.fft(w, n=n_fft) * F_h + np.fft.fft(w[::-1], n=n_fft) * F_h).real[:N]
        
        w_rev = w[::-1]
        F_w_rev = np.fft.fft(w_rev, n=n_fft)
        
        grad = np.fft.ifft(F_w * F_h + F_w_rev * F_h).real[:N]
        
        # Update
        h = h - lr * grad
        
        # Project to sum=1
        h = h - (np.sum(h) - 1.0) / N
        
        # Check value
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Schedule
        tau *= 0.99
        lr *= 0.995
        steps += 1
        
        if steps % 100 == 0:
            # Optional: restart or adjust
            pass
            
    return best_h
