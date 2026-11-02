import numpy as np
import time

N = 100

def _max_overlap(h):
    """Compute max overlap: max_k sum_i h_i * (1 - h_{(i+k)%N})"""
    h = np.asarray(h)
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _project(h, target_sum):
    """Project h onto the simplex {h_i in [0,1], sum(h) = target_sum}"""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if s > 0:
        h = h * (target_sum / s)
    h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    """
    Use spectral method to construct step heights minimizing max overlap.
    
    We work in the frequency domain to equalize the autocorrelation values,
    which minimizes the maximum overlap.
    """
    t0 = time.time()
    deadline = t0 + 120.0
    
    # Start with uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _max_overlap(h)
    
    # Try spectral approach: modify in frequency domain
    # The autocorrelation C_k is related to |FFT(h)|^2
    # We want C_k to be as flat as possible
    
    # Initialize with some randomness
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Start from uniform or perturbed
        if restart == 0:
            h = np.full(N, 0.5)
        else:
            h = np.full(N, 0.5)
            noise = np.random.randn(N) * 0.1
            h = h + noise
            h = _project(h, N/2.0)
        
        # Spectral optimization
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Compute current autocorrelation via FFT
            H = np.fft.fft(h)
            power = np.abs(H) ** 2
            C = np.fft.ifft(power).real  # circular autocorrelation
            
            # Overlap at shift k: O_k = N/2 - C_k
            overlaps = N/2.0 - C
            max_ov = np.max(overlaps)
            
            if max_ov < best_score:
                best_score = max_ov
                best_h = h.copy()
            
            # Gradient: we want to increase C_k where C_k is smallest
            # to reduce max overlap
            # d(O_k)/d(h_i) = (1 - h_{(i+k)%N}) - h_i * delta(i, (j-k)%N) for each j
            # Simplify: gradient of max overlap w.r.t. h
            grad = np.zeros(N)
            for k in range(N):
                shifted = np.roll(h, k)
                ov_k = np.sum(h * (1.0 - shifted))
                if ov_k > max_ov - 0.01:  # only consider near-max overlaps
                    # d(ov_k)/d(h_i) = (1 - h_{(i+k)%N}) - h_i * (contribution from shift)
                    d_ov_k = (1.0 - np.roll(h, -k)) - np.roll(h, k)
                    # Wait, let me recompute: ov_k = sum_i h_i (1 - h_{(i+k)%N})
                    # d(ov_k)/d(h_i) = (1 - h_{(i+k)%N}) + h_i * d(1 - h_{(i+k)%N})/d(h_i)
                    # The second term: if we change h_i, it affects h_{(j-k)%N} for j = i+k
                    # So d(ov_k)/d(h_i) = (1 - h_{(i+k)%N}) - h_{(i-k)%N}
                    d_ov_k = (1.0 - np.roll(h, k)) - np.roll(h, -k)
                    grad += d_ov_k
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            # Project gradient to satisfy sum constraint
            grad = grad - np.mean(grad)
            
            # Step size
            lr = 0.01 / (1.0 + step * 0.01)
            h = h - lr * grad
            h = _project(h, N/2.0)
        
        # Local polish
        for polish_step in range(100):
            if time.time() > deadline - 5.0:
                break
            
            current_ov = _max_overlap(h)
            if current_ov < best_score:
                best_score = current_ov
                best_h = h.copy()
            
            # Try small perturbations
            improved = False
            for _ in range(10):
                i, j = np.random.randint(0, N, 2)
                if i == j:
                    continue
                delta = np.random.choice([-0.05, 0.05])
                h_new = h.copy()
                h_new[i] += delta
                h_new[j] -= delta
                h_new = _project(h_new, N/2.0)
                ov_new = _max_overlap(h_new)
                if ov_new < current_ov:
                    h = h_new
                    current_ov = ov_new
                    improved = True
                    if current_ov < best_score:
                        best_score = current_ov
                        best_h = h.copy()
                    break
            if not improved:
                break
    
    return best_h
