import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Objective function parameters
    beta = 20.0  # Sharpness of log-sum-exp
    
    def compute_score(h):
        # h is normalized to sum 1
        # Compute autoconvolution
        H = np.fft.fft(h, n=2*N)
        G = H * H
        g = np.real(np.fft.ifft(G))
        # g has length 2*N. The relevant part is roughly the middle
        # We take the max over the whole array to be safe, or just the support
        # The autoconvolution of a function supported on [0, N-1] is supported on [0, 2N-2]
        # We can just take max of g
        max_g = np.max(g)
        # Sum h is 1
        return max_g
        
    def grad_score(h):
        H = np.fft.fft(h, n=2*N)
        G = H * H
        g = np.real(np.fft.ifft(G))
        
        # Compute weights w_k = exp(beta * g_k) / sum(exp(beta * g_k))
        # To avoid overflow, subtract max
        g_max = np.max(g)
        exp_g = np.exp(beta * (g - g_max))
        w = exp_g / np.sum(exp_g)
        
        # Gradient is 2 * (w * h)
        W = np.fft.fft(w, n=2*N)
        grad_fft = 2.0 * W * H
        grad = np.real(np.fft.ifft(grad_fft))
        
        # The gradient is defined on the same grid as h (padded)
        # We only care about the first N elements corresponding to the support
        # Wait, the convolution w*h produces a vector of length 2N.
        # The gradient wrt h_i (i in 0..N-1) is the sum of contributions.
        # The FFT approach gives the full convolution.
        # We need to extract the part corresponding to h.
        # Actually, the derivative of sum(w_k g_k) wrt h_i is (w*h)_i + (w_rev * h)_i?
        # If w is symmetric, it's 2(w*h)_i.
        # The vector grad from IFFT is the linear convolution of w and h.
        # We need the values at indices 0..N-1.
        # However, due to circular convolution in FFT, we need to be careful.
        # If we zero-pad to 2N, the linear convolution is contained in the first 2N-1 samples.
        # The gradient for h[0..N-1] corresponds to the first N samples of the linear convolution?
        # Let's check.
        # g_k = sum_j h_j h_{k-j}.
        # d/dh_i sum_k w_k g_k = sum_k w_k (h_{k-i} + h_{i-k}).
        # This is indeed the linear convolution of w and h, evaluated at i.
        # So we take grad[0:N].
        return grad[:N]

    best_h = None
    best_score = float('inf')
    
    # Initializations
    # 1. Gaussian
    x = np.linspace(-1, 1, N)
    sigma = 0.5
    h_gauss = np.exp(-x**2 / (2*sigma**2))
    h_gauss /= np.sum(h_gauss)
    
    # 2. Uniform
    h_unif = np.ones(N) / N
    
    # 3. Random
    np.random.seed(42)
    h_rand = np.random.rand(N)
    h_rand /= np.sum(h_rand)
    
    initializations = [h_gauss, h_unif, h_rand]
    
    for h_init in initializations:
        h = h_init.copy()
        eta = 0.1
        for i in range(10000):
            if time.time() > deadline:
                break
                
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            grad = grad_score(h)
            
            # Gradient Descent Step
            h_new = h - eta * grad
            
            # Project to non-negative
            h_new[h_new < 0] = 0
            
            # Normalize
            s = np.sum(h_new)
            if s < 1e-10:
                break
            h_new /= s
            
            # Check for convergence or oscillation
            if np.max(np.abs(h_new - h)) < 1e-8:
                h = h_new
                break
            else:
                h = h_new
                
            # Adaptive learning rate?
            # If score increased, reduce eta
            # This is a simple heuristic
            new_score = compute_score(h)
            if new_score > score:
                eta *= 0.5
            elif i % 100 == 0:
                eta *= 1.1 # Slowly increase?
        
        if time.time() > deadline:
            break

    if best_h is None:
        return [1.0] * N
        
    return list(best_h)
