"""Program for minimizing the normalized maximum of autoconvolution of a step function."""
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Define the autoconvolution (circular) via FFT for speed
    def autoconv(h):
        # h is a 1D array of length N
        # Circular convolution: (h * h)(k) = sum_i h[i] h[k-i mod N]
        H = np.fft.fft(h)
        C = np.fft.ifft(H * H)
        return C.real

    def optimize(h0, steps=200, beta_start=10.0, beta_end=1000.0, lr_start=0.1, lr_end=0.001):
        h = h0.copy()
        h = h / np.sum(h) # normalize
        
        # Precompute the shift matrix indices for efficiency? 
        # Or just use FFT for the gradient part?
        # grad[i] = sum_k w_k h[k-i]
        # This is the circular correlation of w and h?
        # Let's use the matrix approach for clarity and speed since N is small.
        # M[k, i] = h[(k-i) % N]
        # grad = w @ M
        
        for step in range(steps):
            # Anneal beta and learning rate
            t = step / steps
            beta = beta_start + (beta_end - beta_start) * t
            lr = lr_start * (lr_end / lr_start) ** t
            
            C = autoconv(h)
            
            # Soft-max weights
            # To avoid overflow, subtract max
            C_max = np.max(C)
            exp_C = np.exp(beta * (C - C_max))
            sum_exp = np.sum(exp_C)
            w = exp_C / sum_exp
            
            # Compute Gradient
            # grad[i] = sum_k w_k * h[(k-i) % N]
            # We can construct the matrix M where M[k,i] = h[(k-i) % N]
            # M is a circulant matrix defined by h.
            # Specifically, M is the matrix that performs circular convolution if multiplied by a vector?
            # No, (h * v)_i = sum_k h_k v_{i-k}.
            # We want sum_k w_k h_{k-i}.
            # Let v = w. We want (h * w)_i? No.
            # (h * w)_i = sum_k h_k w_{i-k}.
            # We have sum_k w_k h_{k-i}.
            # If we reverse h, let h_rev[k] = h[-k].
            # Then (h_rev * w)_i = sum_k h_rev[k] w_{i-k} = sum_k h[-k] w_{i-k}.
            # Let j = -k. sum_j h[j] w_{i+j}.
            # This is close.
            
            # Let's just use the explicit matrix for N=600. It's fast enough.
            # indices = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
            # M = h[indices]
            # grad = w @ M
            
            # Optimization: 
            # The operation `w @ M` where M is circulant can be done via FFT?
            # M is the matrix for circular convolution with h?
            # If A is circulant with first column h, then A x = h * x.
            # Here we have w @ M. M is circulant?
            # M[k,i] = h[k-i]. 
            # For fixed i, as k varies, it's h shifted.
            # For fixed k, as i varies, it's h reversed and shifted.
            # M is symmetric if h is symmetric.
            # Actually, M is the matrix that maps h to ...?
            # Let's just compute it. 600x600 matrix multiplication is ~1ms. 200 steps = 200ms. Acceptable.
            
            indices = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
            M = h[indices]
            grad = w @ M
            
            # Update
            h -= lr * grad
            
            # Project to non-negative
            h = np.clip(h, 0, None)
            
            # Normalize
            h_sum = np.sum(h)
            if h_sum > 1e-12:
                h /= h_sum
            else:
                # Fallback to uniform
                h = np.ones(N) / N
                
        return h

    # Initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian (narrow)
    x = np.linspace(-0.5, 0.5, N) # [-1/4, 1/4] scaled to [-0.5, 0.5] for shape? 
    # The problem says intervals on [-1/4, 1/4].
    # The positions are i/N - 1/4.
    # Let's use indices directly.
    idx = np.arange(N)
    center = N / 2
    # Gaussian centered at 0 (which is index 0 or N/2? 
    # The function is defined on [-1/4, 1/4]. 
    # Usually step functions are periodic. 
    # Let's assume the grid is 0..N-1 corresponding to -1/4 .. 1/4 - 1/N.
    # A Gaussian centered at 0 (index 0) wraps around.
    # To avoid wrap-around issues in initialization, center at N//2? 
    # No, the domain is [-1/4, 1/4]. 0 is the center of the domain.
    # Index 0 is -1/4. Index N/2 is 0.
    # So we should center the Gaussian at index N//2.
    
    gauss = np.exp(-((idx - N//2)**2) / (2 * (N/20.0)**2))
    inits.append(gauss)
    
    # 3. Cosine bump centered at N//2
    cos = np.cos(np.pi * (idx - N//2) / (N/2)) ** 2
    cos[idx < 0] = 0 # Just in case, though idx is always >= 0
    # The cosine should be 0 at the edges? 
    # cos(pi * (idx - N/2) / (N/2))
    # At idx=0: cos(-pi/2) = 0.
    # At idx=N/2: cos(0) = 1.
    # At idx=N-1: cos(pi/2) ~ 0.
    # This is a nice bump.
    inits.append(cos)
    
    best_h = None
    best_score = np.inf
    
    for h0 in inits:
        h_opt = optimize(h0, steps=300)
        C = autoconv(h_opt)
        score = np.max(C)
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    # Return the best one
    # The interface expects a sequence of nonnegative step heights.
    # We return best_h.
    return list(best_h)
# EVOLVE-BLOCK-END
