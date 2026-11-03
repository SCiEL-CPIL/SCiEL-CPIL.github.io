import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # Initialize with a Hann window, normalized to sum to 1
    x = np.linspace(0, 1, N)
    f = 0.5 * (1 - np.cos(2 * np.pi * x))
    f = np.maximum(f, 1e-10)
    f = f / np.sum(f)
    
    # Parameters
    lambda_smooth = 50.0  # Temperature for log-sum-exp
    lr = 0.01             # Learning rate
    best_val = np.inf
    best_f = f.copy()
    
    # FFT size
    fft_size = 2048
    
    def compute_gradient(f):
        # Compute g = f * f
        F = np.fft.fft(f, n=fft_size)
        G = F * F
        g = np.fft.ifft(G).real[:2*N-1]
        
        # Smooth max of g
        # Use a large lambda for approximation of max
        # To avoid overflow, subtract max
        g_max = np.max(g)
        exp_g = np.exp(lambda_smooth * (g - g_max))
        sum_exp_g = np.sum(exp_g)
        S_g = g_max + np.log(sum_exp_g) / lambda_smooth
        
        # Gradient of S_g w.r.t g
        # dS_g/dg_i = exp(lambda*(g_i - g_max)) / sum_exp * lambda / lambda = exp(...) / sum_exp
        grad_g = exp_g / sum_exp_g
        
        # Gradient of S_g w.r.t f
        # g = f * f
        # dG/dF = 2 F
        # dS_g/df = ifft( 2 * F * fft(grad_g) )
        GG = np.fft.fft(grad_g, n=fft_size)
        grad_F = 2 * F * GG
        grad_f = np.fft.ifft(grad_F).real[:N]
        
        # Smooth max of f
        f_max = np.max(f)
        exp_f = np.exp(lambda_smooth * (f - f_max))
        sum_exp_f = np.sum(exp_f)
        S_f = f_max + np.log(sum_exp_f) / lambda_smooth
        
        # Gradient of S_f w.r.t f
        grad_f_sf = exp_f / sum_exp_f
        
        # Objective J = S_g / S_f
        # dJ/df = (S_f * dS_g/df - S_g * dS_f/df) / S_f^2
        val = S_g / S_f
        grad_J = (S_f * grad_f - S_g * grad_f_sf) / (S_f * S_f)
        
        return val, grad_J

    # Main Loop
    step = 0
    while time.time() < deadline:
        val, grad = compute_gradient(f)
        
        # Update
        f_new = f - lr * grad
        
        # Project to non-negative
        f_new = np.maximum(f_new, 0)
        
        # Normalize to sum 1
        s = np.sum(f_new)
        if s < 1e-10:
            f_new = 1.0 / N * np.ones(N)
            s = 1.0
        f_new = f_new / s
        
        # Check improvement
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
        
        f = f_new
        step += 1
        
        # Adaptive learning rate?
        if step % 100 == 0:
            lr = max(0.001, lr * 0.95)

    return list(best_f)
