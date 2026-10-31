import numpy as np
import time

N = 400

def autoconv_max_abs(a):
    """Compute max absolute value of autoconvolution of a."""
    # Use FFT for speed
    n = len(a)
    # Pad to next power of 2 or just 2*n-1
    size = 2 * n - 1
    # Use a size that is efficient for FFT
    # np.fft.fftfreq size
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
        
    A = np.fft.fft(a, n=fft_size)
    C = np.fft.ifft(A * A)
    # Take the first size elements (linear convolution)
    c = C[:size]
    return np.max(np.abs(c))

def objective_smooth(a, beta=10.0):
    """
    Compute a smooth approximation of max |a*a| and its gradient.
    a is normalized such that sum(a) = 1.
    """
    n = len(a)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
        
    A = np.fft.fft(a, n=fft_size)
    C = np.fft.ifft(A * A)
    c = C[:size]
    
    # Smooth max of |c|
    # M = (1/beta) * log(sum(exp(beta * |c|)))
    # grad_M = (1/beta) * (sum(exp(beta * |c|) * sign(c) * grad_c) / sum(exp(beta * |c|)))
    
    # To avoid overflow in exp, subtract max
    abs_c = np.abs(c)
    max_abs_c = np.max(abs_c)
    
    if max_abs_c == 0:
        return 0.0, np.zeros_like(c)
        
    shifted = beta * (abs_c - max_abs_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    M = max_abs_c + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of M w.r.t c
    # dM/dc_i = (1/beta) * exp(beta * |c_i|) * sign(c_i) / sum_exp
    # Note: d/dx |x| = sign(x)
    # dM/dc = (exp(beta * |c|) * sign(c)) / (beta * sum_exp)
    
    weights = (exp_shifted * np.sign(c)) / (beta * sum_exp)
    
    # Now compute gradient w.r.t a
    # c = a * a (linear convolution)
    # c_k = sum_j a_j a_{k-j}
    # d c_k / d a_i = a_{k-i} + a_{k-i} ? No.
    # c_k = sum_{j} a_j a_{k-j}
    # d c_k / d a_i = a_{k-i} (from j=i) + a_{k-i} (from k-j=i => j=k-i)
    # Wait, if i != k-i, they are distinct terms.
    # d/d a_i (a_i a_{k-i} + a_{k-i} a_i) = 2 a_{k-i}.
    # If i = k-i (i.e., 2i=k), term is a_i^2, derivative is 2 a_i.
    # So d c_k / d a_i = 2 a_{k-i} (with appropriate bounds).
    
    # We need sum_k weights_k * (2 a_{k-i})
    # This is a convolution of weights with a, scaled by 2.
    
    # Let's compute grad_a
    # grad_a_i = sum_k weights_k * 2 * a_{k-i}
    # This is 2 * (weights * a)_i
    
    # Use FFT for this convolution too
    W = np.fft.fft(weights, n=fft_size)
    A_grad = np.fft.ifft(W * A)
    grad_a = 2.0 * A_grad[:n]
    
    return M, grad_a

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with all ones
    a = np.ones(N)
    # Normalize sum to 1
    a = a / np.sum(a)
    
    best_a = a.copy()
    best_score = autoconv_max_abs(best_a)
    
    # Optimization loop
    eta = 0.01
    beta = 5.0
    
    for step in range(2000):
        if time.time() > deadline - 10:
            break
            
        # Compute objective and gradient
        M, grad_a = objective_smooth(a, beta)
        
        # Update
        a_new = a - eta * grad_a
        
        # Project to sum=1
        s = np.sum(a_new)
        if abs(s) < 1e-10:
            # Fallback
            a_new = a
        else:
            a_new = a_new / s
            
        # Evaluate raw score
        score = autoconv_max_abs(a_new)
        
        if score < best_score:
            best_score = score
            best_a = a_new.copy()
            
        # Anneal beta to make max sharper
        if step % 100 == 0 and beta < 50.0:
            beta *= 1.5
        if step % 100 == 0 and eta > 0.001:
            eta *= 0.9

    return best_a
