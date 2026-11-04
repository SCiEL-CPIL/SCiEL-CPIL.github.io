import numpy as np
import time

N = 600
DEADLINE = 150.0
START = time.time()

def time_left():
    return DEADLINE - (time.time() - START)

def autoconv_max(h):
    # h: array of N nonnegative heights
    # discrete autoconvolution: c[k] = sum_i h[i] * h[i+k] for k in -N..N
    # Use FFT for efficiency
    n = 2 * N
    # zero-pad
    h_pad = np.zeros(n)
    h_pad[:N] = h
    fft = np.fft.rfft(h_pad, n=2*n)
    conv = np.fft.irfft(fft * fft, n=2*n)
    # conv has length 2n; the relevant part is conv[0:N] (positive lags including 0)
    # Actually full linear conv length is 2N-1. With padding 2n it's fine.
    return np.max(conv)

def soft_max(x, beta=1.0):
    # log-sum-exp based soft max
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def construct():
    global START, DEADLINE
    # Ensure we start fresh timing per call? The evaluator calls construct once.
    # But to be safe, set deadline relative to module import.
    
    # Initialize: try a few starting points
    # Nonnegative via y = log(h) or h = exp(y). Use h = softplus or exp.
    # Start with uniform
    best_h = np.ones(N)
    best_val = autoconv_max(best_h)
    
    # We want to minimize max of autoconv. 
    # Use gradient descent on y where h = exp(y) (ensures nonneg)
    # Or h = y^2. Use exp for smoothness.
    
    y = np.zeros(N)  # h = exp(y) = 1 initially
    h = np.exp(y)
    
    # Compute gradient of soft_max(autoconv) w.r.t. h, then w.r.t. y
    beta = 10.0  # sharpness of soft max
    
    # Precompute FFT size
    n_fft = 1  # we'll use 2*N
    size = 2 * N
    
    def compute_conv(h):
        h_pad = np.zeros(size)
        h_pad[:N] = h
        fft = np.fft.rfft(h_pad, n=2*size)
        conv = np.fft.irfft(fft * fft, n=2*size)
        return conv
    
    def obj_and_grad(h, beta):
        conv = compute_conv(h)
        # soft max of conv
        cmax = np.max(conv)
        exps = np.exp(beta * (conv - cmax))
        smax = cmax + np.log(np.sum(exps)) / beta
        # grad of soft max w.r.t. conv
        grad_conv = exps / np.sum(exps)
        # grad of conv w.r.t. h: conv[k] = sum_i h[i]h[i+k]
        # This is the linear convolution. grad conv w.r.t. h is the "deconvolution" 
        # Actually d(conv)/d(h[j]) : conv is quadratic in h.
        # conv_full = ifft(|fft(h_pad)|^2). 
        # d(conv)/d(h) can be computed via: grad_h = 2 * ifft(fft(h_pad) * conj(fft(h_pad)))? 
        # Let's think: conv = h_pad * h_pad (linear). 
        # Using FFT: F = fft(h_pad). conv = ifft(F*F).
        # d(conv)/d(h_pad) = 2 * ifft(F * conj(F)) ? No.
        # Actually d/dh [ifft(F(h)^2)] where F = fft(h). 
        # d(conv)/d(h) = 2 * ifft(fft(h) * fft(h)) in circular sense, but we have linear with padding.
        # Simpler: use the fact that conv = h_pad ⊛ h_pad. The gradient is 2*(h_pad ⊛ h_pad) but flipped... 
        # Let me just use: grad_h_pad = 2 * np.convolve(h_pad, h_pad, mode='full')? No that's conv itself.
        # The gradient of (a⊛a) w.r.t. a is 2*(a⊛a) but with the right alignment. Actually d(a⊛a)/da = 2*(a⊛a) where the second a is the same, so it's 2 * convolve(a, a) but that's the conv value, not gradient.
        # Correct: for f(a) = sum_k (sum_i a_i a_{k-i})^2 ... no we want soft max of the conv values, not sum of squares.
        # Let me just compute grad numerically or use the property:
        # conv = ifft(F^2). d(conv)/d(h) : by chain rule through FFT.
        # d(conv)/d(F[m]) = 2 F[m] * ifft_basis... 
        # This is getting complex. Let me use a simpler approach: 
        # grad of soft_max(conv) w.r.t. h = sum_k grad_conv[k] * d(conv[k])/d(h)
        # d(conv[k])/d(h[j]) = h[j+k] + h[j-k] (with bounds) for the linear conv.
        # So grad_h[j] = sum_k grad_conv[k] * (h[j+k] + h[j-k])
        # This is a convolution of grad_conv with h (plus flipped).
        # grad_h = convolve(grad_conv, h, 'full') + convolve(grad_conv[::-1], h, 'full')? 
        # Let me just do: grad_h = np.convolve(grad_conv, h, mode='same') + np.convolve(grad_conv, h[::-1], mode='same')
        # Hmm, let me verify dimensions.
        
        # Actually, the cleanest: use the FFT gradient.
        # conv = ifft(F * F), F = fft(h_pad)
        # d(conv)/d(h_pad) = 2 * ifft(F * conj(F))? Let me test with a small case mentally.
        # For circular conv it's clean. For linear with enough padding it approximates.
        # Let me use: grad_h_pad = 2 * np.fft.irfft(fft * np.conj(fft), n=2*size)
        # Wait that gives |F|^2 which is the autocorrelation, related but not the gradient of conv.
        
        # I'll compute gradient via the direct formula using convolution.
        # conv[k] = sum_i h[i]*h[i-k] (linear, k from -(N-1) to N-1)
        # d(conv[k])/d(h[j]) = h[j-k] + h[j+k] (when in bounds, 0 otherwise)
        # So grad_h[j] = sum_k grad_conv[k] * (h[j-k] + h[j+k])
        # = (grad_conv * h)[j] + (grad_conv * h_flipped)[j] where * is appropriate conv
        # Let me pad and use np.convolve.
        
        gc = grad_conv  # length = size (the conv array)
        # We need conv of gc with h and with h reversed.
        # h has length N. 
        # term1[j] = sum_k gc[k] * h[j-k]  -> this is (gc * h)[j] in cross-correlation sense
        # Let me just build it via FFT too for speed.
        # grad_h = ifft( fft(gc) * (fft(h) + fft(h_flipped)) )? 
        # This is getting too complex. Let me use a numerical-free direct approach with np.convolve on padded arrays.
        
        # Simpler: since N=600, we can afford O(N^2) if needed but FFT is better.
        # Let me just compute grad_h by: 
        # grad_h = np.zeros(N)
        # For each j, grad_h[j] = sum_k gc[k]*(h[j-k]+h[j+k])
        # This is 2 * cross-correlation. Use FFT.
        
        # Build gc padded to size
        gc_pad = np.zeros(size)
        # gc is the conv array of length 2*size from irfft. The meaningful part is gc[0:size] roughly.
        # Let me just use the full gc array.
        gc_full = gc  # length 2*size
        # h_pad
        h_pad = np.zeros(size)
        h_pad[:N] = h
        # cross-correlation: sum_k gc[k] h[j-k] = (gc * h)[j]
        # Use FFT: 
        F_gc = np.fft.rfft(gc_full, n=4*size)
        F_h = np.fft.rfft(h_pad, n=4*size)
        # cross corr = ifft(F_gc * conj(F_h))
        term1 = np.fft.irfft(F_gc * np.conj(F_h), n=4*size)
        # term2: sum_k gc[k] h[j+k] = sum_k gc[k] h_flipped[j-k] where h_flipped = h reversed
        h_flip_pad = np.zeros(size)
        h_flip_pad[:N] = h[::-1]
        F_hf = np.fft.rfft(h_flip_pad, n=4*size)
        term2 = np.fft.irfft(F_gc * np.conj(F_hf), n=4*size)
        grad_h_pad = (term1 + term2)[:size]
        grad_h = grad_h_pad[:N]
        
        # Now grad w.r.t. y: h = exp(y), dh/dy = h
        grad_y = grad_h * h
        return smax, grad_y
    
    # Gradient descent
    lr = 0.01
    max_iter = 5000
    patience = 200
    no_improve = 0
    
    for it in range(max_iter):
        if time_left() < 5.0:
            break
        val, gy = obj_and_grad(h, beta)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > patience and it > 100:
            # reduce lr
            lr *= 0.5
            no_improve = 0
            if lr < 1e-6:
                break
        y -= lr * gy
        h = np.exp(y)
        # Ensure nonneg (exp already ensures)
    
    # Final polish: a few more iterations with smaller lr
    lr = 1e-4
    for it in range(500):
        if time_left() < 3.0:
            break
        val, gy = obj_and_grad(h, beta)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        y -= lr * gy
        h = np.exp(y)
    
    # Return the best
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    return list(best_h)
