import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # Initialize
    # Try a few random starts if time permits? Or just one good one.
    # Let's try random with mean 1 to keep sum positive.
    x = np.random.randn(N) + 1.0
    x = x / np.sum(x)
    
    best_x = x.copy()
    best_val = np.inf
    
    alpha = 50.0 # Soft max parameter
    eta = 0.1    # Learning rate
    
    # Precompute Toeplitz matrix? It changes because weights change.
    # But we can compute the gradient efficiently.
    
    for i in range(10000):
        if time.time() - start_time > deadline:
            break
            
        # Compute autoconvolution
        y = np.convolve(x, x, 'full')
        
        # Compute objective value (raw max abs)
        max_abs = np.max(np.abs(y))
        # Since sum is 1, value is max_abs
        if max_abs < best_val:
            best_val = max_abs
            best_x = x.copy()
            
        # Compute weights for soft-max
        # Use y^2 to handle signs, or just abs?
        # We want to minimize max |y|.
        # Surrogate: sum exp(alpha * y^2)
        # Gradient of log sum exp(alpha y^2)
        # d/dx log(sum exp(alpha y_k^2)) = (sum 2 alpha y_k exp(alpha y_k^2) dy_k/dx) / (sum exp(alpha y_k^2))
        
        # To avoid overflow, subtract max
        y2 = y**2
        max_y2 = np.max(y2)
        exp_terms = np.exp(alpha * (y2 - max_y2))
        sum_exp = np.sum(exp_terms)
        
        # Weights w_k = 2 * alpha * y_k * exp_terms_k / sum_exp
        # Note: The gradient of the soft-max of y^2 involves y_k.
        # If we just want to push down the largest |y_k|, we can use weights proportional to exp(alpha |y_k|) or similar.
        # Let's stick to the gradient of the smooth approximation of max(y^2).
        # This minimizes the "energy" of the peaks.
        
        w = 2 * alpha * y * exp_terms / sum_exp
        
        # Compute gradient g = sum_k w_k * dy_k/dx
        # (g)_i = sum_k w_k * 2 * x_{k-i} (with bounds)
        # (g)_i = 2 * sum_j x_j * w_{i+j}
        
        # Construct the Toeplitz-like multiplication
        # w has length 2N-1
        # We need for i in 0..N-1: sum_{j=0}^{N-1} w_{i+j} x_j
        # This is a sliding dot product.
        
        # Using FFT for speed?
        # Or just np.correlate?
        # np.correlate(x, w, 'full')
        # x length N, w length 2N-1.
        # Result length N + 2N - 2 = 3N - 2.
        # The part we want corresponds to lags?
        # Let's just use a loop for N=400, it's fast enough.
        # Or matrix multiply.
        
        # M[i, j] = w[i+j]
        # We can construct M efficiently?
        # Or just use scipy.signal.oaconvolve? No, standard lib + numpy/scipy.
        # scipy.signal.correlate is available.
        
        # Let's try vectorized approach:
        # g = 2 * np.correlate(x, w, 'full')[N-1 : 2*N-2]?
        # Let's check indices.
        # Correlation c_k = sum_j x_j w_{j+k}.
        # We want sum_j x_j w_{i+j}. This is c_i.
        # So we need the first N elements of the correlation?
        # np.correlate(in1, in2, mode='full')
        # in1=x, in2=w.
        # The output is centered.
        # It's safer to just do the matrix multiply if N is small.
        
        # M = np.zeros((N, N))
        # for i in range(N):
        #   M[i, :] = w[i : i+N]
        # g = 2 * (M @ x)
        
        # This is O(N^2) = 160k ops. 10000 iterations = 1.6e9 ops. Might be slow in Python loop.
        # Let's use FFT.
        # g_i = 2 sum_j x_j w_{i+j}
        # This is the convolution of x and w (reversed?)
        # Let w_rev = w[::-1].
        # (x * w_rev)_i = sum_j x_j w_rev_{i-j} = sum_j x_j w_{2N-2 - (i-j)} = sum_j x_j w_{2N-2-i+j}
        # Not quite.
        
        # Let's stick to the matrix multiplication but optimize it or use a different method.
        # Actually, since w changes every step, we can't precompute M.
        # But N=400 is small.
        # Let's use np.einsum or similar?
        # Or just a simple loop might be okay if we reduce iterations.
        # 120 seconds.
        # If 1 iter takes 1ms, 120s = 120,000 iters.
        # 1 iter with Python loop for gradient: 400 steps. 400 * 400 = 160k ops.
        # 160k ops in Python is ~10-20ms?
        # 120,000 / 10ms = 12,000 iters.
        # That's enough.
        
        # Let's implement the gradient calculation with a loop for safety and clarity, or try to vectorize.
        # Vectorized:
        # Indices i from 0 to N-1. j from 0 to N-1.
        # We need w[i+j].
        # We can create a matrix of indices.
        
        # Better: Use scipy.signal.correlate if available?
        # "Python standard library, NumPy and SciPy are available."
        
        # g = 2 * np.correlate(x, w, mode='full')
        # Let's verify the slice.
        # x = [x0, x1]
        # w = [w0, w1, w2]
        # correlate(x, w, 'full')
        # k=0: x0 w2
        # k=1: x0 w1 + x1 w2
        # k=2: x0 w0 + x1 w1
        # k=3: x1 w0
        # We want for i=0: sum_j x_j w_{0+j} = x0 w0 + x1 w1. This is index 2.
        # For i=1: sum_j x_j w_{1+j} = x0 w1 + x1 w2. This is index 1.
        # So the order is reversed?
        # We want g_i corresponding to sum_j x_j w_{i+j}.
        # This is the cross-correlation at lag i?
        # Standard definition: (x * w)(t) = sum x(t) w(t+tau).
        # Let's just use the matrix multiplication with a pre-allocated buffer or similar.
        
        # Actually, we can compute the gradient using FFT.
        # g_i = 2 sum_j x_j w_{i+j}
        # This is the coefficient of the product of X(z) and W(z^{-1})?
        # Let's just use the simple loop for now, it's robust.
        
        grad = np.zeros(N)
        for i in range(N):
            # grad[i] = 2 * sum(x[j] * w[i+j] for j in range(N))
            # w indices: i to i+N-1
            # Ensure i+N-1 < 2N-1 (len w). Max i=N-1, max index 2N-2. OK.
            grad[i] = 2.0 * np.dot(x, w[i:i+N])
            
        # Update
        x = x - eta * grad
        
        # Normalize
        s = np.sum(x)
        if np.abs(s) < 1e-6:
             # Restart or adjust
             x = np.random.randn(N) + 1.0
             x = x / np.sum(x)
             continue
        x = x / s
        
        # Adaptive learning rate?
        # If max_abs increased, reduce eta.
        
    return best_x
