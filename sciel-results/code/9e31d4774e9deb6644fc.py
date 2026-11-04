import numpy as np
import time

N = 400

def _compute_conv(x):
    """Compute linear autoconvolution using FFT."""
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    C = np.fft.irfft(X * X, n=n_fft)
    return C

def _soft_max_gradient(x, beta):
    """
    Compute gradient of the soft-max surrogate for max |c_k|.
    L(x) = (1/beta) * log(sum(exp(beta * |c_k|)))
    """
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    C = np.fft.irfft(X * X, n=n_fft)
    
    # Weights for the soft-max
    # To avoid overflow, subtract max
    abs_c = np.abs(C)
    max_abs_c = np.max(abs_c)
    weights = np.exp(beta * (abs_c - max_abs_c))
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return np.zeros(N)
    weights /= sum_weights
    
    # Gradient of L(x) w.r.t x
    # d/dx (1/beta * log sum exp(beta |c_k|)) 
    # = sum_k weights_k * d/dx |c_k|
    # Note: d/dx |c_k| = sign(c_k) * d/dx c_k
    
    grad = np.zeros(N)
    
    # Efficient gradient computation using FFT
    # c_k is the k-th element of the linear convolution.
    # The gradient of the k-th convolution element with respect to x is:
    # g_k = IFFT( FFT(x) * conj(FFT(x)) ... wait, c is x*x not x*conj(x).
    # c_k = sum_i x_i x_{i-k}
    # In frequency domain, C = X * X.
    # dC/dX = 2X.
    # The gradient of c_k w.r.t x is the k-th shift of the "gradient kernel".
    # Specifically, if we define G_k = IFFT(2 * X * exp(i * 2 * pi * k * f / n_fft)),
    # then grad = sum_k weights_k * sign(c_k) * G_k
    
    # We can compute all G_k at once? 
    # G_k is the linear cross-correlation of x with itself, shifted?
    # Actually, d c_k / d x_j = x_{j-k} + x_{j+k} (circularly wrapped if using circular conv, but we use linear).
    # Using FFT for linear conv:
    # We can compute the gradient for each k in the sum, but that's O(N^2).
    # Better: 
    # grad = sum_k weights_k * sign(c_k) * (IFFT(2 * X * exp(i 2 pi k f / n_fft))[:N])
    
    # Let's compute the "gradient kernel" in frequency domain.
    # Let W_k = weights_k * sign(c_k).
    # We need sum_k W_k * IFFT(2 * X * exp(i 2 pi k f / n_fft))[:N]
    # = IFFT( 2 * X * sum_k W_k * exp(i 2 pi k f / n_fft) )[:N]
    
    # Let S(f) = sum_k W_k * exp(i 2 pi k f / n_fft).
    # This is the DFT of the sequence W_k (padded to n_fft).
    
    W_full = np.zeros(n_fft)
    W_full[:2*N] = weights * np.sign(C)
    # Note: C is real, length n_fft.
    # W_full is the sequence of weights * signs.
    
    S = np.fft.rfft(W_full)
    
    # Gradient in frequency domain:
    Grad_freq = 2 * X * S
    grad = np.fft.irfft(Grad_freq, n=n_fft)[:N]
    
    return grad

def _project_sum_1(x):
    """Project x onto the hyperplane sum(x) = 1."""
    return x - (np.sum(x) - 1.0) / N

def _objective_value(x):
    """Compute the true objective: max |c_k| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    c = _compute_conv(x)
    return np.max(np.abs(c)) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    best_x = np.ones(N)
    best_score = _objective_value(best_x)
    
    # Strategy: Projected Gradient Descent with momentum
    # Minimize soft-max autoconv peak subject to sum(x)=1
    
    rng = np.random.default_rng(42)
    
    # Try a few different initializations and beta schedules
    inits = []
    # 1. Random Gaussian
    inits.append(rng.standard_normal(N))
    # 2. Random Signs
    inits.append(rng.choice([-1, 1], size=N))
    # 3. Smoothed Random (low pass)
    x_smooth = rng.standard_normal(N)
    # Simple low pass: average with neighbors
    x_smooth = np.convolve(x_smooth, np.ones(5)/5, mode='same')
    inits.append(x_smooth)
    # 4. Chirp
    t = np.linspace(0, 10, N)
    inits.append(np.sin(2 * np.pi * t))
    
    # Beta schedule: start low, increase
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for init in inits:
        x = _project_sum_1(init.copy())
        
        for beta in betas:
            if time.time() > deadline:
                return best_x
            
            # Run PGD for a fixed number of steps
            steps = 200
            lr = 1.0 / (beta * N) # Heuristic learning rate
            momentum = 0.9
            v = np.zeros(N)
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                grad = _soft_max_gradient(x, beta)
                
                # Momentum update
                v = momentum * v - lr * grad
                x_new = x + v
                x_new = _project_sum_1(x_new)
                
                x = x_new
            
            # Evaluate true objective
            score = _objective_value(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
                
            # Optional: Refine with higher beta if improving
            # The loop over betas handles this by restarting PGD with higher beta
            # using the current x as initialization for the next beta phase?
            # Yes, x is carried over.

    # Final polish: Simple local search on the best found
    # Try flipping small subsets? No, continuous.
    # Try a few more PGD steps with high beta
    x = best_x.copy()
    beta = 100.0
    steps = 100
    lr = 1.0 / (beta * N)
    momentum = 0.9
    v = np.zeros(N)
    
    for step in range(steps):
        if time.time() > deadline:
            break
        grad = _soft_max_gradient(x, beta)
        v = momentum * v - lr * grad
        x_new = x + v
        x_new = _project_sum_1(x_new)
        x = x_new
        
    score = _objective_value(x)
    if score < best_score:
        best_score = score
        best_x = x.copy()

    return best_x
