import numpy as np
from scipy.optimize import minimize

N = 400

def compute_value(h):
    """Compute the objective value for a given height vector."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s ** 2))

def objective_and_gradient(x):
    """
    Objective: minimize max(c) where c = convolve(h, h), subject to sum(h)=1.
    We use a smooth approximation: minimize logsumexp(c) to handle the max.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros(N)
    
    # Normalize to sum=1 for evaluation
    h_norm = h / s
    c = np.convolve(h_norm, h_norm)
    
    # Smooth max using logsumexp with temperature
    beta = 10.0
    max_c = np.max(c)
    # logsumexp for stability
    shifted = c - max_c
    logsumexp_val = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # Gradient of logsumexp(c) wrt h
    # d/dh_i logsumexp(c) = sum_k exp(beta*(c_k - max_c)) / sum_j exp(beta*(c_j - max_c)) * d c_k / d h_i
    weights = np.exp(beta * (c - max_c))
    weights /= np.sum(weights)
    
    # d c_k / d h_i = 2 * h_{k-i} (with h_j=0 outside [0,N-1])
    # grad_i = 2 * sum_k weights_k * h_{k-i}
    # This is 2 * cross_correlation(weights, h) at lag i
    # Using FFT: grad = 2 * ifft(fft(weights, 2N) * conj(fft(h, 2N)))[:N]
    L = 2 * N
    W = np.fft.fft(weights, n=L)
    H = np.fft.fft(h_norm, n=L)
    grad_smooth = 2.0 * np.fft.ifft(W * np.conj(H), n=L).real[:N]
    
    # Account for normalization: h_norm = h / s
    # d h_norm_i / d h_j = delta_{ij}/s - h_i / s^2
    # So actual gradient wrt x: grad_x = (1/s) * (grad_smooth - mean(grad_smooth) * h_norm)
    # But we're optimizing over x directly, and h = x (not normalized in the variable)
    # Let's just use the gradient of the objective with respect to x
    # Objective f(x) = logsumexp(convolve(x/s, x/s))
    # df/dx_i = (1/s^2) * [2 * sum_k w_k * (x_{k-i} - x_i * sum_j x_j / s)]
    # Simplify: since s = sum(x), and we want gradient wrt x
    # Let's compute numerically or use chain rule carefully
    
    # Actually, let's just minimize with the constraint sum(h)=1 using Lagrange multiplier
    # Or better: parameterize h = (1 + epsilon) where sum(epsilon) = 0
    
    # For simplicity, return the gradient assuming we're in normalized space
    # The optimizer will handle the constraint via projection
    
    return logsumexp_val, grad_smooth

def construct():
    # Start with a good initial guess
    k = np.arange(N)
    
    # Try multiple initializations
    best_h = None
    best_val = np.inf
    
    # Initialization 1: constant + low-frequency anti-symmetric
    h1 = np.ones(N) + 0.3 * np.sin(2 * np.pi * k / N)
    h1 = h1 / np.sum(h1)
    
    # Initialization 2: constant + higher frequency
    h2 = np.ones(N) + 0.3 * np.sin(4 * np.pi * k / N)
    h2 = h2 / np.sum(h2)
    
    # Initialization 3: triangular bump
    h3 = np.ones(N)
    center = N // 2
    for i in range(N):
        dist = min(abs(i - center), N - abs(i - center))
        h3[i] = 1.0 - 0.5 * dist / (N / 2)
    h3 = h3 / np.sum(h3)
    
    # Initialization 4: random with zero mean perturbation
    rng = np.random.default_rng(42)
    h4 = np.ones(N) + 0.2 * rng.standard_normal(N)
    h4 -= np.mean(h4)
    h4 = 1.0 + 0.5 * h4 / np.std(h4)
    h4 = h4 / np.sum(h4)
    
    inits = [h1, h2, h3, h4]
    
    for h_init in inits:
        # Use projected gradient descent with constraint sum(h)=1
        h = h_init.copy()
        s = np.sum(h)
        h = h / s
        
        best_local_h = h.copy()
        best_local_val = compute_value(h)
        
        eta = 0.05
        mu = 0.9
        momentum = np.zeros(N)
        
        for i in range(300):
            # Compute gradient of smooth max
            c = np.convolve(h, h)
            max_c = np.max(c)
            beta = 5.0 + 5.0 * (i / 300.0)  # Increase sharpness over time
            shifted = c - max_c
            weights = np.exp(beta * shifted)
            weights /= np.sum(weights)
            
            # Gradient: 2 * cross_correlation(weights, h)
            L = 2 * N
            W = np.fft.fft(weights, n=L)
            H = np.fft.fft(h, n=L)
            grad = 2.0 * np.fft.ifft(W * np.conj(H), n=L).real[:N]
            
            # Project gradient to maintain sum=1: subtract mean
            grad -= np.mean(grad)
            
            # Gradient step with momentum
            momentum = mu * momentum - eta * grad
            h_new = h + momentum
            
            # Project back to sum=1
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-12:
                break
            h_new = h_new / s_new
            
            # Ensure no extreme values
            if np.any(np.abs(h_new) > 10.0):
                eta *= 0.5
                momentum = mu * momentum
                continue
            
            h = h_new
            
            val = compute_value(h)
            if val < best_local_val:
                best_local_val = val
                best_local_h = h.copy()
            
            # Adaptive learning rate
            if i % 50 == 0 and i > 0:
                # Simple adaptive: reduce if stuck
                if best_local_val < 1.05:
                    eta = max(0.01, eta * 0.8)
        
        if best_local_val < best_val:
            best_val = best_local_val
            best_h = best_local_h.copy()
        
        # If we're already good, stop early
        if best_val < 1.02:
            break
    
    # Final polish with smaller steps
    if best_h is not None:
        h = best_h.copy()
        eta = 0.01
        mu = 0.9
        momentum = np.zeros(N)
        best_val = compute_value(h)
        
        for i in range(200):
            c = np.convolve(h, h)
            max_c = np.max(c)
            beta = 10.0
            shifted = c - max_c
            weights = np.exp(beta * shifted)
            weights /= np.sum(weights)
            
            L = 2 * N
            W = np.fft.fft(weights, n=L)
            H = np.fft.fft(h, n=L)
            grad = 2.0 * np.fft.ifft(W * np.conj(H), n=L).real[:N]
            grad -= np.mean(grad)
            
            momentum = mu * momentum - eta * grad
            h_new = h + momentum
            
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-12:
                break
            h_new = h_new / s_new
            
            if np.any(np.abs(h_new) > 10.0):
                eta *= 0.5
                continue
            
            h = h_new
            val = compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        
        return best_h
    else:
        # Fallback
        h = np.ones(N)
        return h
