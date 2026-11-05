import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initialize
    # Random start
    rng = np.random.default_rng(42)
    h = rng.standard_normal(N)
    
    # Normalize sum to 1
    h = h / np.sum(h)
    
    best_val = np.inf
    best_h = h.copy()
    
    # Soft max parameter
    tau = 1.0
    tau_min = 0.01
    
    # Optimization loop
    lr = 0.01
    
    while time.time() < deadline:
        # Compute convolution
        c = np.convolve(h, h)
        
        # Compute objective
        val = 2.0 * N * np.max(c) / (np.sum(h) ** 2)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Smooth max
        # exp(c/tau) can overflow, so shift c
        c_shifted = c - np.max(c)
        soft_max_val = tau * np.log(np.sum(np.exp(c_shifted / tau)))
        
        # Gradient of soft max w.r.t c
        # d/dx (tau log sum exp(x/tau)) = (1/sum exp) * exp(x/tau)
        weights = np.exp(c_shifted / tau)
        weights /= np.sum(weights)
        
        # Gradient of soft max w.r.t h
        # c = h * h
        # dc/dh is a convolution with h (reversed? no, convolution is symmetric in derivative if we treat it as inner product?)
        # c_k = sum h_i h_{k-i}
        # d c_k / d h_j = h_{k-j} + h_{j-k} (if indices valid)
        # Actually, grad_h (h*h) = 2 * h * h_reversed?
        # Let's use the property that grad of sum w_k c_k is 2 * w * h (convolved with w?)
        # If J = sum w_k c_k, then dJ/dh_j = sum_k w_k (h_{k-j} + h_{j-k})
        # This is 2 * (w * h)[j] if w is symmetric?
        # w is not symmetric.
        # Let's just compute the gradient numerically or analytically.
        # Analytical:
        # grad = 2 * np.convolve(weights, h) + 2 * np.convolve(h, weights) ?
        # No.
        # c = h * h.
        # d c / d h is a linear operator.
        # grad_h (w^T c) = 2 * (w * h) ? No.
        # Let's use: grad = 2 * np.convolve(weights, h)
        # Check: if h=[1,1], w=[1,0,0]. c=[1,2,1].
        # w^T c = 1.
        # d/dh0 (1*1) = 1.
        # 2 * conv([1,0,0], [1,1]) = 2 * [1, 1] = [2, 2].
        # h0 deriv is 2? No, c0 = h0^2. d/dh0 = 2h0 = 2.
        # c1 = 2h0h1. d/dh0 = 2h1 = 2.
        # w=[1,0,0] only picks c0. So grad should be [2, 0].
        # My formula gave [2, 2].
        # Correct formula: grad_h = 2 * np.convolve(weights, h, mode='full')?
        # The convolution of weights and h gives the gradient of the correlation.
        
        # Simpler: Use finite differences for a few steps if analytical is tricky?
        # No, N=400, too slow.
        
        # Correct gradient:
        # J = sum_k w_k sum_i h_i h_{k-i}
        # dJ/dh_j = sum_k w_k (h_{k-j} + h_{j-k})
        # Let u_k = w_k.
        # Term 1: sum_k w_k h_{k-j}. Let m = k-j. k=m+j.
        # sum_m w_{m+j} h_m = (w_shifted * h)[j]?
        # This is equivalent to 2 * (w * h) if w is symmetric.
        # If w is not symmetric, it's w * h + h * w.
        
        grad_c = 2 * (np.convolve(weights, h) + np.convolve(h, weights))
        # Note: np.convolve(h, weights) is the same as np.convolve(weights, h) for real numbers?
        # (a*b)[n] = sum a_k b_{n-k}.
        # (b*a)[n] = sum b_k a_{n-k}. Let m=n-k. sum b_{n-m} a_m.
        # Yes, convolution is commutative.
        # So grad_c = 4 * np.convolve(weights, h)?
        # Wait.
        # dJ/dh_j = sum_k w_k (h_{k-j} + h_{j-k})
        # Let f_k = w_k.
        # sum_k f_k h_{k-j} is the convolution (f * h)[j] ?
        # (f * h)[j] = sum_m f_m h_{j-m}.
        # My sum is over k. Let m = k-j. k=m+j.
        # sum_m f_{m+j} h_m.
        # This is cross-correlation, not convolution.
        
        # Let's stick to the property:
        # grad = 2 * np.convolve(weights, h)
        # Let's test with h=[1,1], w=[1,0,0].
        # conv([1,0,0], [1,1]) = [1, 1, 0].
        # 2 * [1, 1, 0] = [2, 2, 0].
        # We need grad for h (length 2).
        # The convolution result has length 3.
        # The gradient of J w.r.t h0 is 2.
        # The gradient of J w.r.t h1 is 0?
        # J = c0 = h0^2.
        # dJ/dh0 = 2. dJ/dh1 = 0.
        # My conv gave [2, 2, 0].
        # So we take the central part?
        # Length of h is N. Length of w is 2N-1.
        # Conv length is 3N-1.
        # The gradient should be length N.
        
        # Actually, easier:
        # grad_h = 2 * np.convolve(weights, h)
        # Then extract the valid part?
        # No, the convolution of w (len 2N-1) and h (len N) is len 3N-1.
        # The derivative is embedded.
        
        # Let's use a simpler update:
        # Project gradient onto subspace sum(h)=1.
        
        # For now, let's just use a random restart strategy with normalization.
        # Given time constraints, a robust optimizer is better than a buggy gradient.
        
        # Let's try a simple coordinate descent or random perturbation.
        # Perturb h, normalize, check val.
        
        # Better: Use scipy.optimize.minimize?
        # Might be slow.
        
        # Let's implement the gradient correctly.
        # grad_h = 2 * np.convolve(weights, h, mode='full')
        # The gradient is the central N elements?
        # Let's assume the standard result:
        # grad = 2 * np.convolve(weights, h)
        # And we take the slice corresponding to h.
        
        # Since this is risky, I will use a random search with local refinement.
        
        h_new = h + lr * rng.standard_normal(N)
        h_new = h_new / np.sum(h_new)
        
        val_new = 2.0 * N * np.max(np.convolve(h_new, h_new)) / (np.sum(h_new) ** 2)
        
        if val_new < val:
            h = h_new
            lr *= 1.05
        else:
            lr *= 0.95
            
        # Reduce tau
        tau = max(tau_min, tau * 0.99)

    return best_h

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    h = rng.standard_normal(N)
    h = h / np.sum(h)
    
    best_val = np.inf
    best_h = h.copy()
    
    lr = 0.05
    tau = 1.0
    tau_min = 0.01
    
    while time.time() < deadline:
        c = np.convolve(h, h)
        val = 2.0 * N * np.max(c) / (np.sum(h) ** 2)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Random perturbation
        h_new = h + lr * rng.standard_normal(N)
        s = np.sum(h_new)
        if s == 0:
            continue
        h_new = h_new / s
        
        c_new = np.convolve(h_new, h_new)
        val_new = 2.0 * N * np.max(c_new) / (s ** 2)
        
        if val_new < val:
            h = h_new
            lr *= 1.01
        else:
            lr *= 0.95
            
        tau = max(tau_min, tau * 0.99)

    return best_h
# EVOLVE-BLOCK-END
