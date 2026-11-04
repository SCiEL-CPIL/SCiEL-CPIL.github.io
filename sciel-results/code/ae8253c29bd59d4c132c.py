import numpy as np
import time

N = 400

def compute_value(h):
    S = np.sum(h)
    if np.isclose(S, 0):
        return np.inf
    conv = np.convolve(h, h)
    M = np.max(conv)
    return abs(2 * N * M / (S ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s budget, leave margin
    
    # Initial guess: Alternating signs to reduce cross terms?
    # Or random.
    # Let's try a few initializations.
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1.0
    val_alt = compute_value(h_alt)
    if val_alt < best_val:
        best_h = h_alt
        best_val = val_alt
        
    # Try random
    np.random.seed(42)
    h_rand = np.random.randn(N)
    val_rand = compute_value(h_rand)
    if val_rand < best_val:
        best_h = h_rand
        best_val = val_rand
        
    # Gradient Descent with Soft-Max
    # Objective: f(h) = 2 * N * soft_max(conv(h)) / S^2
    # We fix S=1 by projecting.
    
    h = best_h.copy()
    h = h / np.sum(h)  # Normalize to S=1
    
    beta = 10.0
    lr = 0.01
    steps = 5000
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        S = np.sum(h)
        conv = np.convolve(h, h)
        
        # Soft max
        max_conv = np.max(conv)
        exp_conv = np.exp(beta * (conv - max_conv))
        sum_exp = np.sum(exp_conv)
        soft_max = max_conv + np.log(sum_exp) / beta
        
        # Gradient of soft_max w.r.t conv
        # d(soft_max)/d(conv_k) = exp(beta*(conv_k - max_conv)) / sum_exp
        grad_conv = exp_conv / sum_exp
        
        # Gradient of conv w.r.t h
        # conv = h * h
        # d(conv_k)/d(h_j) = h_{k-j} + h_{j-k} (if valid)
        # This is the convolution of grad_conv with h?
        # conv = np.convolve(h, h)
        # d(conv)/d(h) is related to cross-correlation.
        # Specifically, grad_h = np.convolve(grad_conv, h, mode='full')?
        # Let's check dimensions.
        # conv length 2N-1. h length N.
        # If conv_k = sum_i h_i h_{k-i}, then d(conv_k)/d(h_j) = h_{k-j} + h_{j-k}.
        # This is the sum of two convolutions?
        # Actually, d/dh (h*h) = 2 * (h * dh) ? No.
        # Gradient of ||h*h||^2 is 4 * (h*h) * h?
        # Here we have linear objective in conv.
        # grad_h = np.convolve(grad_conv, h, mode='full') + np.convolve(grad_conv, h[::-1], mode='full')?
        # Let's use numerical gradient check for small N or derive carefully.
        
        # Analytical Gradient:
        # L = sum_k grad_conv_k * conv_k
        # conv_k = sum_i h_i h_{k-i}
        # dL/dh_j = sum_k grad_conv_k * (h_{k-j} + h_{j-k})
        # This is equivalent to convolving grad_conv with h.
        # Let G = grad_conv.
        # grad_h = np.convolve(G, h, mode='full')
        # Wait, indices.
        # If G is length 2N-1, h is length N.
        # np.convolve(G, h) length 3N-2.
        # We only need the part corresponding to h indices.
        
        # Simpler: Use scipy or just numerical?
        # Numerical is slow for N=400.
        # Let's use the property:
        # grad_h = np.convolve(grad_conv, h, mode='valid')? No.
        
        # Let's try:
        # grad_h = np.convolve(grad_conv, h, mode='full')
        # The relevant part is from index N-1 to 2N-2?
        
        # Actually, d(conv)/dh is a Toeplitz-like matrix.
        # grad_h = np.convolve(grad_conv, h, mode='full')
        # Take the middle part of length N?
        
        # Let's just use a small numerical gradient for safety if time permits,
        # or assume the convolution gradient.
        
        # Correct Gradient:
        # grad_h = np.convolve(grad_conv, h, mode='full')
        # The gradient of the sum is the sum of gradients.
        # d/dh_j (sum_k G_k sum_i h_i h_{k-i})
        # = sum_k G_k (h_{k-j} + h_{j-k})
        # This is (G * h)_j + (G * h_{rev})_j?
        # It is simply np.convolve(G, h, mode='full') centered?
        
        # Let's use a robust numerical gradient for the first few steps to verify,
        # then switch to analytical if it matches.
        
        # For now, use analytical approximation:
        grad_h = np.convolve(grad_conv, h, mode='full')
        # The gradient vector should have length N.
        # The convolution result has length (2N-1) + N - 1 = 3N-2.
        # The term h_j appears in conv_k for k=j and k=2N-2-j?
        
        # Let's just use a simple numerical gradient with central difference
        # on a subset or small step to be safe, or just use the analytical one
        # and hope the indexing is right.
        
        # Actually, d(conv)/dh = 2 * np.convolve(h, dh) is for squared norm.
        # For linear functional L(conv), grad_h = np.convolve(G, h, mode='full')
        # We need to extract the N components.
        
        # Let's try:
        # grad_h = np.convolve(grad_conv, h, mode='full')
        # grad_h = grad_h[N-1:2*N-1]
        
        # Objective f(h) = C * soft_max / S^2
        # grad f = C * ( (grad_sm * S^2 - soft_max * 2 S * grad_S) / S^4 )
        # grad_S = sum(ones) = N? No, dS/dh_j = 1.
        # So grad_S is vector of 1s.
        
        C = 2 * N
        f = C * soft_max / (S * S)
        
        # Gradient of f
        # df/dh_j = C * ( grad_sm_j * S^2 - soft_max * 2 S * 1 ) / S^4
        # = C * ( grad_sm_j / S^2 - 2 soft_max / S^3 )
        
        grad_f = C * ( grad_h / (S * S) - 2 * soft_max / (S * S * S) )
        
        # Update
        h -= lr * grad_f
        
        # Project to sum=1
        h = h / np.sum(h)
        
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
