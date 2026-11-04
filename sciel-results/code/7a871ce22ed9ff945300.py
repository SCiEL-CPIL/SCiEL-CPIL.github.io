import numpy as np
import time

N = 400

def compute_value(h):
    S = np.sum(h)
    if S == 0:
        return float('inf')
    C = np.convolve(h, h)
    max_C = np.max(C)
    return abs(2 * N * max_C / (S ** 2))

def construct():
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try a few initializations
    inits = []
    inits.append(np.ones(N))
    inits.append(np.random.randn(N))
    inits.append(np.sin(np.linspace(0, 10*np.pi, N)))
    
    for h_init in inits:
        h = h_init.copy()
        # Normalize sum to 1
        S = np.sum(h)
        if S == 0:
            h += 0.1
            S = np.sum(h)
        h = h / S
        
        # Gradient Descent on max_C
        alpha = 0.01
        # Adaptive alpha?
        
        for step in range(2000):
            if time.time() - start_time > time_limit - safety_margin:
                break
            
            C = np.convolve(h, h)
            k_star = np.argmax(C)
            max_C = C[k_star]
            
            # Gradient of C[k_star] w.r.t h
            # d/dh_j C_k = h_{k-j} + h_j (with bounds)
            # We can compute this efficiently
            # Let's just use numerical gradient for simplicity and robustness, 
            # or vectorized analytic.
            
            # Analytic gradient:
            # g_j = h_{k-j} (if 0<=k-j<N) + h_{k-j} (if j != k-j and 0<=k-j<N? No)
            # Term 1: i=j -> h_{k-j}
            # Term 2: i=k-j -> h_j
            # So g_j = h_{k-j} + h_j if k-j is valid.
            # If k-j == j, g_j = 2 h_j.
            
            g = np.zeros(N)
            # We need to add h shifted by k and h shifted by -k?
            # Let's construct the gradient vector.
            # For each j, contribution is h[k-j] (if valid) + h[j] (if valid, which is always)
            # Wait, the second term h_j comes from i = k-j.
            # So g_j = h_{k-j} (if 0 <= k-j < N) + h_j
            # But wait, if k-j is out of bounds, that term is 0.
            
            # Let's implement this loop or vectorize.
            # Vectorization:
            # g = h.copy() # from the h_j term
            # Add h[k-j] term.
            # This is a reversed and shifted h.
            
            # Let's do a simple loop for clarity, N=400 is small.
            for j in range(N):
                k_minus_j = k_star - j
                if 0 <= k_minus_j < N:
                    g[j] += h[k_minus_j]
                # The h_j part is already in g if we initialized g=h?
                # No, let's be explicit.
                g[j] = h[j] # from i=k-j term? No.
                
            # Let's restart gradient logic.
            # C_k = sum_i h_i h_{k-i}
            # d C_k / d h_j = h_{k-j} (if k-j valid) + h_{k-j} (if j valid and k-j != j? No)
            # If i=j, term is h_j h_{k-j}. Deriv w.r.t h_j is h_{k-j}.
            # If i=k-j, term is h_{k-j} h_j. Deriv w.r.t h_j is h_{k-j}.
            # So if k-j is valid and k-j != j, derivative is 2 * h_{k-j}.
            # If k-j == j, derivative is 2 * h_j.
            # If k-j is invalid, derivative is 0 (from first term) + h_{k-j} (invalid) -> 0?
            # Wait. If k-j is invalid, the term h_{k-j} h_j doesn't exist.
            # So only the term where i=j exists?
            # If i=j, we need k-i = k-j to be a valid index for h_{k-j}.
            # If k-j is invalid, the product h_j h_{k-j} is 0.
            # So derivative is 0.
            
            # So:
            # if 0 <= k-j < N:
            #    if k-j == j: g[j] = 2 * h[j]
            #    else: g[j] = 2 * h[k-j]
            # else:
            #    g[j] = 0
            
            # Let's verify.
            # h = [1, 2], N=2. k=1.
            # C_1 = h_0 h_1 + h_1 h_0 = 2*2=4.
            # d C_1 / d h_0: i=0 -> h_1. i=1 -> h_0. Sum = 2+1=3?
            # Formula: j=0. k-j=1. Valid. k-j != j. g[0] = 2*h[1] = 4.
            # j=1. k-j=0. Valid. k-j != j. g[1] = 2*h[0] = 2.
            # Let's check manually.
            # C_1 = 2 h_0 h_1.
            # d/dh_0 = 2 h_1 = 4. Correct.
            # d/dh_1 = 2 h_0 = 2. Correct.
            
            g = np.zeros(N)
            for j in range(N):
                kmj = k_star - j
                if 0 <= kmj < N:
                    if kmj == j:
                        g[j] = 2 * h[j]
                    else:
                        g[j] = 2 * h[kmj]
            
            # Update
            h -= alpha * g
            
            # Normalize
            S = np.sum(h)
            if S == 0:
                h += 1e-6
                S = np.sum(h)
            h = h / S
            
            # Evaluate
            val = compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Adaptive alpha?
            # If val increased, reduce alpha?
            
    return best_h
