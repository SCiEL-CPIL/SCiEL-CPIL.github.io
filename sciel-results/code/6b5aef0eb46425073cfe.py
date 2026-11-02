import numpy as np

N = 600

def _autoconvolve(h):
    return np.convolve(h, h, mode='full')

def construct():
    best_h = None
    best_score = float('inf')
    
    # Try a few restarts
    for seed in [42, 123, 456, 789, 101]:
        np.random.seed(seed)
        h = np.random.rand(N)
        h = h / np.linalg.norm(h)
        
        # Annealing schedule
        T_start = 1.0
        T_end = 0.01
        steps = 500
        
        for i in range(steps):
            # Anneal T
            T = T_start * (T_end / T_start) ** (i / steps)
            
            # Compute convolution
            c = _autoconvolve(h)
            
            # Soft max
            # c / T can be large, use shift for stability
            c_max = np.max(c)
            exp_c = np.exp((c - c_max) / T)
            sum_exp = np.sum(exp_c)
            # Objective value (not strictly needed for gradient, but good for tracking)
            # val = T * (c_max + np.log(sum_exp))
            
            # Gradient of T * log(sum(exp(c/T)))
            # d/dh [ T * log(sum(exp(c_k/T))) ]
            # = T * (1/sum) * sum (exp(c_k/T) / T * dc_k/dh)
            # = (1/sum) * sum (exp(c_k/T) * dc_k/dh)
            # Weights w_k = exp(c_k/T) / sum_exp
            w = exp_c / sum_exp
            
            # Gradient of c_k w.r.t h
            # c_k = sum_i h_i h_{k-i}
            # dc_k/dh_j = h_{k-j} + h_{j-k} (if in range)
            # This is the convolution of h with a shifted delta?
            # Actually, the gradient of the whole vector c is a Toeplitz matrix times h.
            # But we only need the gradient of the weighted sum: sum w_k c_k
            # grad = sum_k w_k (dc_k/dh)
            # This is equivalent to convolving w with h?
            # Let's check.
            # c_k = <h, shift_k(h)>
            # sum w_k c_k = <h, sum w_k shift_k(h)>
            # So grad = sum_k w_k shift_k(h)
            
            # We need to compute sum_k w_k * shift_k(h)
            # This is a convolution of w and h?
            # shift_k(h) is h shifted by k.
            # sum_k w_k h_{i-k} = (w * h)_i ?
            # Yes, if w is treated as a kernel.
            # But w is defined on the range of c (length 2N-1).
            # h is length N.
            # The convolution w * h will have length (2N-1) + N - 1 = 3N-2.
            # We only care about the part corresponding to h's indices (0 to N-1).
            
            # Let's compute the gradient directly using the correlation property.
            # grad_j = sum_k w_k (h_{k-j} + h_{j-k})
            # Note that indices must be valid for h (0..N-1).
            
            # Efficient way:
            # grad = np.convolve(w, h, mode='valid')? No.
            # Let's just compute it via the definition if N is small, but N=600 is moderate.
            # We can use FFT? Or just a loop? 600*1200 is ~1M ops, fast enough in Python if vectorized.
            
            # Vectorized gradient computation:
            # We want G_j = sum_k w_k h_{k-j} + sum_k w_k h_{j-k}
            # Let's construct the matrix or use convolutions.
            # term1_j = sum_k w_k h_{k-j}
            # This is (w * h)_j if we align them correctly.
            # w has length 2N-1. h has length N.
            # Let's pad h to length 2N-1? No.
            
            # Alternative:
            # grad = np.zeros(N)
            # for k in range(len(w)):
            #    if w[k] > 1e-6:
            #       # shift h by k
            #       # h_shifted[i] = h[i-k] if 0<=i-k<N else 0
            #       # This is slow in Python loop.
            
            # Let's use FFT for the gradient calculation.
            # grad = IFFT(FFT(w) * FFT(h)) ?
            # The operation is sum_k w_k shift_k(h).
            # This is a linear operator.
            
            # Simpler: Just use the subgradient of the hard max if T is small.
            # But for annealing, we need the smooth gradient.
            
            # Let's try a different formulation:
            # The gradient of the objective w.r.t h is:
            # grad = (w * h) + (w * h)[::-1] ? No.
            
            # Let's stick to a simpler Subgradient Method for the hard max, 
            # which is more robust and doesn't require complex gradient derivation.
            # And use random restarts.
            
            break # Break out of annealing loop to implement subgradient instead
            
        # Implement Subgradient Descent
        h = np.random.rand(N)
        h = h / np.linalg.norm(h)
        
        alpha = 0.1
        for i in range(2000):
            c = _autoconvolve(h)
            k_max = np.argmax(c)
            score = c[k_max]
            
            # Gradient of c_k_max w.r.t h
            # c_k = sum_i h_i h_{k-i}
            # grad_j = h_{k-j} + h_{j-k}
            
            # We compute this vectorized
            # h is length N.
            # k_max is in [0, 2N-2].
            
            # term1: h_{k-j}
            # valid j where 0 <= k-j < N => k-N < j <= k
            # term2: h_{j-k}
            # valid j where 0 <= j-k < N => k <= j < k+N
            
            grad = np.zeros(N)
            
            # Term 1: h_{k-j}
            # Let u = k-j. j = k-u.
            # u in [0, N-1]. j in [k-(N-1), k].
            j_start = max(0, k_max - (N - 1))
            j_end = min(N, k_max + 1)
            if j_start < j_end:
                # j indices
                js = np.arange(j_start, j_end)
                # u = k - j
                us = k_max - js
                grad[js] += h[us]
                
            # Term 2: h_{j-k}
            # Let v = j-k. j = k+v.
            # v in [0, N-1]. j in [k, k+N-1].
            j_start2 = max(0, k_max)
            j_end2 = min(N, k_max + N)
            if j_start2 < j_end2:
                js = np.arange(j_start2, j_end2)
                vs = js - k_max
                grad[js] += h[vs]
                
            # Update
            h = h - alpha * grad
            h = np.clip(h, 0, None)
            
            norm = np.linalg.norm(h)
            if norm < 1e-12:
                h = np.random.rand(N)
                h = h / np.linalg.norm(h)
            else:
                h = h / norm
                
            # Decay alpha
            alpha *= 0.999
            
        # Final score
        c_final = _autoconvolve(h)
        final_score = np.max(c_final)
        
        if final_score < best_score:
            best_score = final_score
            best_h = h
            
    return list(best_h)
