import numpy as np
import time
from scipy.optimize import minimize

N = 400

def compute_objective_and_grad(h, beta=10.0):
    # h is shape (N,)
    # conv is shape (2N-1,)
    c = np.convolve(h, h)
    
    # Soft max
    # To avoid overflow, subtract max
    c_max = np.max(c)
    exp_c = np.exp(beta * (c - c_max))
    sum_exp = np.sum(exp_c)
    
    # Objective value (approx)
    obj = c_max + np.log(sum_exp) / beta
    
    # Gradient of soft max
    # weights
    w = exp_c / sum_exp
    
    # Gradient of c w.r.t h
    # c_k = sum_i h_i h_{k-i}
    # grad_j = sum_k w_k * (h_{k-j} + h_{j-k})  (careful with bounds)
    
    # Efficient way:
    # grad = convolve(w, h) + convolve(w[::-1], h) ??
    # Let's derive.
    # c_k = sum_i h_i h_{k-i}
    # d c_k / d h_j = h_{k-j} + h_{j-k} (if indices valid)
    
    # So grad_j = sum_k w_k (h_{k-j} + h_{j-k})
    # Term 1: sum_k w_k h_{k-j}. Let m = k-j => k = m+j.
    # sum_m w_{m+j} h_m. This is correlation of w and h?
    # np.correlate(w, h, mode='full')?
    # Let's just use the correlation definition.
    # Actually, simpler:
    # grad = np.convolve(w, h) is not quite right because of the "full" length.
    
    # Let's stick to the definition with valid indices.
    # For j in 0..N-1:
    # g[j] = 0
    # for k in 0..2N-2:
    #   i = k - j
    #   if 0 <= i < N: g[j] += w[k] * h[i]
    #   i2 = j - k
    #   if 0 <= i2 < N: g[j] += w[k] * h[i2]
    
    # This is O(N^2) per step, which is 160,000 ops. Very fast.
    
    g = np.zeros(N)
    # Vectorized approach using convolution/correlation
    # Part 1: sum_k w_k h_{k-j}
    # This is the cross-correlation of w and h.
    # np.correlate(w, h, 'full') gives sum_x w[x] h[x+y] ??
    # Let's just use the explicit loop or a small matrix if N is small, but N=400 is fine for loops if not in python.
    # Better: use FFT? Or just np.convolve.
    
    # grad_part1[j] = sum_k w_k h_{k-j}
    # This is (w * h)[j] if we consider appropriate shifts?
    # Actually, np.correlate(w, h, mode='same')?
    
    # Let's use the property:
    # grad = np.convolve(w, h, mode='full') + np.convolve(w[::-1], h, mode='full')
    # Let's verify indices.
    
    # We will compute the gradient using the explicit formula with numpy vectorization on indices.
    # k ranges 0 to 2N-2
    # j ranges 0 to N-1
    # We need pairs (k, j) such that i = k-j is in [0, N-1] OR i2 = j-k is in [0, N-1].
    
    # Efficient trick:
    # grad1 = np.correlate(w, h, mode='full')
    # grad2 = np.correlate(w, h, mode='full') # symmetric?
    
    # Let's just do the O(N^2) in numpy by broadcasting? No, memory.
    # Just use a simple loop over N? 400 iterations of vectorized ops.
    
    grad = np.zeros(N)
    # w is length 2N-1
    # h is length N
    
    # For each j, we need sum of w[k] * h[k-j] for valid k-j
    # This is the cross-correlation.
    # np.correlate(w, h, 'full')[N-1+j] ?
    
    # Let's use scipy.signal.correlate? No, standard np is fine.
    # np.correlate(a, v, mode) is sum a[n] v[n-k].
    # We want sum w[k] h[k-j]. Let n=k, m=j. sum w[n] h[n-m].
    # This is correlate(w, h)[m] in 'full'?
    # np.correlate(w, h, 'full')[m] = sum_n w[n] h[n-m]. Yes.
    
    # So part1[j] = np.correlate(w, h, 'full')[j]
    # Part2[j] = sum_k w[k] h[j-k]. Let n=k. sum w[n] h[j-n].
    # This is correlate(h, w, 'full')[j] ??
    # correlate(h, w)[j] = sum_n h[n] w[n-j]. No.
    # sum_k w[k] h[j-k]. Let m=k. sum_m w[m] h[j-m].
    # This is (w * h)[j] if h was infinite?
    # It is the convolution at j.
    # np.convolve(w, h, 'full')[j]?
    # convolve(w, h)[j] = sum_i w[i] h[j-i]. Yes.
    
    # So grad = np.correlate(w, h, 'full')[0:N] + np.convolve(w, h, 'full')[0:N]
    
    part1 = np.correlate(w, h, mode='full')[:N]
    part2 = np.convolve(w, h, mode='full')[:N]
    grad = part1 + part2
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Initialize
    np.random.seed(42)
    h = np.random.randn(N)
    h /= np.sum(h) # Normalize sum to 1
    
    best_h = h.copy()
    best_val = np.inf
    
    # Objective function for tracking
    def get_val(h):
        S = np.sum(h)
        if S == 0: return np.inf
        c = np.convolve(h, h)
        return abs(2 * N * np.max(c) / S**2)
    
    best_val = get_val(h)
    
    # Gradient Descent with Soft-Max
    beta = 1.0
    eta = 0.01
    
    for i in range(100000):
        if time.time() > deadline:
            break
        
        # Compute soft max objective and grad
        # We use the soft max of the convolution to approximate max
        c = np.convolve(h, h)
        c_max = np.max(c)
        # Scale c to prevent overflow in exp
        exp_c = np.exp(beta * (c - c_max))
        sum_exp = np.sum(exp_c)
        w = exp_c / sum_exp
        
        # Gradient
        part1 = np.correlate(w, h, mode='full')[:N]
        part2 = np.convolve(w, h, mode='full')[:N]
        grad = part1 + part2
        
        # Update
        h = h - eta * grad
        
        # Project to sum=1
        s = np.sum(h)
        if s == 0:
             h = np.random.randn(N)
             h /= np.sum(h)
        else:
             h = h - (s - 1.0) / N
        
        # Evaluate
        val = get_val(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Adaptive step size?
        if i % 1000 == 0:
             # Decay eta or adjust beta
             eta = max(0.001, eta * 0.95)
    
    return best_h
