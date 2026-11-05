"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_and_grad(h, beta=50.0):
    """Soft-max approximation of max of autoconvolution entries and its gradient."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)
        soft_max = m + np.log(np.sum(exp_shifted)) / beta
        
        # Analytic gradient of soft_max w.r.t. h
        # d(soft_max)/d(h_i) = sum_j weights_j * d(conv_j)/d(h_i)
        # d(conv_j)/d(h_i) = h_{j-i} + h_{i-j} (with out-of-bounds = 0)
        # This is: 2 * (weights * h) convolved appropriately
        # More precisely: for each i, sum_j weights_j * (h[j-i] + h[i-j]) where valid
        # = 2 * sum_j weights_j * h[j-i] where 0<=j-i<N (by symmetry of the sum)
        # Actually it's: sum_j weights_j * h[j-i] (for 0<=j-i<N) + sum_j weights_j * h[i-j] (for 0<=i-j<N)
        # The second sum is the same as the first with j->2N-1-j, so:
        # grad[i] = sum_j weights[j] * h[j-i] (valid) + sum_j weights[j] * h[i-j] (valid)
        # = (weights * h) correlation-like operation
        
        # Efficient computation:
        # term1[i] = sum_{j: 0<=j-i<N} weights[j] * h[j-i] = sum_{k: 0<=k<N} weights[i+k] * h[k]
        # term2[i] = sum_{j: 0<=i-j<N} weights[j] * h[i-j] = sum_{k: 0<=k<N} weights[i-k] * h[k]
        
        # Use FFT for efficiency
        n_conv = len(conv)  # 2N-1
        n_fft = 1
        while n_fft < n_conv + N - 1:
            n_fft *= 2
        
        W = np.zeros(n_fft, dtype=complex)
        H = np.zeros(n_fft, dtype=complex)
        W[:n_conv] = weights
        H[:N] = h
        
        Wf = np.fft.rfft(W)
        Hf = np.fft.rfft(H)
        
        # term1: (weights * h) circular correlation -> linear
        # term1[i] = sum_k weights[i+k] * h[k] for k=0..N-1, i=0..N-1
        # This is cross-correlation of weights and h
        # In FFT: ifft(conj(fft(weights)) * fft(h)) gives correlation
        
        # Actually simpler: just use np.convolve with the weights
        # term1[i] = sum_{k=0}^{N-1} weights[i+k] * h[k]
        # This is like a cross-correlation
        
        # Let's just do it directly for clarity (N=400, n_conv=799, this is fast enough)
        grad_soft_max = np.zeros(N)
        for i in range(N):
            # term1: sum_k weights[i+k] * h[k] where 0<=i+k<n_conv and 0<=k<N
            k_max = min(N, n_conv - i)
            if k_max > 0:
                term1 = np.dot(weights[i:i+k_max], h[:k_max])
            else:
                term1 = 0.0
            
            # term2: sum_k weights[i-k] * h[k] where 0<=i-k<n_conv and 0<=k<N
            # i-k >= 0 => k <= i, i-k < n_conv => k > i - n_conv (always true since i<N<n_conv)
            k_max2 = min(i + 1, N)
            if k_max2 > 0:
                term2 = np.dot(weights[:k_max2][::-1], h[:k_max2])
            else:
                term2 = 0.0
            
            grad_soft_max[i] = term1 + term2
        
        return soft_max, grad_soft_max
    else:
        return m, np.zeros(N)

def _soft_max_obj(x, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv_max, _ = _autoconv_max_and_grad(h, beta)
    return 2.0 * N * conv_max / (s * s)

def _soft_max_obj_and_grad(x, beta=50.0):
    """Objective and gradient for L-BFGS-B."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv_max, grad_conv = _autoconv_max_and_grad(h, beta)
    
    # f = 2*N*conv_max/s^2
    # df/dh_i = 2*N * (grad_conv_i * s^2 - conv_max * 2*s) / s^4
    #          = 2*N * (grad_conv_i * s - 2*conv_max) / s^3
    
    obj = 2.0 * N * conv_max / (s * s)
    grad = 2.0 * N * (grad_conv * s - 2.0 * conv_max) / (s * s * s)
    
    return obj, grad

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(3):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns
    for period in [2, 3, 4]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Wave patterns
    for freq in [3, 7, 15]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Block patterns
    for block_size in [20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # Optimize each initialization
    betas = [30.0, 80.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 40.0:
            break
        
        for beta in betas:
            if time.time() > deadline - 40.0:
                break
            
            h_curr = h0.copy()
            
            # L-BFGS-B optimization with analytic gradient
            try:
                res = minimize(_soft_max_obj_and_grad, h_curr, 
                              method='L-BFGS-B',
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                h_new = res.x
                val_new = _objective_hard(h_new)
                
                if val_new < best_val:
                    best_val = val_new
                    best_h = h_new.copy()
            except Exception:
                pass
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 15.0:
        for beta in [100.0, 200.0]:
            if time.time() > deadline - 15.0:
                break
            try:
                res = minimize(_soft_max_obj_and_grad, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
