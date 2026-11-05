"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:n + n - 1]
    return conv

def _compute_value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj_and_grad(h, beta=50.0):
    """
    Compute soft-max smoothed objective and its gradient.
    Objective: 2*N*soft_max(conv) / (sum(h))^2
    where soft_max approximates max(conv).
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    m = np.max(conv)
    # Numerically stable soft max
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max = m + np.log(sum_exp) / beta
    weights = exp_shifted / sum_exp
    
    # Gradient of soft_max w.r.t. h
    # d(soft_max)/d(h_i) = sum_j weights_j * d(conv_j)/d(h_i)
    # d(conv_j)/d(h_i) = h_{j-i} + h_{i-j} (with 0 outside [0,N-1])
    # This equals: for each i, sum_j weights_j * (h[j-i] if 0<=j-i<N else 0 + h[i-j] if 0<=i-j<N else 0)
    # = 2 * sum_j weights_j * h[|j-i|] with appropriate bounds, but let's do it properly
    # 
    # Actually: d(conv_j)/d(h_i) = h_{j-i} + h_{i-j} where h_k = 0 for k outside [0,N-1]
    # Let's compute this via correlation
    # term1[i] = sum_j weights_j * h_{j-i} = (weights * h)[i] in correlation sense
    # This is the cross-correlation of weights and h
    # weights has length 2N-1, h has length N
    # For i in 0..N-1: term1[i] = sum_{j=max(0,i)}^{min(2N-2, i+N-1)} weights[j] * h[j-i]
    #                    = sum_{k=0}^{N-1} weights[i+k] * h[k]  (where k=j-i, valid when 0<=i+k<2N-1)
    # term2[i] = sum_j weights_j * h_{i-j} = sum_{j=max(0,i-N+1)}^{min(2N-2,i)} weights[j] * h[i-j]
    #                    = sum_{k=0}^{N-1} weights[i-k] * h[k]  (where k=i-j, valid when 0<=i-k<2N-1)
    
    # Use FFT for efficient computation
    # term1: correlate weights with h
    # term2: correlate reversed weights with h, or equivalently correlate weights with reversed h
    
    # Let's just do it directly since N=400 is small
    grad_sm = np.zeros(N)
    for i in range(N):
        # term1: sum over k where 0<=k<N and 0<=i+k<2N-1
        k_min = max(0, 0)
        k_max = min(N-1, 2*N-2-i)
        if k_min <= k_max:
            term1 = np.dot(weights[i+k_min:i+k_max+1], h[k_min:k_max+1])
        else:
            term1 = 0.0
        # term2: sum over k where 0<=k<N and 0<=i-k<2N-1, i.e., i-N+1<=k<=i
        k_min2 = max(0, i-(2*N-2))
        k_max2 = min(N-1, i)
        if k_min2 <= k_max2:
            term2 = np.dot(weights[k_min2:k_max2+1], h[k_min2:k_max2+1])
        else:
            term2 = 0.0
        grad_sm[i] = term1 + term2
    
    # Full objective: f = 2*N*soft_max / s^2
    # df/dh_i = 2*N * (grad_sm_i * s^2 - soft_max * 2*s) / s^4
    #          = 2*N * (grad_sm_i * s - 2*soft_max) / s^3
    obj = 2.0 * N * soft_max / (s * s)
    grad = 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)
    
    return obj, grad

def _objective_wrapper(h, beta):
    obj, _ = _soft_max_obj_and_grad(h, beta)
    return obj

def _grad_wrapper(h, beta):
    _, grad = _soft_max_obj_and_grad(h, beta)
    return grad

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Comb patterns (alternating signs) - these have naturally low autocorrelation peaks
    for period in [2, 3, 4, 5, 7]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.05 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 2. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.05 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Random normal
    for _ in range(5):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 4. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 5. Cosine window
    x = np.linspace(0, 1, N)
    initializations.append(0.5 + 0.5 * np.cos(2 * np.pi * x))
    
    # 6. Gaussian
    initializations.append(np.exp(-0.5 * ((x - 0.5) * 4) ** 2))
    
    # 7. Two-lobe
    initializations.append(1.0 + 0.5 * np.sin(np.pi * x))
    
    # Optimize each initialization with L-BFGS-B using soft-max objective
    betas = [30.0, 50.0, 100.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Ensure sum is not zero
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 1.0
        
        h_curr = h0.copy()
        
        # Try different beta values
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
            
            try:
                res = minimize(_objective_wrapper, h_curr, 
                              method='L-BFGS-B',
                              jac=lambda x: _grad_wrapper(x, beta),
                              args=(beta,),
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
            
            h_curr = h_new.copy()
        
        # Also try hard evaluation of the initial
        val0 = _objective_hard(h0)
        if val0 < best_val:
            best_val = val0
            best_h = h0.copy()
    
    # Final refinement: one more L-BFGS-B pass on the best
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective_wrapper, best_h, 
                          method='L-BFGS-B',
                          jac=lambda x: _grad_wrapper(x, 100.0),
                          args=(100.0,),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000})
            h_final = res.x
            val_final = _objective_hard(h_final)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    s_final = np.sum(best_h)
    if abs(s_final) < 1e-10:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
