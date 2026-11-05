"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective_hard(h):
    """Compute the hard objective value (the value the evaluator uses)."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta):
    """
    Soft-max surrogate for 2*N*max(conv)/(sum(h))^2.
    Uses log-sum-exp for numerical stability.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _soft_max_grad(h, beta):
    """
    Analytical gradient of the soft-max surrogate objective:
        f(h) = 2*N * softmax_beta(conv(h,h)) / (sum(h))^2
    
    where softmax_beta(conv) = m + log(sum(exp(beta*(conv-m))))/beta,
    and conv = np.convolve(h, h, mode='full').
    
    Derivation:
        Let S = sum(h), C = conv, M = softmax_beta(C).
        f = 2*N * M / S^2
        df/dh_i = 2*N * ( (dM/dh_i) * S^2 - M * 2*S ) / S^4
                = 2*N * ( dM/dh_i * S - 2*M ) / S^3
        
        dM/dh_i = sum_j w_j * dC_j/dh_i
        where w_j = exp(beta*(C_j - m)) / sum_k exp(beta*(C_k - m))
        
        C_j = sum_k h_k * h_{k + j - (N-1)}  (with valid indices only)
        dC_j/dh_i = h_{i + j - (N-1)} + h_{i - j + (N-1)}  (with valid indices)
        
        We can compute dM/dh efficiently using the weights w and h.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    M = m + np.log(sum_exp) / beta
    
    # Compute dM/dh:
    # dM/dh_i = sum_j weights_j * (h_{i + j - N + 1} + h_{i - j + N - 1})
    # Let k = j - (N-1), so k ranges from -(N-1) to (N-1)
    # dM/dh_i = sum_k weights_{k+N-1} * (h_{i+k} + h_{i-k})
    # This is essentially a weighted correlation/convolution.
    
    # We can compute this as:
    # term1[i] = sum_k weights_{k+N-1} * h_{i+k}  (valid when 0<=i+k<N)
    # term2[i] = sum_k weights_{k+N-1} * h_{i-k}  (valid when 0<=i-k<N)
    
    # Note: weights is symmetric because conv is symmetric and softmax preserves symmetry.
    # So weights_{k+N-1} = weights_{-k+N-1} = weights_{2N-2-(k+N-1)} = weights_{N-1-k}
    # Actually, let's just compute directly using numpy.
    
    # Create an array for the "kernel" of weights indexed by lag k
    # k ranges from -(N-1) to (N-1), index in weights is k + N - 1
    # We need to compute for each i:
    #   sum over k where 0<=i+k<N of weights[k+N-1] * h[i+k]
    #   + sum over k where 0<=i-k<N of weights[k+N-1] * h[i-k]
    
    # Efficient approach using convolution:
    # Let w_full be the weights array of length 2N-1.
    # term1 = correlate(h, w_full) but with appropriate alignment.
    # Actually, let's use a direct but vectorized approach.
    
    # For each i, the valid k for term1: k in [max(-(N-1), -i), min(N-1, N-1-i)]
    # This is complex. Let's use a different formulation.
    
    # Alternative: dM/dh = 2 * (w_conv * h) where w_conv is the "centered" weights
    # Actually, let's think about it differently.
    
    # C_j = sum_i h_i * h_{i+k} where k = j-N+1
    # dC_j/dh_i = h_{i+k} + h_{i-k}
    # dM/dh_i = sum_j w_j * (h_{i+k_j} + h_{i-k_j}) where k_j = j-N+1
    
    # Let's define an array W of length 2N-1 where W[j] = w_j
    # For a fixed i, dM/dh_i = sum_j W[j] * h_{i+j-N+1} + sum_j W[j] * h_{i-j+N-1}
    
    # The first sum: for each j, we need h at index i+j-N+1 to be valid (0 to N-1)
    # The second sum: for each j, we need h at index i-j+N-1 to be valid
    
    # We can compute this using np.convolve or np.correlate with padding.
    
    # Let's create a padded version of h and W.
    # Actually, the cleanest way is to note that:
    # sum_j W[j] * h_{i+j-N+1} is the (i-N+1)-th entry of convolve(h, W, 'full')? 
    # Let's check: convolve(h, W, 'full')[k] = sum_i h[i] * W[k-i]
    # We want sum_j W[j] * h[i+j-N+1]. Let m = i+j-N+1, so j = m-i+N-1
    # sum_m h[m] * W[m-i+N-1] where 0<=m<N and 0<=m-i+N-1<2N-1
    # This is convolve(h, W, 'full')[i-N+1 + (N-1)]? 
    # This is getting confusing. Let's just use a loop over the lags, which is O(N^2) but N=400, so 160k operations, which is fast.
    
    dM_dh = np.zeros(N)
    for j in range(2*N - 1):
        if weights[j] < 1e-15:
            continue
        k = j - (N - 1)
        # term1: h[i+k] for valid i
        # i + k in [0, N-1] => i in [-k, N-1-k]
        i_start = max(0, -k)
        i_end = min(N, N - k)
        if i_start < i_end:
            dM_dh[i_start:i_end] += weights[j] * h[i_start+k:i_end+k]
        
        # term2: h[i-k] for valid i
        # i - k in [0, N-1] => i in [k, k+N-1]
        i_start2 = max(0, k)
        i_end2 = min(N, k + N)
        if i_start2 < i_end2:
            dM_dh[i_start2:i_end2] += weights[j] * h[i_start2-k:i_end2-k]
    
    # Now compute the full gradient
    # df/dh_i = 2*N * (dM_dh_i * S - 2*M) / S^3
    grad = 2.0 * N * (dM_dh * s - 2.0 * M) / (s * s * s)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal with positive mean
    for _ in range(12):
        h0 = rng.standard_normal(N)
        h0 += 0.5
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 2. Sign sequences with DC offset
    for _ in range(12):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.3 * rng.standard_normal()
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 3. Sine waves with various frequencies and phases
    for freq in [1, 2, 3, 5, 7, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.8
        initializations.append(h0)
        
    # 4. Polynomial perturbations
    for _ in range(8):
        h0 = np.ones(N)
        t = np.linspace(0, 1, N)
        h0 += 0.2 * (rng.random() * t**2 - 0.1)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 5. Gaussian bumps
    for _ in range(8):
        h0 = np.zeros(N)
        center = rng.integers(0, N)
        width = rng.uniform(10, 50)
        t = np.arange(N)
        h0 += np.exp(-0.5 * ((t - center) / width)**2)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)

    # Optimization using scipy's minimize with analytical gradient
    betas = [30.0, 50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, h_curr, 
                              jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Final polish with high beta
    if time.time() < deadline - 10.0:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 5.0:
                break
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, best_h,
                              jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
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
