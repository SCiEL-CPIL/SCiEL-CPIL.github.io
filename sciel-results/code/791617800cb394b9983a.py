import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _evaluate(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def _smooth_objective(x, beta):
    """
    Smooth surrogate: use log-sum-exp to approximate max(conv).
    We work with log-sum-exp of the convolution entries.
    Since we want to minimize max(conv)/sum(h)^2, and by scale invariance
    we can fix sum(h)=1. Then we minimize max(conv).
    
    Actually, let's think about this differently. The objective is:
    abs(2*N*max(conv)/sum(h)^2)
    
    By scale invariance, set sum(h) = 1. Then minimize abs(2*N*max(conv)).
    Since we want the value to be small, and 2*N is positive, we want 
    max(conv) to be as small as possible (ideally negative, but abs makes it 
    |2*N*max(conv)|).
    
    Wait - if max(conv) is negative, abs makes it positive. So we want max(conv)
    close to 0 from either side. But actually, for the value to be < 2, we need
    |max(conv)| < sum(h)^2/N = 1/N when sum(h)=1.
    
    Let's use: minimize log-sum-exp(beta * conv) which approximates log(max(conv))
    when beta is large. But we need to handle the case where max(conv) could be negative.
    
    Actually, a better approach: minimize the maximum of conv directly using a 
    smooth approximation. The log-sum-exp of beta*conv approximates max(conv) for large beta.
    But if all conv entries are negative, log-sum-exp will be very negative, which is good.
    
    Let's just minimize log(sum(exp(beta*conv))) - this is a smooth upper bound on log(max(conv)).
    Minimizing this will push max(conv) down.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(x, x)
    # Shift to avoid overflow in exp
    conv_max = np.max(conv)
    logsumexp = conv_max + np.log(np.sum(np.exp(beta * (conv - conv_max))))
    return logsumexp

def _smooth_grad(x, beta):
    """Gradient of the smooth objective."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.zeros_like(x)
    conv = np.convolve(x, x)
    conv_max = np.max(conv)
    exp_vals = np.exp(beta * (conv - conv_max))
    denom = np.sum(exp_vals)
    # d(logsumexp)/d(conv_i) = exp(beta*(conv_i - conv_max)) / sum(exp(...))
    weights = exp_vals / denom
    # d(conv_j)/d(x_i) = x_{j-i} + x_{j+N-1-i} for full convolution
    # conv[j] = sum_i x[i] * x[j-i] where x[k]=0 for k<0 or k>=N
    # So d(conv[j])/d(x[i]) = x[j-i] if 0<=j-i<N else 0, plus x[j+N-1-i] if 0<=j+N-1-i<N else 0
    # Actually for full convolution of length 2N-1:
    # conv[j] = sum_{i=max(0,j-N+1)}^{min(N-1,j)} x[i]*x[j-i]
    # d(conv[j])/d(x[i]) = x[j-i] + (if j+i >= N) x[j-i+N-1]... 
    # Let me think more carefully.
    # np.convolve(x,x) with mode='full' gives result of length 2N-1.
    # result[k] = sum_i x[i] * x[k-i] where the sum is over valid i.
    # d(result[k])/d(x[i]) = x[k-i] + x[k-i+N-1]... no.
    # Actually: result[k] = sum_{i} x[i]*x[k-i]
    # d(result[k])/d(x[i]) = x[k-i] (treating x as fixed in the second factor)
    #                     + x[k-i] (treating x as fixed in the first factor)
    # Wait no. result[k] = sum_i x[i] * x[k-i]
    # If we differentiate with respect to x[i0]:
    # d/dx[i0] sum_i x[i]*x[k-i] = x[k-i0] + x[k-i0] = 2*x[k-i0] if i0 is in range
    # But only if k-i0 is a valid index for x.
    # Hmm, actually it's: sum_i [delta_{i,i0}*x[k-i] + x[i]*delta_{k-i,i0}]
    # = x[k-i0] + x[i0] (if k-i0 is valid)
    # Wait: delta_{k-i,i0} means k-i=i0, so i=k-i0.
    # So d(result[k])/d(x[i0]) = x[k-i0] (from first term, when i=i0) + x[k-i0] (from second term, when i=k-i0)
    # = 2*x[k-i0] if 0 <= k-i0 < N, else just one term or zero.
    # Actually: 
    # First term: i=i0 gives x[k-i0] if 0<=k-i0<N
    # Second term: i=k-i0 gives x[k-i0] if 0<=k-i0<N (and i=k-i0 must be in [0,N-1])
    # So if 0<=k-i0<N: d(result[k])/d(x[i0]) = 2*x[k-i0]
    # Otherwise: 0
    # But wait, that's not right either. Let me redo.
    # result[k] = sum_{i=0}^{N-1} x[i] * x[k-i] where x[j]=0 if j<0 or j>=N
    # d(result[k])/d(x[i0]) = sum_i [delta_{i,i0} * x[k-i] + x[i] * delta_{k-i,i0}]
    # = x[k-i0] + sum_i x[i] * delta_{k-i,i0}
    # = x[k-i0] + x[k-i0]  (since delta_{k-i,i0} is 1 when i=k-i0)
    # = 2*x[k-i0] if 0<=k-i0<N, else 0 (with the understanding that x[j]=0 for invalid j)
    
    # So grad[i0] = sum_k weights[k] * 2*x[k-i0] where 0<=k-i0<N
    # Let's vectorize this.
    grad = np.zeros(N)
    for i in range(N):
        # k-i must be in [0, N-1], so k in [i, i+N-1]
        k_min = i
        k_max = min(i + N - 1, 2*N - 2)
        if k_min <= k_max:
            grad[i] = 2.0 * np.dot(weights[k_min:k_max+1], x[0:k_max-k_min+1])
    return grad

def _better_smooth_grad(x, beta):
    """Vectorized gradient computation."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.zeros_like(x)
    conv = np.convolve(x, x)
    conv_max = np.max(conv)
    exp_vals = np.exp(beta * (conv - conv_max))
    denom = np.sum(exp_vals)
    weights = exp_vals / denom
    
    # grad[i] = 2 * sum_{k=i}^{i+N-1} weights[k] * x[k-i] for valid k
    # This is like a cross-correlation of weights with x
    # grad[i] = 2 * (weights * x_reversed)[i] ... let me think
    # Actually: sum_{k=i}^{i+N-1} weights[k] * x[k-i] = sum_{j=0}^{N-1} weights[i+j] * x[j]
    # This is the cross-correlation of weights and x, evaluated at lag i.
    # np.correlate(weights, x, mode='full')[i + N - 1] ... 
    # Let's just use FFT or direct computation.
    
    # Using the fact that this is a convolution-like operation:
    # Let's flip x and convolve with weights
    x_flip = x[::-1]
    # conv_weights_xflip = np.convolve(weights, x_flip)
    # Then grad[i] = 2 * conv_weights_xflip[i + N - 1] ... need to check indexing
    
    # Simpler: just do it with a loop but vectorize the inner product
    grad = np.zeros(N)
    # For each i, grad[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * x[j] where i+j < 2N-1
    for i in range(N):
        max_j = min(N, 2*N-1-i)
        if max_j > 0:
            grad[i] = 2.0 * np.dot(weights[i:i+max_j], x[:max_j])
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    def _update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Try different initializations and optimize
    initializations = []
    
    # 1. Constant (baseline)
    initializations.append(np.ones(N, dtype=np.float64))
    
    # 2. Spectral: sin waves
    for freq in [1, 2, 3, 5, 8, 13, 21]:
        t = np.arange(N)
        initializations.append(np.sin(2 * np.pi * freq * t / N))
        initializations.append(np.cos(2 * np.pi * freq * t / N))
    
    # 3. Alternating signs
    initializations.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 4. Block patterns
    for block_size in [10, 20, 50, 100]:
        h = np.ones(N)
        for i in range(0, N, block_size):
            if (i // block_size) % 2 == 1:
                h[i:i+block_size] = -1.0
        initializations.append(h)
    
    # 5. Chebyshev-like
    t = np.linspace(-1, 1, N)
    for degree in [2, 4, 8, 16]:
        initializations.append(np.cos(degree * np.arccos(t)))
    
    # 6. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        initializations.append(rng.standard_normal(N))
    
    # 7. Spike-like
    for spike_pos in [N//4, N//2, 3*N//4]:
        h = np.ones(N)
        h[spike_pos] = -5.0
        initializations.append(h)
    
    # Optimize each initialization
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        h0 = init.copy()
        # Normalize so sum is 1 for optimization
        s = np.sum(h0)
        if abs(s) < 1e-12:
            continue
        h0 = h0 / s
        
        # Try different beta values
        for beta in [10, 50, 200, 800]:
            if time.time() > deadline - 5:
                break
            
            try:
                result = minimize(
                    _smooth_objective,
                    h0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_better_smooth_grad,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < 1e10:
                    h_opt = result.x
                    val = _evaluate(h_opt)
                    _update_best(h_opt, val)
                    
                    # Also try negation
                    h_neg = -h_opt
                    val_neg = _evaluate(h_neg)
                    _update_best(h_neg, val_neg)
                    
            except Exception:
                pass
    
    # Final polish with higher beta on best candidate
    if time.time() < deadline - 5:
        h0 = best_h.copy()
        s = np.sum(h0)
        if abs(s) > 1e-12:
            h0 = h0 / s
            for beta in [2000, 5000, 10000]:
                if time.time() > deadline - 2:
                    break
                try:
                    result = minimize(
                        _smooth_objective,
                        h0,
                        args=(beta,),
                        method='L-BFGS-B',
                        jac=_better_smooth_grad,
                        options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if result.fun < 1e10:
                        h_opt = result.x
                        val = _evaluate(h_opt)
                        _update_best(h_opt, val)
                        h0 = h_opt
                except Exception:
                    pass
    
    return best_h
