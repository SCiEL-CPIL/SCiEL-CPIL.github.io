import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 118.0  # Safety margin before 120s/150s kill

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    return np.max(np.abs(ac[1:N]))

def _objective_soft_max(x, beta=100.0):
    """
    Objective: Soft-max of squared sidelobes.
    This is a smooth surrogate for the max sidelobe problem.
    As beta -> infinity, it approaches the max.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes = ac[1:N]
    sq_sidelobes = sidelobes**2
    
    # Log-sum-exp for numerical stability
    max_sq = np.max(sq_sidelobes)
    if max_sq == 0:
        return 0.0, np.zeros(N)
    
    # obj = max_sq + log(sum(exp((sq - max_sq)/beta))) * beta ?
    # Actually, soft-max is (1/beta) * log(sum(exp(beta * x_i)))
    # We want to minimize the max.
    # Let's use a simpler robust approach:
    # Sum of exp(beta * (sq - max_sq)) is not differentiable at max.
    # Standard soft-max: (1/beta) * log( sum( exp(beta * sq) ) )
    
    # To avoid overflow:
    # log( sum( exp(beta * sq) ) ) = beta * max_sq + log( sum( exp(beta * (sq - max_sq)) ) )
    # So obj = max_sq + (1/beta) * log( sum( exp(beta * (sq - max_sq)) ) )
    
    shifted = beta * (sq_sidelobes - max_sq)
    # Clip shifted to avoid underflow/overflow issues in exp, though exp(-large) is 0 which is fine.
    # exp(shifted) where shifted <= 0.
    sum_exp = np.sum(np.exp(shifted))
    
    if sum_exp == 0:
        obj = max_sq
    else:
        obj = max_sq + np.log(sum_exp) / beta
        
    # Gradient calculation
    # d/dx_j obj = d/dx_j [ max_sq + (1/beta) log(sum_k exp(beta (c_k^2 - max_sq))) ]
    # This is complex because max_sq is also a function of x.
    # However, for large beta, the gradient is dominated by the terms near the max.
    # A simpler surrogate that is easier to differentiate:
    # Sum of squares is easy.
    # Let's stick to Sum of Squares but with a penalty on the max?
    # Or just use the gradient of the soft-max properly?
    
    # Let's try a different smooth surrogate: 
    # f(x) = sum(c_k^2) + lambda * (max(c_k^2))^p ? No, max is not smooth.
    
    # Let's try: f(x) = log(sum(exp(beta * c_k^2))) / beta
    # Gradient:
    # Let S = sum(exp(beta * c_k^2)).
    # f = (1/beta) log S.
    # df/dx_j = (1/beta) * (1/S) * dS/dx_j
    # dS/dx_j = sum_k exp(beta * c_k^2) * d(c_k^2)/dx_j
    # d(c_k^2)/dx_j = 2 * c_k * dc_k/dx_j
    # dc_k/dx_j is the derivative of the autocorrelation.
    # c_k = sum_i h_i h_{i+k}.
    # dc_k/dx_j = h_{j+k} (if j+k < N) + h_{j-k} (if j-k >= 0).
    
    # This gradient is O(N^2) to compute if we do it naively, or O(N log N) with FFT tricks?
    # Actually, we can compute the gradient of the sum of squares easily.
    # Let's revert to Sum of Squares for the gradient, but use Soft-Max for the value?
    # No, that's inconsistent.
    
    # Let's use the gradient of the Soft-Max.
    # We need dc_k/dx_j for all k.
    # This is equivalent to the convolution of the "influence" of x_j.
    # Actually, let's just use the Sum of Squares objective but with a higher weight on the peak?
    # No, let's try a simpler improvement:
    # The current objective is Sum of Squares.
    # Let's try: Objective = Sum(c_k^2) + alpha * Max(c_k^2) is not smooth.
    
    # Alternative: Use the gradient of the Sum of Squares, but evaluate the Soft-Max for selection?
    # That's not standard.
    
    # Let's try a different approach:
    # Keep the Sum of Squares objective, but use a better optimizer or more iterations?
    # The bottleneck might be that L-BFGS-B is too slow or not effective enough.
    
    # Let's try to implement the gradient for the Soft-Max properly.
    # It might be worth it.
    
    # c_k = sum_{i=0}^{N-1-k} h_i h_{i+k}
    # c_k^2
    # sum_k c_k^2
    # Gradient of sum_k c_k^2:
    # g_j = 2 * sum_k c_k * (h_{j+k} + h_{j-k})
    # This is what the current code does.
    
    # Gradient of Soft-Max:
    # g_j = (1/beta) * (1/S) * sum_k exp(beta * c_k^2) * 2 * c_k * (h_{j+k} + h_{j-k})
    # Let w_k = exp(beta * c_k^2) * 2 * c_k / S
    # Then g_j = (1/beta) * sum_k w_k * (h_{j+k} + h_{j-k})
    
    # We can compute w_k for all k.
    # Then for each j, we need sum_k w_k * (h_{j+k} + h_{j-k}).
    # This is a convolution-like operation.
    # Let A_j = sum_k w_k * h_{j+k}. This is a correlation of w and h.
    # Let B_j = sum_k w_k * h_{j-k}. This is a correlation of w and h.
    # We can compute these using FFT!
    
    # Let's implement this.
    
    # 1. Compute c_k
    # 2. Compute weights w_k
    # 3. Compute gradients via FFT
    
    # c_k is ac[1:N]
    # w_k = exp(beta * c_k^2) * 2 * c_k / S
    # Note: c_k can be negative. c_k^2 is positive.
    # The term 2*c_k handles the sign.
    
    # Let's define the vector w of length N-1.
    # w[k-1] = exp(beta * c_k^2) * 2 * c_k / S
    
    # We need to compute:
    # G_j = sum_{k=1}^{N-1} w_{k-1} * (h_{j+k} + h_{j-k})
    # G_j = sum_{k=1}^{N-1} w_{k-1} h_{j+k} + sum_{k=1}^{N-1} w_{k-1} h_{j-k}
    
    # Let W be a vector of length 2*N.
    # W[k] = w_{k-1} for k=1..N-1
    # W[-k] = w_{k-1} for k=1..N-1 (i.e. W[N-k] in zero-indexed if we shift?)
    # Actually, let's use circular convolution.
    # Pad w to length 2*N.
    # w_padded = np.zeros(2*N)
    # w_padded[1:N] = w  # w[0] corresponds to k=1
    # w_padded[N+1:2*N] = w[::-1] # w[N-1] corresponds to k=N-1?
    # Wait, h_{j-k} term.
    # If we convolve h with w_padded, we get sum_i h_i w_padded[j-i].
    # We want sum_k w_k h_{j+k}.
    # Let i = j+k => k = i-j.
    # sum_i h_i w_{i-j}.
    # This is cross-correlation.
    
    # Let's just compute the gradient numerically? No, N=400, O(N^2) is 160k ops.
    # In Python, a loop of 400 iterations with vectorized inner ops is fast.
    # But we are in a gradient function called by L-BFGS-B.
    # L-BFGS-B calls this many times.
    # O(N^2) per gradient call might be slow if there are many iterations.
    # N=400. 160,000 ops.
    # If L-BFGS-B does 100 iterations, that's 16 million ops.
    # NumPy vectorization makes this very fast (< 0.1s).
    
    # Let's stick to the O(N^2) gradient computation for the Soft-Max.
    
    # 1. Compute c_k (already done in ac)
    # 2. Compute S = sum(exp(beta * c_k^2))
    # 3. Compute w_k = exp(beta * c_k^2) * 2 * c_k / S
    # 4. Compute grad_j = (1/beta) * sum_k w_k * (h_{j+k} + h_{j-k})
    
    # Vectorized gradient computation:
    # grad = np.zeros(N)
    # for k in range(1, N):
    #     wk = w[k-1]
    #     if wk == 0: continue
    #     # h_{j+k} for j=0..N-1-k
    #     grad[0:N-k] += wk * h[k:N]
    #     # h_{j-k} for j=k..N-1
    #     grad[k:N] += wk * h[0:N-k]
    
    # This is O(N^2) but with small constant.
    
    # Let's implement this.
    
    max_sq = np.max(sq_sidelobes)
    if max_sq == 0:
        return 0.0, np.zeros(N)
    
    shifted = beta * (sq_sidelobes - max_sq)
    exp_shifted = np.exp(shifted)
    S = np.sum(exp_shifted)
    
    if S == 0:
        obj = max_sq
        grad = np.zeros(N)
    else:
        obj = max_sq + np.log(S) / beta
        
        # w_k corresponds to c_k (index k-1 in sq_sidelobes)
        # w_k = exp(beta * c_k^2) * 2 * c_k / S
        # Note: c_k = ac[1:N][k-1]
        c_k = ac[1:N]
        w = exp_shifted * 2.0 * c_k / S
        
        grad = np.zeros(N)
        for k in range(1, N):
            wk = w[k-1]
            if wk == 0:
                continue
            # h_{j+k}
            grad[0:N-k] += wk * h[k:N]
            # h_{j-k}
            grad[k:N] += wk * h[0:N-k]
            
        grad /= beta
        
    return obj, grad

def _local_search_sa_fast(h, time_limit=5.0, T0=None, cooling_rate=0.9995):
    """
    Simulated Annealing local search with O(N) incremental autocorrelation updates.
    """
    start_time = time.time()
    # Initial autocorrelation
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    if T0 is None:
        T0 = max(current_val * 0.1, 0.01)
    
    h = h.copy()
    ac = ac.copy()
    
    for it in range(1000000): # Large cap
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        old_h_i = h[i]
        
        # Modify ac
        ac_orig = ac.copy()
        
        # Apply update
        if i > 0:
            ac[1 : i+1] += -2.0 * h[i] * h[i-1::-1]
        if i < N-1:
            ac[1 : N-i] += -2.0 * h[i] * h[i+1:]
            
        new_val = _max_sidelobe(ac)
        
        delta = new_val - current_val
        T = T0 * (cooling_rate ** it)
        
        if delta < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta / T)):
            h[i] = -h[i]
            current_val = new_val
            # ac is already updated
        else:
            # Revert
            ac[:] = ac_orig
            # h[i] is still old_h_i

    return h, current_val

def _greedy_polish(h, time_limit=2.0):
    """
    Strict greedy search using fast updates.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            ac_orig = ac.copy()
            
            # Apply update
            if i > 0:
                ac[1 : i+1] += -2.0 * h[i] * h[i-1::-1]
            if i < N-1:
                ac[1 : N-i] += -2.0 * h[i] * h[i+1:]
                
            new_val = _max_sidelobe(ac)
            
            if new_val < current_val:
                h[i] = -h[i]
                current_val = new_val
                improved = True
            else:
                ac[:] = ac_orig
                # h[i] unchanged
                
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Soft-Max Objective
    num_continuous_starts = 5
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 20.0:
            break
        
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        try:
            # Use Soft-Max objective
            res = minimize(_objective_soft_max, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 20})
        except Exception:
            continue
        
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        if best_h is not None and time.time() < deadline - 10.0:
            time_for_sa = min(4.0, max(1.0, (deadline - 10.0 - time.time()) / max(1, num_continuous_starts - i)))
            
            h_sa, val_sa = _local_search_sa_fast(best_h.copy(), time_limit=time_for_sa)
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
    
    if best_h is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time > 2.0:
            h_final, val_final = _local_search_sa_fast(best_h.copy(), time_limit=remaining_time)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    if best_h is not None and time.time() < deadline - 1.0:
        h_polish, val_polish = _greedy_polish(best_h.copy(), time_limit=min(2.0, deadline - 1.0 - time.time()))
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
        
    if best_h is None:
        return np.ones(N)
        
    return best_h
