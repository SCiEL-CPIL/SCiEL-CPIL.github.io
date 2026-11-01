import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # np.correlate(h, h, 'full') gives lags from -(N-1) to N-1.
    # Index N-1 corresponds to lag 0.
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _soft_max(x, k):
    """
    Soft maximum approximation using log-sum-exp.
    max(x) ~ (1/k) * log(sum(exp(k*x)))
    """
    if len(x) == 0:
        return 0.0
    # Scale for numerical stability
    max_val = np.max(x)
    if max_val > 1e10:
        return max_val
    # Avoid overflow
    x_scaled = k * (x - max_val)
    # Clip to prevent overflow in exp
    x_scaled = np.clip(x_scaled, -50, 50)
    sum_exp = np.sum(np.exp(x_scaled))
    if sum_exp == 0:
        return max_val
    return max_val + np.log(sum_exp) / k

def _objective_and_grad(x, k=10.0):
    """
    Objective function and its gradient for continuous optimization.
    x is a vector of length N, representing continuous values.
    We want to minimize max_{k!=0} |corr_k(x)|.
    Since corr_k(x) is linear in x (actually bilinear, but for fixed h it's linear? No, h is the variable).
    corr_k(x) = sum_i x[i] x[i+k].
    This is a quadratic form.
    
    Let's use a soft-max surrogate:
    f(x) = (1/k) * log( sum_{k!=0} exp(k * |corr_k(x)|) )
    
    Gradient:
    d/dx_j f(x) = (1/k) * [ sum_{k!=0} k * exp(k*|c_k|) * sign(c_k) * d(c_k)/dx_j ] / sum_{k!=0} exp(k*|c_k|)
    
    d(c_k)/dx_j = x[j+k] (if j+k in range) + x[j-k] (if j-k in range) ... wait.
    c_k = sum_{i} x[i] x[i+k]
    d(c_k)/dx_j = x[j+k] (term where i=j) + x[j-k] (term where i+k=j => i=j-k)
    
    So grad_j = (1/k) * sum_{k!=0} w_k * sign(c_k) * (x[j+k] + x[j-k])
    where w_k = exp(k*|c_k|) / sum exp(k*|c_k|)
    """
    # Compute correlations
    # We need c_k for k in -(N-1)..(N-1), k!=0
    # Using FFT for speed
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    # Power spectrum
    P = X * np.conj(X)
    # Inverse FFT to get circular correlation
    corr_circ = np.fft.irfft(P, n=n_fft)
    # corr_circ has length n_fft
    # The linear correlation c_k for k=0..N-1 is in corr_circ[0..N-1]
    # The linear correlation c_k for k=-(N-1)..-1 is in corr_circ[N_fft-N+1..N_fft-1]
    # Let's extract the lags we need.
    # Lag 0 is at index 0.
    # Lag k (positive) is at index k.
    # Lag k (negative) is at index N_fft + k.
    
    # We want all k != 0.
    # Indices for positive lags 1..N-1: 1..N-1
    # Indices for negative lags -(N-1)..-1: N_fft-(N-1) .. N_fft-1
    
    idx_pos = np.arange(1, N)
    idx_neg = np.arange(n_fft - (N-1), n_fft)
    
    c_vals = np.concatenate((corr_circ[idx_neg], corr_circ[idx_pos]))
    
    # Soft max
    # k parameter for soft max
    # We need to be careful with numerical stability
    max_c = np.max(np.abs(c_vals))
    if max_c == 0:
        return 0.0, np.zeros(N)
    
    # Scale c_vals to avoid overflow
    # We want to minimize max |c_k|
    # f = (1/k) log sum exp(k |c_k|)
    # Let's use a fixed k, say 20.
    k_soft = 20.0
    abs_c = np.abs(c_vals)
    
    # Log-sum-exp trick
    max_abs = np.max(abs_c)
    if max_abs > 0:
        shifted = k_soft * (abs_c - max_abs)
        shifted = np.clip(shifted, -50, 50)
        sum_exp = np.sum(np.exp(shifted))
        obj = max_abs + np.log(sum_exp) / k_soft
    else:
        obj = 0.0
        sum_exp = 1.0
        weights = np.ones(len(c_vals)) / len(c_vals)
    
    # Compute weights w_k = exp(k |c_k|) / sum exp(k |c_k|)
    # Using the shifted version
    if max_abs > 0:
        log_weights = shifted - np.log(sum_exp)
        weights = np.exp(log_weights)
    else:
        weights = np.ones(len(c_vals)) / len(c_vals)
    
    # Now compute gradient
    # grad_j = (1/k_soft) * sum_{k} w_k * sign(c_k) * (x[j+k] + x[j-k])
    # Note: the 1/k_soft comes from derivative of (1/k) log(...)
    # Actually, d/dx (1/k log sum exp(k|c|)) = (1/k) * sum (k exp(k|c|)/sum) * sign(c) * c'
    # = sum (exp(k|c|)/sum) * sign(c) * c'
    # So the 1/k and k cancel out.
    
    # We need to sum over all lags k (both pos and neg)
    # Let's iterate over lags
    # This is O(N^2) if done naively.
    # We can use convolution again?
    # grad = (1/k) * ( w_pos * sign(c_pos) convolved with x ) + ( w_neg * sign(c_neg) convolved with x )
    # Wait, c_k = sum_i x[i] x[i+k].
    # d(c_k)/dx_j = x[j+k] + x[j-k].
    # So grad_j = sum_k w_k sign(c_k) (x[j+k] + x[j-k])
    # This is a convolution of the "weights" with x.
    
    # Let's construct the weight sequence for convolution.
    # We have c_vals corresponding to lags: -(N-1)..-1, 1..N-1
    # Let's create an array of weights for lags -(N-1)..N-1, excluding 0.
    # w_full of length 2N-1?
    # Index mapping:
    # Lag k is at index k + (N-1) in a full correlation array of length 2N-1.
    
    w_full = np.zeros(2*N-1)
    # c_vals first part is neg lags: -(N-1)..-1
    # c_vals second part is pos lags: 1..N-1
    
    # Neg lags: k = -(N-1) .. -1
    # Indices in w_full: k + N - 1
    # k = -(N-1) -> 0
    # k = -1 -> N-2
    w_full[0:N-1] = weights[0:N-1] * np.sign(c_vals[0:N-1])
    
    # Pos lags: k = 1 .. N-1
    # Indices in w_full: k + N - 1
    # k = 1 -> N
    # k = N-1 -> 2N-2
    w_full[N:2*N-1] = weights[N-1:2*N-1] * np.sign(c_vals[N-1:2*N-1])
    
    # Now grad = convolve(w_full, x, 'full')?
    # Let's check.
    # (w * x)[j] = sum_i w[i] x[j-i]
    # We want sum_k w_k (x[j+k] + x[j-k])
    # Let's define a symmetric weight vector?
    # Actually, w_k for lag k is associated with c_k.
    # Note c_{-k} = c_k (for real x).
    # So w_{-k} = w_k and sign(c_{-k}) = sign(c_k).
    # So w_full is symmetric?
    # w_full[N-1+k] = w_k.
    # w_full[N-1-k] = w_{-k} = w_k.
    # So w_full is symmetric around N-1.
    
    # Let's just compute the gradient directly using FFT or loops if N is small enough.
    # N=400. N^2 = 160,000. This is fast in Python if vectorized.
    # But we are in a loop of optimization, so we need it to be very fast.
    # Let's use the convolution property.
    # grad_j = sum_k w_k sign(c_k) x[j+k] + sum_k w_k sign(c_k) x[j-k]
    # Let v_k = w_k sign(c_k).
    # Then grad = v * x (convolution) + v_rev * x (convolution)?
    # Actually, sum_k v_k x[j+k] is the correlation of x and v?
    # np.correlate(x, v, 'full')?
    
    # Let's try:
    # term1_j = sum_k v_k x[j+k]
    # term2_j = sum_k v_k x[j-k]
    
    # v is defined for k in -(N-1)..N-1.
    # Let's create v array of length 2N-1, centered at N-1.
    # v[N-1+k] = v_k.
    
    # We already have w_full which is v_k.
    # v = w_full.
    
    # term1 = correlate(x, v, 'full')?
    # np.correlate(a, v, 'full')[n] = sum_k a[k] v[n-k]
    # We want sum_k v_k x[j+k]. Let i = j+k => k = i-j.
    # sum_i v[i-j] x[i].
    # This is (x * v)[j] ? No, (x * v)[j] = sum_i x[i] v[j-i].
    # If v is symmetric, v[j-i] = v[i-j].
    # Since v is symmetric (w_k = w_-k and sign(c_k)=sign(c_-k)), v is even.
    # So term1_j = sum_i x[i] v[i-j] = (x * v)[j].
    # And term2_j = sum_k v_k x[j-k] = sum_i v[i-j] x[i] = (x * v)[j].
    # So grad_j = 2 * (x * v)[j]?
    
    # Let's verify with a small example.
    # x = [1, 2], N=2.
    # Lags: -1, 1.
    # c_1 = x[0]x[1] = 2. c_-1 = x[1]x[0] = 2.
    # v_1 = w_1 sign(2) = w_1. v_-1 = w_-1 sign(2) = w_1.
    # v = [w_1, 0, w_1] (indices 0,1,2 for lags -1,0,1).
    # grad_0 = v_-1 x[1] + v_1 x[-1] (0) + v_0 x[0] (0) ... wait.
    # Formula: grad_j = sum_k v_k (x[j+k] + x[j-k])
    # j=0:
    # k=-1: v_-1 (x[-1] + x[1]) = w_1 (0 + 2) = 2 w_1
    # k=0: 0
    # k=1: v_1 (x[1] + x[-1]) = w_1 (2 + 0) = 2 w_1
    # Total = 4 w_1.
    
    # Convolution (x * v)[0]:
    # x = [1, 2], v = [w_1, 0, w_1]
    # (x*v)[0] = x[0]v[0] + x[1]v[-1](0) = 1*w_1 = w_1.
    # This doesn't match.
    
    # The convolution definition in numpy:
    # np.convolve(x, v, 'full')
    # x=[1,2], v=[w,0,w]
    # result[0] = 1*w = w
    # result[1] = 1*0 + 2*w = 2w
    # result[2] = 1*w + 2*0 = w
    # result[3] = 2*w = 2w
    
    # We want grad_0 = 4w, grad_1 = 4w.
    # It seems grad = 2 * (something).
    
    # Let's just compute it directly using vectorized operations for N=400.
    # It might be fast enough.
    # grad = np.zeros(N)
    # for k in range(1, N):
    #     v_k = w_full[N-1+k]
    #     # term for x[j+k]
    #     # j ranges 0 to N-k-1
    #     # x[j+k] is x[k:]
    #     # term for x[j-k]
    #     # j ranges k to N-1
    #     # x[j-k] is x[:-k]
    #     grad[:N-k] += v_k * x[k:]
    #     grad[k:] += v_k * x[:-k]
    #     # Also for -k
    #     # v_-k = v_k
    #     # x[j-k] for k -> x[j+k] for -k?
    #     # The formula sum_k v_k (x[j+k] + x[j-k]) already accounts for both?
    #     # No, k iterates over all non-zero lags.
    #     # If we iterate k=1..N-1, we cover positive lags.
    #     # We need to add the negative lags too.
    #     # But v_-k = v_k and the terms are symmetric.
    #     # So we can just do 2 * sum_{k=1}^{N-1} v_k (x[j+k] + x[j-k])?
    #     # Wait, if k iterates 1..N-1, then -k iterates -(N-1)..-1.
    #     # So yes, we can just double the sum for k=1..N-1.
    
    # Let's implement this vectorized loop.
    grad = np.zeros(N)
    # v_k for k=1..N-1 is in w_full[N:2*N-1]
    v_pos = w_full[N:2*N-1]
    
    # We want to compute sum_{k=1}^{N-1} v_k (x[j+k] + x[j-k])
    # This can be done with slicing.
    # For each k, we add v_k * x[k:] to grad[:N-k]
    # and v_k * x[:-k] to grad[k:]
    
    # Vectorizing this loop is tricky but N=400 is small.
    # Let's try a simple loop first to ensure correctness, then optimize if needed.
    # Actually, we can use np.correlate if we construct the right kernel.
    # Let's stick to the loop for safety in this step, or use a more efficient method.
    # Given N=400, a loop of 400 iterations with vector operations is fast.
    
    for k in range(1, N):
        vk = v_pos[k-1]
        if vk == 0:
            continue
        # x[j+k] for j=0..N-k-1 is x[k:]
        grad[:N-k] += vk * x[k:]
        # x[j-k] for j=k..N-1 is x[:-k]
        grad[k:] += vk * x[:-k]
        
        # And for negative lag -k
        # v_-k = v_k
        # term is v_k (x[j-k] + x[j+k])
        # This is the same terms!
        # So we just multiply the whole thing by 2?
        # Yes, because the sum is over all k != 0.
        # Sum_{k!=0} = Sum_{k>0} + Sum_{k<0}
        # Term for k>0: v_k (x[j+k] + x[j-k])
        # Term for k<0: let m = -k > 0. v_-m (x[j-m] + x[j+m]) = v_m (x[j-m] + x[j+m])
        # So Term_k + Term_-k = 2 v_k (x[j+k] + x[j-k])
        
    grad *= 2.0
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Initial random guesses
    num_initial = 20
    for _ in range(num_initial):
        if time.time() > deadline - 10:
            break
        h = np.random.choice([-1, 1], size=N)
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    # Refine the best one using gradient-based optimization
    if best_h is not None:
        x0 = best_h.astype(float)
        
        # Try to optimize
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50, 'maxfun': 5000}
            )
            x_opt = res.x
            # Project back to +/- 1
            h_opt = np.sign(x_opt)
            # Ensure no zeros (sign(0)=0)
            h_opt[h_opt == 0] = 1
            
            score_opt = _calc_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception as e:
            # If optimization fails, stick with best_h
            pass
        
        # If time remains, try a few more random restarts and optimize them
        if time.time() < deadline - 30:
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                h = np.random.choice([-1, 1], size=N)
                score = _calc_score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
                
                x0 = best_h.astype(float)
                try:
                    res = minimize(
                        _objective_and_grad,
                        x0,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50, 'maxfun': 2000}
                    )
                    x_opt = res.x
                    h_opt = np.sign(x_opt)
                    h_opt[h_opt == 0] = 1
                    
                    score_opt = _calc_score(h_opt)
                    if score_opt < best_score:
                        best_score = score_opt
                        best_h = h_opt
                except Exception:
                    pass

    return best_h
