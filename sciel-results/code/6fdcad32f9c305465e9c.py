import numpy as np
import time

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def normalize(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += np.random.normal(0, 1e-6, N)
            s = np.sum(h)
        return h / s

    def optimize_from(h_init, max_iter=5000):
        h = h_init.copy().astype(float)
        h = normalize(h)
        
        # Adam optimizer state
        m = np.zeros_like(h)
        v = np.zeros_like(h)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        # Smoothness parameter beta for log-sum-exp
        beta = 10.0
        beta_end = 1000.0
        
        best_local_val = np.inf
        best_local_h = h.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            # Compute convolution
            conv = np.convolve(h, h)
            
            # Compute smooth max gradient
            # To avoid overflow, subtract max
            max_conv = np.max(conv)
            # If beta is large, exp(beta * conv) can overflow.
            # We compute p_k = exp(beta * (C_k - max_C)) / sum(...)
            # Note: max C is max_conv.
            shifted_conv = conv - max_conv
            # Scale by beta
            # If beta * max_shifted is very negative, it underflows to 0, which is fine.
            # If beta is small, it's fine.
            # We need to be careful if beta * (C_k - max_C) is large positive? No, max is 0.
            # So exp(beta * (C_k - max_C)) <= 1.
            
            # Adaptive beta: start small, increase to sharpen
            # beta_t = beta_start * (beta_end / beta_start) ** (it / max_iter)
            # Or simply linear/increase. Let's try a schedule.
            if it < max_iter // 2:
                curr_beta = 5.0
            elif it < 3 * max_iter // 4:
                curr_beta = 50.0
            else:
                curr_beta = 500.0
                
            # Calculate weights p_k
            # exp(curr_beta * shifted_conv)
            # Since shifted_conv <= 0, exp is <= 1.
            exp_terms = np.exp(curr_beta * shifted_conv)
            sum_exp = np.sum(exp_terms)
            if sum_exp == 0:
                p = np.zeros_like(conv)
                p[np.argmax(conv)] = 1.0
            else:
                p = exp_terms / sum_exp
            
            # Gradient of smoothed objective J(h) = (1/beta) * log(sum(exp(beta*C)))
            # dJ/dh_m = (1/beta) * sum_k p_k * (d C_k / d h_m)
            # d C_k / d h_m = 2 * h_{k-m} (if valid index)
            # So grad[m] = (2/beta) * sum_k p_k * h_{k-m}
            
            # We can compute this as a convolution of p with h?
            # Let's check indices.
            # C_k = sum_i h_i h_{k-i}
            # d C_k / d h_m = h_{k-m} + h_{k-m} (if m != k-m) or 2 h_m (if m == k-m)
            # Actually, d/dh_m (h_m h_{k-m}) + d/dh_m (h_{k-m} h_m) ... 
            # C_k = sum_i h_i h_{k-i}.
            # Term involving h_m: i=m -> h_m h_{k-m}. i=k-m -> h_{k-m} h_m.
            # If m != k-m, there are two terms: h_{k-m} and h_{k-m}. Sum = 2 h_{k-m}.
            # If m == k-m, i=m=k-m, term is h_m^2. Deriv = 2 h_m.
            # So in all cases, d C_k / d h_m = 2 h_{k-m} (where h_j=0 if j out of bounds).
            
            # So grad[m] = (2/beta) * sum_k p_k h_{k-m}
            # This is essentially (2/beta) * (p * h)[m] ?
            # Convolution (p * h)[m] = sum_k p_k h_{m-k}.
            # We need sum_k p_k h_{k-m}.
            # Let j = k-m. k = j+m.
            # sum_j p_{j+m} h_j.
            # This is a cross-correlation or a flipped convolution.
            # Let q = flip(p).
            # (q * h)[m] = sum_j q_j h_{m-j} = sum_j p_{L-1-j} h_{m-j}.
            # This doesn't look immediately like standard np.convolve without shifts.
            
            # Simpler: use the subgradient logic but weighted by p.
            # g = zeros(N)
            # for k in range(len(conv)):
            #   if p[k] > 0:
            #     idx = k - np.arange(N)
            #     valid = (idx >= 0) & (idx < N)
            #     g[valid] += p[k] * 2.0 * h[idx[valid]]
            
            # This loop is O(N * len(conv)) = O(N^2) = 160,000 ops. Acceptable.
            # But we can vectorize.
            # We want g[m] = (2/beta) * sum_k p_k h_{k-m}
            
            # Construct a matrix P where P[k, m] = p[k] if 0 <= k-m < N else 0.
            # Then g = (2/beta) * P.T @ h.
            
            # Or simpler:
            # Note that sum_k p_k h_{k-m} is the correlation of p and h.
            # np.correlate(p, h, mode='full')?
            # Let's stick to the vectorized loop over k or use FFT if needed.
            # N=400, len(conv)=799.
            # Loop over k is fine.
            
            g = np.zeros(N)
            # Optimization: only iterate over k where p[k] is significant
            # p is sparse-ish if beta is large.
            # Threshold p
            threshold = 1e-6
            active_k = np.where(p > threshold)[0]
            
            for k in active_k:
                idx = k - np.arange(N)
                valid = (idx >= 0) & (idx < N)
                g[valid] += p[k] * 2.0 * h[idx[valid]]
            
            g /= curr_beta
            
            # Adam update
            t = it + 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Learning rate
            # Start with 0.1, decay?
            lr = 0.1 * (1.0 / (1.0 + it * 1e-4))
            
            h_new = h - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Project to sum=1
            h_new = normalize(h_new)
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(h_new)):
                h_new = h
                lr *= 0.1
                h_new = h - lr * m_hat / (np.sqrt(v_hat) + eps)
                h_new = normalize(h_new)
            
            h = h_new
            
            # Evaluate current value (using true max, not smoothed)
            conv_curr = np.convolve(h, h)
            max_curr = np.max(conv_curr)
            val_curr = 2.0 * N * max_curr # Since sum(h)=1
            
            if val_curr < best_local_val:
                best_local_val = val_curr
                best_local_h = h.copy()
                
        return best_local_h, best_local_val

    # Initializations
    inits = []
    
    # Random
    rng = np.random.RandomState(42)
    for _ in range(10):
        inits.append(rng.uniform(-1, 1, N))
        
    # Geometric progressions
    for r in [0.9, 0.95, 0.99, 1.1, 0.8]:
        h_geom = np.power(r, np.arange(N))
        inits.append(h_geom)
        
    # Cosines with phase
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 7]:
        for phase in [0, np.pi/2, np.pi]:
            inits.append(np.cos(2*np.pi*freq*t/N + phase))
            
    # Linear ramp
    inits.append(np.linspace(1, -1, N))
    inits.append(np.linspace(-1, 1, N))
    
    # Bimodal
    for _ in range(5):
        h = np.zeros(N)
        h[:N//2] = 1
        h[N//2:] = -1
        # Add noise
        h += rng.normal(0, 0.1, N)
        inits.append(h)
        
    # Run optimization from each init
    for h_init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Try different numbers of iterations
        if time.time() - start_time < 10:
            max_iter = 20000
        elif time.time() - start_time < 60:
            max_iter = 10000
        else:
            max_iter = 5000
            
        h_opt, val_opt_unscaled = optimize_from(h_init, max_iter)
        
        # Evaluate the result
        val_final, valid = evaluate(h_opt)
        if valid and val_final < best_value:
            best_value = val_final
            best_h = h_opt.copy()
            
        # If we found a good one, do more refinement
        if best_value < 1.2:
            if time.time() - start_time < DEADLINE - 10:
                # Refine with higher max_iter
                h_ref, _ = optimize_from(best_h, 30000)
                val_ref, valid_ref = evaluate(h_ref)
                if valid_ref and val_ref < best_value:
                    best_value = val_ref
                    best_h = h_ref.copy()
                    
    # Final check
    val_final, valid = evaluate(best_h)
    if not valid:
        best_h = np.ones(N)
        best_value = 2.0
    else:
        best_value = val_final
        
    return best_h
