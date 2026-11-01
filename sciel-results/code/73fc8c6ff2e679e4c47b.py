import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 135s to ensure return.
    deadline = start_time + 135.0
    
    # Pre-compute FFT size for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def max_autocorr(h):
        """Compute the maximum autocorrelation value (excluding lag 0)."""
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        # We want the max of the sidelobes (lags 1 to N-1)
        # Note: acf[0] is the energy (sum of squares), which is constant for +/-1.
        # The task asks for the "upper bound on C3" which is typically related to the max correlation.
        # Based on the seed, it used max(acf[:N]) which includes lag 0? 
        # Wait, the seed said: "return np.max(acf[:N])". 
        # For a +/-1 sequence, acf[0] = N. The sidelobes are much smaller.
        # If the score is "upper bound on C3", and C3 is likely the max correlation coefficient or similar.
        # Let's stick to the seed's definition to be safe: max of the first N lags.
        # However, usually "autoconvolution upper bound" implies minimizing the peak sidelobe.
        # Let's check the seed again: "score is the upper bound on C3".
        # If C3 is the max correlation, minimizing it is the goal.
        # The seed maximizes acf[:N]. If acf[0] is included, it's always N. That doesn't make sense for optimization.
        # It must be the max of the *sidelobes*.
        # Let's assume the metric is max(acf[1:N]) or similar.
        # The seed code: return np.max(acf[:N])
        # If h is +/-1, acf[0] is N. acf[k] for k>0 are the correlations.
        # If the score is lower is better, and the seed returns max(acf[:N]), then if acf[0] is included,
        # the score is always N. That would be constant.
        # Therefore, the metric MUST exclude lag 0.
        # I will compute max(acf[1:N]) to be the peak sidelobe.
        return np.max(acf[1:N])

    def smooth_max_acf(h, beta):
        """
        Compute a smooth approximation of the max autocorrelation (excluding lag 0)
        and its gradient with respect to h.
        Uses log-sum-exp on the autocorrelation lags.
        """
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        
        # We care about lags 1 to N-1
        lags = acf[1:N]
        
        # Log-sum-exp trick for stability
        m = np.max(lags)
        if m == 0:
            return 0.0, np.zeros(N)
        
        # Softmax weights
        exp_lags = np.exp(beta * (lags - m))
        sum_exp = np.sum(exp_lags)
        weights = exp_lags / sum_exp
        
        # The value is (1/beta) * log(sum(exp(beta * lags))) + m
        val = m + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient of the value with respect to h
        # d/dh (sum_k w_k * acf_k) where w_k = exp(beta * acf_k) / sum(exp(beta * acf_j))
        # Note: w_k depends on h, but for the gradient of the smooth max, 
        # we can use the derivative of the log-sum-exp directly.
        # d/dh log(sum(exp(beta * acf_k))) = beta * sum_k w_k * (d acf_k / d h)
        # So grad_val = sum_k w_k * (d acf_k / d h)
        
        # How to compute d acf_k / d h?
        # acf is the circular autocorrelation of h (padded).
        # acf = ifft(|fft(h)|^2)
        # This is equivalent to the circular convolution of h with h_conj.
        # Since h is real, acf_k = sum_i h[i] h[i-k] (circular).
        # The gradient of sum_k w_k * acf_k with respect to h[i] is:
        # sum_k w_k * (h[i-k] + h[i+k]) (circular indices)
        
        # We can compute this efficiently using FFT.
        # Let W be the weights. We want g = ifft(fft(W) * (fft(h) + fft(-h)? No.))
        # Actually, the gradient of the circular correlation R_h[k] = sum h[n]h[n-k]
        # is given by the circular convolution of h with the "impulse response" of the weights?
        # Let's use the property: 
        # grad_i = sum_k w_k (h_{i-k} + h_{i+k})
        # This is the circular convolution of h with a symmetric kernel K where K[k] = w_k + w_{-k}?
        # Wait, w_k is defined for k=1..N-1. w_0 is 0 (we excluded lag 0).
        # The term is sum_{k=1}^{N-1} w_k (h_{i-k} + h_{i+k})
        # This is equivalent to the circular convolution of h with a kernel K
        # where K[k] = w_k for k=1..N-1 and K[-k] = w_k?
        # Let K be a vector of length N.
        # K[0] = 0
        # K[k] = w_k for k=1..N-1
        # K[N-k] = w_k for k=1..N-1 (since -k mod N is N-k)
        # Then grad = ifft(fft(h) * fft(K).conj())?
        # For real signals, circular convolution is ifft(fft(h) * fft(K)).
        
        K = np.zeros(N)
        # lags are acf[1:N], so w corresponds to lags 1..N-1
        K[1:N] = weights
        K[:0:-1] = weights[::-1] # K[N-1] = w_{N-1}, ..., K[1] = w_1? 
        # Wait, indices: 
        # lags[0] is acf[1]. lags[N-2] is acf[N-1].
        # weights[0] corresponds to lag 1.
        # K[1] should be weights[0].
        # K[N-1] should be weights[0] (lag -1 is same as lag N-1).
        # K[2] should be weights[1].
        # K[N-2] should be weights[1].
        # ...
        # K[k] = weights[k-1] for k=1..N-1
        # K[N-k] = weights[k-1] for k=1..N-1
        
        K[1:] = weights
        K[:0:-1] = weights[::-1]
        
        # Compute circular convolution of h and K
        grad = np.fft.ifft(np.fft.fft(h, n=n_fft) * np.fft.fft(K, n=n_fft)).real
        
        # The gradient of the value is this grad.
        # Note: The log-sum-exp value is (1/beta) log(sum(exp(beta*lags))) + m.
        # The derivative is sum w_k * grad_acf_k.
        # So the gradient is exactly `grad`.
        
        return val, grad

    def project_to_sign(x):
        """Project vector x to the closest +/-1 vector."""
        return np.sign(x)
        # Handle zeros: np.sign(0) is 0. We need +/-1.
        # If x is 0, we can flip a coin or just pick 1.
        # For stability, we can use np.where(x >= 0, 1, -1)
        
    def local_search(h, max_iters=1000):
        """Quick local search to refine the sign pattern."""
        current_h = h.copy()
        current_val = max_autocorr(current_h)
        
        for _ in range(max_iters):
            if time.time() >= deadline - 5.0:
                break
            # Try flipping a few random bits
            improved = False
            for _ in range(10):
                idx = np.random.randint(N)
                trial = current_h.copy()
                trial[idx] = -trial[idx]
                trial_val = max_autocorr(trial)
                if trial_val < current_val:
                    current_h = trial
                    current_val = trial_val
                    improved = True
                    break
            if not improved:
                # Try a swap
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if idx1 != idx2:
                    trial = current_h.copy()
                    trial[idx1] = -trial[idx1]
                    trial[idx2] = -trial[idx2]
                    trial_val = max_autocorr(trial)
                    if trial_val < current_val:
                        current_h = trial
                        current_val = trial_val
                        improved = True
            if not improved and np.random.random() < 0.1:
                # Random restart of local search neighborhood? Or just continue
                pass
                
        return current_h, current_val

    # Main optimization loop
    best_h = None
    best_val = np.inf
    
    # Use a fixed seed for reproducibility in the "construct" phase if needed, 
    # but here we use global np.random which is fine.
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() >= deadline - 30.0:
            break
            
        # Initialize x as a continuous vector
        # Start with random signs, or small noise around 0?
        # Projected gradient descent works best if we start in the interior of the convex hull.
        # Let's start with random values in [-1, 1]
        x = np.random.uniform(-1, 1, N)
        
        # Hyperparameters
        beta = 10.0  # Sharpness of log-sum-exp
        lr = 0.1     # Learning rate
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() >= deadline - 10.0:
                break
                
            # Compute smooth value and gradient
            val, grad = smooth_max_acf(x, beta)
            
            # Gradient step
            x = x - lr * grad
            
            # Project to sign constraints
            x = np.where(x >= 0, 1.0, -1.0)
            
            # Periodic evaluation of the true objective on the projected vector
            if step % 100 == 0:
                true_val = max_autocorr(x)
                if true_val < best_val:
                    best_val = true_val
                    best_h = x.copy()
                    
            # Adaptive learning rate? Maybe decay it.
            # lr = lr * 0.999
            
        # Final local search on the best found so far in this restart
        if best_h is not None:
            # Actually, we should local search the current x or the best_h from this restart
            # Let's local search the current x
            h_ls, val_ls = local_search(x, max_iters=2000)
            if val_ls < best_val:
                best_val = val_ls
                best_h = h_ls

    # If we didn't find anything (shouldn't happen), return random
    if best_h is None:
        return np.random.choice([1, -1], N)
        
    return best_h
# EVOLVE-BLOCK-END
