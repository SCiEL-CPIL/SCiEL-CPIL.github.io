import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def sinkhorn_minmax(h_init, max_iter=500, eta=0.1, tol=1e-12):
        """
        Min-max formulation: 
        Maximize over h >= 0, sum(h)=1:  l2_sq / max_conv
        (since l1=1 after normalization)
        
        Equivalently, minimize over h >= 0, sum(h)=1:  max_conv / l2_sq
        
        We use a multiplicative update approach inspired by Sinkhorn-Knopp.
        The "attacker" chooses the lag k that maximizes conv[k].
        The "defender" adjusts h to minimize max_conv / l2_sq.
        
        Update rule:
        h_i *= exp(eta * (2*h_i/l2_sq - sum_k w_k * d(conv_k)/dh_i))
        where w_k are weights on the lags (attacker's strategy)
        and d(conv_k)/dh_i = h_{i+k} + h_{i-k} (with bounds)
        """
        h = np.maximum(h_init, 0).astype(float)
        h_sum = np.sum(h)
        if h_sum < 1e-15:
            h = np.ones(N) / N
        else:
            h = h / h_sum
        
        # Attacker's strategy: uniform over all lags initially
        # The full convolution has indices 0 to 2*N-2
        # Lag k corresponds to index N-1+k, so k ranges from -(N-1) to N-1
        num_lags = 2 * N - 1
        w = np.ones(num_lags) / num_lags
        
        best_h = h.copy()
        best_score = score(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            l1 = np.sum(h)
            if l1 < 1e-15:
                break
            l2_sq = np.sum(h**2)
            if l2_sq < 1e-15:
                break
            
            conv = np.convolve(h, h, mode='full')
            
            # Attacker update: w_k proportional to exp(conv_k / temperature)
            # Use a soft-max to concentrate on the peak
            temp = 0.1
            exp_conv = np.exp(conv / temp - np.max(conv) / temp)
            w = exp_conv / np.sum(exp_conv)
            
            # Defender update: compute gradient of log(max_conv) w.r.t. h
            # Using the weighted sum of conv_lags as a smooth approximation
            # d/dh_i [sum_k w_k conv_k] = sum_k w_k (h_{i+k} + h_{i-k})
            # where conv_k = sum_j h_j h_{j+k}
            
            grad_max = np.zeros(N)
            for k in range(-N + 1, N):
                idx = N - 1 + k  # index in full convolution
                wk = w[idx]
                if wk < 1e-15:
                    continue
                # Contribution to grad from lag k: h_{i+k} + h_{i-k}
                for i in range(N):
                    if 0 <= i + k < N:
                        grad_max[i] += wk * h[i + k]
                    if 0 <= i - k < N:
                        grad_max[i] += wk * h[i - k]
            
            # Gradient of log(ratio) = d log(l2_sq) - d log(max_conv)
            # = 2*h/l2_sq - grad_max / (weighted_max_conv)
            weighted_max_conv = np.dot(w, conv)
            if weighted_max_conv < 1e-15:
                break
            
            grad_log = 2.0 * h / l2_sq - grad_max / weighted_max_conv
            
            # Multiplicative update
            h_new = h * np.exp(eta * grad_log)
            h_new = np.maximum(h_new, 0)
            h_new_sum = np.sum(h_new)
            if h_new_sum < 1e-15:
                break
            h = h_new / h_new_sum
            
            s = score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            
            if it > 20 and abs(s - best_score) < tol:
                break
        
        return best_h, best_score

    def coordinate_descent(h_init, max_iter=200, tol=1e-10):
        """Simple coordinate descent for polishing."""
        h = np.maximum(h_init, 0).astype(float)
        h_sum = np.sum(h)
        if h_sum < 1e-15:
            return h, 0.0
        h = h / h_sum
        
        best_h = h.copy()
        best_s = score(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            for i in range(N):
                if h[i] < 1e-15:
                    # Try adding a small amount
                    for delta in [0.001, 0.005, 0.01]:
                        h_trial = h.copy()
                        h_trial[i] += delta
                        s = score(h_trial)
                        if s > best_s:
                            best_s = s
                            best_h = h_trial.copy()
                            h = h_trial.copy()
                            improved = True
                            break
                else:
                    # Try adjusting
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02, 0.001, -0.001]:
                        h_trial = h.copy()
                        h_trial[i] = max(0, h_trial[i] + delta)
                        s = score(h_trial)
                        if s > best_s:
                            best_s = s
                            best_h = h_trial.copy()
                            h = h_trial.copy()
                            improved = True
                            break
            if not improved:
                break
        
        return best_h, best_s

    # Start with various initializations
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Initial guess 1: uniform
    h_init = np.ones(N) / N
    h_opt, s = sinkhorn_minmax(h_init, max_iter=300)
    if s > best_score:
        best_score = s
        best_h = h_opt.copy()
    
    # Initial guess 2: concentrated on few points (Sidon-like)
    for seed_set in [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 2, 5, 11, 19, 24, 31],
        [0, 3, 7, 12, 18, 25, 33, 42],
    ]:
        if time.time() > deadline - 10.0:
            break
        h_init = np.zeros(N)
        for idx in seed_set:
            h_init[idx] = 1.0 / len(seed_set)
        h_opt, s = sinkhorn_minmax(h_init, max_iter=300)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Random restarts
    rng = np.random.RandomState(42)
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
        
        # Random sparse initialization
        k = rng.randint(3, 15)
        S = rng.choice(N, k, replace=False)
        h_init = np.zeros(N)
        h_init[S] = 1.0 / k
        # Add small noise
        h_init += rng.random(N) * 0.01
        h_init = np.maximum(h_init, 0)
        h_init = h_init / np.sum(h_init)
        
        h_opt, s = sinkhorn_minmax(h_init, max_iter=200)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Polish with coordinate descent
    h_polished, s_polished = coordinate_descent(best_h, max_iter=100)
    if s_polished > best_score:
        best_score = s_polished
        best_h = h_polished.copy()
    
    # Final round of sinkhorn from polished solution
    h_final, s_final = sinkhorn_minmax(best_h, max_iter=200)
    if s_final > best_score:
        best_score = s_final
        best_h = h_final.copy()
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
