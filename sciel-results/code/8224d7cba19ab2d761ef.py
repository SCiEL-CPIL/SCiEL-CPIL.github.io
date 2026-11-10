import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    def compute_score(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Exact calculation for small n, or vectorized for larger
        # For n up to ~1000, vectorized is fine
        # P = sum p_k p_l P(X1+X2 < 2s_l - s_k)
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        p_prefix = np.cumsum(p_sorted)
        
        T = 2.0 * support[None, :] - support[:, None]
        idx = np.searchsorted(s_sorted, T, side='left')
        
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        prob_z_less = p_prefix_pad[idx]
        
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        return total_prob

    def optimize_grid(N, L, time_budget, rng):
        # Grid setup
        s = np.linspace(0, L, N)
        p = np.ones(N) / N
        
        # FFT size
        # Max index for X1+X2+X3 is 3*(N-1) approx 3N
        # We need to handle 2X4 - (X2+X3).
        # 2X4 index up to 2N. X2+X3 index up to 2N.
        # Result index range roughly [-2N, 2N].
        # We use circular convolution with size M >= 4N to avoid wrap-around into the region of interest.
        M = 1
        while M < 4 * N:
            M *= 2
            if M > 4 * N + 100: break
        
        def get_gradient(p_curr):
            p_curr = p_curr / np.sum(p_curr)
            
            # 1. Compute V(l) = P(X1+X2+X3 < 2s_l)
            # PMF of X1+X2+X3
            F_p = np.fft.fft(p_curr, M)
            F_sum3 = F_p ** 3
            P_sum3 = np.real(np.fft.ifft(F_sum3))
            P_sum3 = np.maximum(P_sum3, 0)
            
            # CDF of Sum3
            CDF_sum3 = np.cumsum(P_sum3)
            
            # V(l) corresponds to index 2*l in the grid
            # s_l = l * (L/(N-1)). The grid for sums is in units of delta = L/(N-1).
            # So index in P_sum3 corresponding to value 2*s_l is 2*l.
            V = np.zeros(N)
            # Be careful with indices out of bounds
            idx_V = np.minimum(2 * np.arange(N), M - 1)
            V = CDF_sum3[idx_V - 1] if N > 0 else 0
            # If idx is 0, prob is 0.
            V = np.where(idx_V > 0, CDF_sum3[idx_V - 1], 0.0)
            # Actually, strict inequality < 2s_l.
            # CDF[k] = P(Z <= k*delta).
            # We want P(Z < 2l*delta) = CDF[2l-1].
            
            # 2. Compute H(m) = P(s_m + X2 + X3 < 2X4) = P(s_m < 2X4 - X2 - X3)
            # PMF of A = X2+X3
            F_sum2 = F_p ** 2
            P_sum2 = np.real(np.fft.ifft(F_sum2))
            P_sum2 = np.maximum(P_sum2, 0)
            
            # PMF of 2X4
            P_2X = np.zeros(M)
            P_2X[2 * np.arange(N)] = p_curr
            
            # PMF of -A
            # A has mass at k. -A has mass at -k.
            # In circular array of size M, -k is M-k.
            P_negA = np.zeros(M)
            # P_sum2 is non-zero for k in [0, 2N].
            # We need to map k -> M-k.
            # For k=0, M-0 = M (which is 0 in circular).
            # So P_negA[(M - k) % M] = P_sum2[k].
            k_indices = np.arange(M)
            # Only non-zero parts of P_sum2
            nz_k = np.where(P_sum2 > 1e-9)[0]
            P_negA[(M - nz_k) % M] = P_sum2[nz_k]
            
            # Convolution P_2X * P_negA -> P_Y
            F_Y = np.fft.fft(P_2X) * np.fft.fft(P_negA)
            P_Y = np.real(np.fft.ifft(F_Y))
            P_Y = np.maximum(P_Y, 0)
            
            # CDF of Y
            # We want P(Y < s_m) = P(Y < m * delta).
            # Y index for value m*delta is M + m (since we shifted by M).
            # Wait, did we shift?
            # P_negA has mass at M-k. P_2X has mass at 2j.
            # Sum index is 2j + M - k = M + (2j - k).
            # Value is (2j - k) delta.
            # So index in P_Y corresponding to value v is M + v.
            # We want CDF at value m. So index M + m.
            # P(Y < m) = sum_{k < M+m} P_Y[k].
            
            CDF_Y = np.cumsum(P_Y)
            
            H = np.zeros(N)
            idx_H = M + np.arange(N)
            # Ensure idx_H is within [1, M]
            # M + m is always >= M.
            # CDF_Y is 0-indexed. CDF_Y[i] = sum_{k=0}^i P_Y[k].
            # We want sum_{k < M+m} P_Y[k] = CDF_Y[M+m - 1].
            idx_H_cdf = idx_H - 1
            # Clip to valid range [0, M-1]
            idx_H_cdf = np.clip(idx_H_cdf, 0, M - 1)
            H = CDF_Y[idx_H_cdf]
            
            grad = 3 * H + V
            return grad

        best_score = -1
        best_p = p.copy()
        
        lr = 0.01
        momentum = 0.5
        v_mom = np.zeros(N)
        
        # Simulated Annealing / Adam-like
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_t = np.zeros(N)
        v_t = np.zeros(N)
        t = 0
        
        time_limit = time_budget
        while time.time() < start_time + time_limit:
            p = np.maximum(p, 0)
            p = p / np.sum(p)
            
            score = compute_score(s, p)
            if score > best_score:
                best_score = score
                best_p = p.copy()
            
            grad = get_gradient(p)
            
            t += 1
            m_t = beta1 * m_t + (1 - beta1) * grad
            v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)
            m_hat = m_t / (1 - beta1 ** t)
            v_hat = v_t / (1 - beta2 ** t)
            
            p_new = p + lr * m_hat / (np.sqrt(v_hat) + eps)
            
            p_new = np.maximum(p_new, 0)
            p_sum = np.sum(p_new)
            if p_sum < 1e-12:
                p_new = np.ones(N) / N
            else:
                p_new = p_new / p_sum
            
            p = p_new
            
            if t % 100 == 0:
                # Decay learning rate
                lr *= 0.95
        
        return s, best_p, best_score

    rng = np.random.default_rng(42)
    
    best_score_overall = -1
    best_support_overall = None
    best_weights_overall = None
    
    # Try different grid configurations
    configs = [
        (128, 1.0),
        (256, 1.0),
        (128, 2.0),
        (256, 2.0),
        (64, 0.5),
    ]
    
    for N, L in configs:
        if time.time() > deadline - 30:
            break
        time_budget = min(60, (deadline - time.time()) / 2)
        s, p, score = optimize_grid(N, L, time_budget, rng)
        
        if score > best_score_overall:
            best_score_overall = score
            best_support_overall = s
            best_weights_overall = p
    
    if best_support_overall is None:
        best_support_overall = np.array([0.0])
        best_weights_overall = np.array([1.0])
        
    # Prune zeros? No, keep full grid.
    return best_support_overall.flatten(), best_weights_overall.flatten()
