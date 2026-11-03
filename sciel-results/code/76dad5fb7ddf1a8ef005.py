"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Work in cosine basis: h[i] = 0.5 + sum_{j=1}^{N//2} a_j * cos(2*pi*j*i/N)
    # This automatically gives sum(h) = N/2 (DC fixed at 0.5)
    # a_j are free variables, but we need 0 <= h[i] <= 1
    
    M = N // 2  # number of cosine modes (j=1 to M)
    
    # Build cosine matrix: C[i,j] = cos(2*pi*j*i/N) for j=1..M
    i_idx = np.arange(N)
    j_idx = np.arange(1, M + 1)
    C = np.cos(2 * np.pi * np.outer(i_idx, j_idx) / N)  # shape (N, M)
    
    def h_from_a(a):
        return 0.5 + C @ a
    
    def compute_Q(h):
        """Compute circular autocorrelation Q(k) = sum_i h[i]*h[(i+k)%N] for k=0..N-1"""
        # Use FFT for efficiency
        H = np.fft.fft(h)
        power = np.abs(H) ** 2
        Q = np.fft.ifft(power).real  # Q[k] = sum_i h[i]*h[(i+k)%N]
        return Q
    
    def soft_min(Q_pos, beta):
        """Soft minimum of Q_pos (positive lags k=1..N-1)"""
        # Use log-sum-exp for numerical stability
        m = np.min(Q_pos)
        return m + np.log(np.sum(np.exp(beta * (Q_pos - m)))) / beta
    
    def objective(a, beta):
        h = h_from_a(a)
        Q = compute_Q(h)
        Q_pos = Q[1:]  # positive lags
        return -soft_min(Q_pos, beta)  # minimize negative soft-min = maximize soft-min
    
    def grad_objective(a, beta):
        h = h_from_a(a)
        Q = compute_Q(h)
        Q_pos = Q[1:]
        
        # Gradient of soft_min w.r.t. Q_pos
        m = np.min(Q_pos)
        exp_terms = np.exp(beta * (Q_pos - m))
        weights = exp_terms / np.sum(exp_terms)
        
        # d(soft_min)/dQ_pos = weights
        # dQ_pos/da: Q[k] = sum_i h[i]*h[(i+k)%N]
        # dQ[k]/dh[i] = h[(i+k)%N] + h[(i-k)%N] (for k != 0)
        # For k=1..N-1: dQ[k]/dh[i] = h[(i+k)%N] + h[(i-k)%N]
        
        # d(objective)/da = -sum_k weights[k] * dQ[k]/da
        # dQ[k]/da = C^T * (dQ[k]/dh) where dQ[k]/dh[i] = h[(i+k)%N] + h[(i-k)%N]
        
        # Efficiently: for each k, the gradient contribution is weights[k] * (h shifted + h shifted back)
        # Sum over k: sum_k weights[k] * (h[(i+k)%N] + h[(i-k)%N])
        # = sum_k weights[k]*h[(i+k)%N] + sum_k weights[k]*h[(i-k)%N]
        # = (weights * h) circular conv + (weights * h) circular conv (since weights is symmetric-ish)
        
        # Actually: sum_k w[k] * h[(i+k)%N] = (w * h)[i] (circular convolution)
        # sum_k w[k] * h[(i-k)%N] = (w * h)[i] as well (by change of variable)
        # So total = 2 * (w * h)[i]
        
        w = np.zeros(N)
        w[1:] = weights  # w[k] for k=1..N-1
        
        # Circular convolution w * h
        W = np.fft.fft(w)
        H = np.fft.fft(h)
        conv = np.fft.ifft(W * H).real
        
        # dQ_pos/da = C^T @ (2 * conv) ... wait, let me be more careful
        # dQ[k]/dh[i] for k>=1: h[(i+k)%N] + h[(i-k)%N]
        # d(objective)/dh[i] = -sum_{k=1}^{N-1} w[k] * (h[(i+k)%N] + h[(i-k)%N])
        # = -2 * (w * h)[i]  (since sum_k w[k]*h[(i+k)%N] = sum_k w[k]*h[(i-k)%N] by reindexing)
        
        dh = -2.0 * conv
        da = C.T @ dh
        
        return da
    
    # Initialize: small perturbation
    a = np.zeros(M)
    
    # Adaptive beta schedule
    beta_start = 5.0
    beta_end = 50.0
    n_iters = 3000
    
    best_a = a.copy()
    best_val = -soft_min(compute_Q(h_from_a(a))[1:], beta_end)
    
    # Nesterov-accelerated projected gradient in cosine space
    a_curr = a.copy()
    a_prev = a.copy()
    
    for it in range(n_iters):
        if time.time() > deadline - 10:
            break
        
        # Anneal beta
        progress = it / n_iters
        beta = beta_start + (beta_end - beta_start) * (progress ** 0.5)
        
        # Nesterov step
        momentum = (1 + np.sqrt(1 + 2 * it)) / (1 + np.sqrt(1 + 2 * (it + 1)))
        a_nest = a_curr + momentum * (a_curr - a_prev)
        
        # Gradient step
        g = grad_objective(a_nest, beta)
        lr = 0.1 / (1 + it * 0.001)
        a_new = a_nest - lr * g
        
        # Project: ensure 0 <= h[i] <= 1
        # This is a projection onto the set {a : 0 <= 0.5 + C@a <= 1}
        # i.e., -0.5 <= C@a <= 0.5
        # Simple approach: clip h and project back (approximate)
        h_new = h_from_a(a_new)
        h_clipped = np.clip(h_new, 0, 1)
        # Project back to cosine space (least squares)
        # a_proj = argmin ||C@a - (h_clipped - 0.5)||^2
        # a_proj = (C^T C)^{-1} C^T (h_clipped - 0.5)
        # Precompute C^T C inverse
        # Actually, let's just use a simple clipping in a-space with step size control
        
        # Alternative: reduce step size if constraint violated
        if np.any(h_new < -1e-10) or np.any(h_new > 1 + 1e-10):
            # Binary search for valid step
            lo, hi = 0.0, 1.0
            for _ in range(10):
                mid = (lo + hi) / 2
                a_test = a_nest + mid * (a_new - a_nest)
                h_test = h_from_a(a_test)
                if np.all(h_test >= -1e-10) and np.all(h_test <= 1 + 1e-10):
                    lo = mid
                else:
                    hi = mid
            a_new = a_nest + lo * (a_new - a_nest)
        
        a_prev = a_curr
        a_curr = a_new
        
        # Track best
        val = -soft_min(compute_Q(h_from_a(a_curr))[1:], beta_end)
        if val > best_val:
            best_val = val
            best_a = a_curr.copy()
    
    # Final polish: use best_a, do a few more iterations with high beta
    a_curr = best_a.copy()
    a_prev = a_curr.copy()
    beta = 100.0
    for it in range(500):
        if time.time() > deadline - 2:
            break
        momentum = (1 + np.sqrt(1 + 2 * it)) / (1 + np.sqrt(1 + 2 * (it + 1)))
        a_nest = a_curr + momentum * (a_curr - a_prev)
        g = grad_objective(a_nest, beta)
        lr = 0.05 / (1 + it * 0.005)
        a_new = a_nest - lr * g
        
        h_new = h_from_a(a_new)
        if np.any(h_new < -1e-10) or np.any(h_new > 1 + 1e-10):
            lo, hi = 0.0, 1.0
            for _ in range(10):
                mid = (lo + hi) / 2
                a_test = a_nest + mid * (a_new - a_nest)
                h_test = h_from_a(a_test)
                if np.all(h_test >= -1e-10) and np.all(h_test <= 1 + 1e-10):
                    lo = mid
                else:
                    hi = mid
            a_new = a_nest + lo * (a_new - a_nest)
        
        a_prev = a_curr
        a_curr = a_new
    
    h_final = h_from_a(a_curr)
    # Ensure exact constraints
    h_final = np.clip(h_final, 0, 1)
    # Adjust to make sum exactly N/2
    diff = N / 2.0 - np.sum(h_final)
    h_final += diff / N
    
    # Final clip and re-adjust
    h_final = np.clip(h_final, 0, 1)
    diff = N / 2.0 - np.sum(h_final)
    h_final += diff / N
    
    return h_final
# EVOLVE-BLOCK-END
