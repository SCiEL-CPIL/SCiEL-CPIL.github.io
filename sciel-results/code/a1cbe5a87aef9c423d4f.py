import numpy as np
import time
import random

N = 600
DEADLINE = 145

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv(h):
        """Compute full linear autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h."""
        conv = compute_autoconv(h)
        return np.max(conv)
    
    def grad_autoconv_max_analytic(h):
        """
        Compute the subgradient of max(autoconv) w.r.t. h.
        
        The autoconvolution is conv[k] = sum_i h[i] * h[i+k] (with h=0 outside [0,N-1]).
        The max is achieved at some lag k*. The subgradient is the gradient of conv[k*] w.r.t. h.
        
        d(conv[k])/d(h[j]) = h[j+k] + h[j-k]  (with h out of bounds = 0)
        
        Note: due to symmetry of autoconvolution, conv[k] = conv[2N-2-k] for the linear convolution.
        In our FFT setup with size=2N, the convolution indices are 0 to 2N-1.
        For k in [0, N-1]: conv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
        For k in [N, 2N-2]: by symmetry, conv[k] = conv[2N-2-k]
        For k = 2N-1: conv[2N-1] = 0 (out of range)
        
        So we just need to find k* = argmax(conv) and compute the gradient.
        """
        conv = compute_autoconv(h)
        k_star = np.argmax(conv)
        
        # The gradient of conv[k_star] w.r.t. h[j]
        # conv[k] = sum_i h[i] * h[i+k] for k >= 0
        # d/dh[j] = h[j+k] + h[j-k] (with appropriate bounds)
        
        # But wait: in the FFT-based linear convolution with padding,
        # conv[m] for m in [0, 2N-1] corresponds to the linear convolution.
        # For m < N: conv[m] = sum_{i=0}^{N-1-m} h[i] * h[i+m]
        # For m >= N: conv[m] = sum_{i=m-N+1}^{N-1} h[i] * h[i+m-2N+1]  -- this is getting confusing.
        
        # Let me just use the direct formula:
        # The linear autoconvolution at lag L (where L ranges from -(N-1) to N-1) is:
        # (h*h)[L] = sum_i h[i] * h[i+L]
        # In our FFT output, the index mapping is:
        # conv_fft[m] for m in [0, N-1] corresponds to lag L = m
        # conv_fft[m] for m in [N, 2N-1] corresponds to lag L = m - 2N (negative lags)
        
        # So k_star in [0, 2N-1] maps to:
        if k_star < N:
            L = k_star
        else:
            L = k_star - 2 * N
        
        # Now compute d(conv[L])/d(h[j]) for j in [0, N-1]
        # conv[L] = sum_i h[i] * h[i+L]
        # d/dh[j] = h[j+L] + h[j-L]  (with h out of [0,N-1] = 0)
        
        grad = np.zeros(N)
        
        # h[j+L] term: valid when 0 <= j+L <= N-1
        if L >= 0:
            # j+L in [0, N-1] => j in [0, N-1-L]
            if N - 1 - L >= 0:
                grad[:N-L] += h[L:N]
        else:
            # j+L in [0, N-1] => j in [-L, N-1]
            if N - 1 + L >= 0:
                grad[-L:N] += h[:L]  # h[j+L] where j from -L to N-1, so h[0] to h[N-1+L]
        
        # h[j-L] term: valid when 0 <= j-L <= N-1
        if L <= 0:
            # j-L in [0, N-1] => j in [L, N-1+L]
            if N - 1 + L >= 0:
                grad[L:N] += h[:N+L]  # h[j-L] where j from L to N-1+L, so h[0] to h[N-1]
        else:
            # j-L in [0, N-1] => j in [L, N-1+L]
            if N - 1 + L >= 0:
                grad[L:N] += h[:N+L]  # h[j-L] where j from L to N-1, so h[0] to h[N-1-L]
        
        # Wait, I need to be more careful. Let me redo this.
        # conv[L] = sum_{i: 0<=i<N, 0<=i+L<N} h[i] * h[i+L]
        # d/dh[j] conv[L]:
        #   - h[j] appears in the term where i=j: contributes h[j+L] (if 0<=j+L<N)
        #   - h[j] appears in the term where i+L=j, i.e., i=j-L: contributes h[j-L] (if 0<=j-L<N)
        
        # So: d/dh[j] = (h[j+L] if 0<=j+L<N else 0) + (h[j-L] if 0<=j-L<N else 0)
        
        grad = np.zeros(N)
        
        # First term: h[j+L] for j where 0 <= j+L < N
        if L >= 0:
            # j+L < N => j < N-L, and j >= 0
            # So j in [0, N-L-1]
            if N - L > 0:
                grad[:N-L] += h[L:N]  # h[L], h[L+1], ..., h[N-1] assigned to grad[0], grad[1], ...
        else:
            # j+L < N => j < N-L, and j >= 0, but L < 0 so N-L > N
            # j+L >= 0 => j >= -L
            # So j in [max(0, -L), N-1]
            if -L < N:
                grad[-L:] += h[:N+L]  # h[0], ..., h[N-1+L] assigned to grad[-L], ...
        
        # Second term: h[j-L] for j where 0 <= j-L < N
        if L >= 0:
            # j-L >= 0 => j >= L
            # j-L < N => j < N+L, but j <= N-1, so j in [L, N-1]
            if L < N:
                grad[L:] += h[:N-L]  # h[0], ..., h[N-L-1] assigned to grad[L], ...
        else:
            # j-L >= 0 => j >= L, but L < 0, so j >= 0
            # j-L < N => j < N+L
            # So j in [0, N+L-1]
            if N + L > 0:
                grad[:N+L] += h[-L:]  # h[-L], ..., h[N-1] assigned to grad[0], ...
        
        return grad, k_star
    
    def ipf_step(h, target_conv, temperature=1.0, n_iters=50):
        """
        Iterative Proportional Fitting to match a target autoconvolution.
        
        Given current h, we want to find h' such that autoconv(h') ≈ target_conv.
        IPF alternates between updating rows and columns of a "matrix" representation.
        
        For autoconvolution, we can think of it as:
        conv[k] = sum_i h[i] * h[i+k]
        
        This is like a quadratic form. A simpler approach:
        Use multiplicative updates: h_new[j] = h[j] * exp(-eta * grad[j])
        where grad is the subgradient of the max, but scaled to reduce the peak.
        
        Actually, let's use a different approach: 
        We want to minimize the L1 norm of the autoconvolution, or at least reduce the peak.
        The idea is to use the subgradient to guide updates, but with a multiplicative (exponential) update
        to maintain non-negativity and the simplex constraint.
        """
        h_current = h.copy()
        h_current /= np.sum(h_current)
        
        for it in range(n_iters):
            if time_left() < 5:
                break
            
            grad, k_star = grad_autoconv_max_analytic(h_current)
            
            # Multiplicative update: h_new[j] = h[j] * exp(-eta * grad[j])
            # This naturally maintains non-negativity.
            # Then normalize to sum to 1.
            
            # Use a small eta to avoid numerical issues
            eta = 0.1
            h_new = h_current * np.exp(-eta * grad)
            
            # Normalize
            s = np.sum(h_new)
            if s < 1e-10:
                break
            h_new /= s
            
            # Check if we improved
            val_old = compute_autoconv_max(h_current)
            val_new = compute_autoconv_max(h_new)
            
            if val_new < val_old:
                h_current = h_new
            else:
                # Reduce eta
                eta *= 0.5
                if eta < 1e-10:
                    break
                h_new = h_current * np.exp(-eta * grad)
                s = np.sum(h_new)
                if s < 1e-10:
                    break
                h_new /= s
        
        return h_current
    
    def projected_grad_descent(h0, lr_init=0.01, n_iter=200):
        """Projected gradient descent with backtracking line search."""
        h = h0.copy()
        h /= np.sum(h)
        current_val = compute_autoconv_max(h)
        lr = lr_init
        
        for it in range(n_iter):
            if time_left() < 10:
                break
            
            # Compute gradient
            grad, _ = grad_autoconv_max_analytic(h)
            
            # Projected gradient step: minimize <grad, h_new> subject to h_new >= 0, sum(h_new) = 1
            # Use the standard simplex projection
            
            h_temp = h - lr * grad
            
            # Project onto simplex {x >= 0, sum(x) = 1}
            h_sorted = np.sort(h_temp)[::-1]
            css = np.cumsum(h_sorted)
            rho = np.nonzero(h_sorted * np.arange(1, N + 1) > (css[-1] - 1))[0][-1]
            tau = (css[rho] - 1) / (rho + 1)
            h_new = np.maximum(h_temp - tau, 0)
            
            # Line search: reduce lr if needed
            new_val = compute_autoconv_max(h_new)
            
            backtrack_count = 0
            while new_val > current_val + 1e-12 and lr > 1e-10 and backtrack_count < 20:
                lr *= 0.5
                h_temp = h - lr * grad
                h_sorted = np.sort(h_temp)[::-1]
                css = np.cumsum(h_sorted)
                rho = np.nonzero(h_sorted * np.arange(1, N + 1) > (css[-1] - 1))[0][-1]
                tau = (css[rho] - 1) / (rho + 1)
                h_new = np.maximum(h_temp - tau, 0)
                new_val = compute_autoconv_max(h_new)
                backtrack_count += 1
            
            if new_val < current_val:
                h = h_new
                current_val = new_val
                if lr < lr_init:
                    lr = min(lr * 1.05, lr_init)
            else:
                lr *= 0.9
        
        return h, current_val
    
    def simulated_annealing(h0, n_iter=500, T_init=0.01, T_final=1e-6):
        """Simulated annealing for fine-tuning."""
        h = h0.copy()
        h /= np.sum(h)
        current_val = compute_autoconv_max(h)
        best_h = h.copy()
        best_val = current_val
        
        for it in range(n_iter):
            if time_left() < 10:
                break
            
            T = T_init * (T_final / T_init) ** (it / n_iter)
            
            # Random perturbation
            h_perturbed = h.copy()
            # Pick a few random indices and perturb them
            n_perturb = random.randint(1, 5)
            indices = random.sample(range(N), n_perturb)
            for idx in indices:
                h_perturbed[idx] *= random.uniform(0.5, 1.5)
            
            # Normalize
            s = np.sum(h_perturbed)
            if s < 1e-10:
                continue
            h_perturbed /= s
            
            new_val = compute_autoconv_max(h_perturbed)
            
            # Accept if better, or with probability exp(-(new_val - current_val) / T)
            if new_val < current_val or random.random() < np.exp(-(new_val - current_val) / max(T, 1e-10)):
                h = h_perturbed
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
        
        return best_h, best_val
    
    # Generate diverse initializations
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal
    for sigma in [N/12, N/10, N/8, N/6]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
    
    # U-shaped (high at edges)
    for a in [0.5, 1.0, 1.5, 2.0]:
        u_shape = 1.0 + a * ((i - center) / (N/2))**2
        inits.append(u_shape)
    
    # W-shape
    for freq in [3, 4, 5]:
        for amp in [0.5, 1.0, 1.5]:
            w_shape = 1.0 + amp * np.cos(2 * np.pi * freq * i / N)
            inits.append(np.maximum(w_shape, 0.01))
    
    # Smooth random
    for _ in range(8):
        freqs = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=4, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=4)
        amps = np.random.uniform(0.3, 1.2, size=4)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    # Triangular
    tri = 1.0 - np.abs(i - center) / (N/2)
    inits.append(np.maximum(tri, 0.01))
    
    # Gaussian
    for sigma in [N/8, N/6, N/5, N/4]:
        gauss = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
        inits.append(gauss)
    
    best_h = None
    best_val = np.inf
    
    # Evaluate initializations
    init_results = []
    for init in inits:
        if time_left() < 30:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        init_results.append((val, h.copy()))
    
    # Sort by initial value
    init_results.sort(key=lambda x: x[0])
    
    # Run gradient descent from top 5 initializations
    n_restarts = min(5, len(init_results))
    for val, h in init_results[:n_restarts]:
        if time_left() < 20:
            break
        
        h_opt, val_opt = projected_grad_descent(h, lr_init=0.005, n_iter=100)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Apply IPF/multiplicative updates to refine
    if time_left() > 20:
        h_ipf = ipf_step(best_h, None, temperature=1.0, n_iters=30)
        val_ipf = compute_autoconv_max(h_ipf)
        if val_ipf < best_val:
            best_val = val_ipf
            best_h = h_ipf.copy()
    
    # Simulated annealing for final fine-tuning
    if time_left() > 15:
        h_sa, val_sa = simulated_annealing(best_h, n_iter=200, T_init=0.005, T_final=1e-7)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa.copy()
    
    return best_h.tolist()
