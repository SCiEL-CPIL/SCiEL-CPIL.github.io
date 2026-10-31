import numpy as np
import time

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    i = np.arange(N, dtype=float)
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
    
    def compute_smooth_max(h, beta=50.0):
        """Compute smooth max of autoconvolution using log-sum-exp."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        m = np.max(conv)
        log_sum_exp = m + np.log(np.sum(np.exp((conv - m) * beta))) / beta
        return log_sum_exp
    
    def compute_grad_smooth_max(h, beta=50.0):
        """Compute gradient of smooth max autoconvolution w.r.t. h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)
        
        w_full = np.zeros(size)
        w_full[:N] = weights[:N]
        # Also include the wrap-around part
        w_full[N:] = weights[N:]
        
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        return grad_h
    
    # Generate diverse initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal with various parameters
    center = (N - 1) / 2.0
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
    
    # Polynomial shapes
    for c, d, e in [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 0.5, 1), (2, 1, 0), (1, 3, 0)]:
        x = (i - center) / (N / 2)
        val = c + d * x**2 + e * x**4
        inits.append(np.maximum(val, 0.01))
    
    # U-shaped
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)
    
    # W-shape
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # Random smooth
    np.random.seed(42)
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    best_h = None
    best_val = np.inf
    
    # Stage 1: Evaluate all initializations
    for init in inits:
        if time_left() < 10:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Stage 2: Projected gradient descent with smooth max surrogate
    if time_left() > 40:
        current_h = best_h.copy()
        current_val = best_val
        
        # Adaptive beta for smooth max
        beta = 30.0
        eta = 0.01
        patience = 0
        max_patience = 50
        
        for k in range(2000):
            if time_left() < 15:
                break
            
            # Compute smooth max and gradient
            smooth_val = compute_smooth_max(current_h, beta)
            grad = compute_grad_smooth_max(current_h, beta)
            
            # Projected gradient step
            new_h = current_h - eta * grad
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s < 1e-10:
                eta *= 0.5
                continue
            new_h = new_h / s
            
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                patience = 0
                eta = min(eta * 1.02, 0.05)
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                patience += 1
                eta *= 0.95
                if patience > max_patience:
                    # Increase beta for sharper max approximation
                    beta *= 1.5
                    eta = 0.01
                    patience = 0
        
        # Final polish with higher beta
        if time_left() > 10:
            beta = 100.0
            eta = 0.005
            for k in range(500):
                if time_left() < 5:
                    break
                
                smooth_val = compute_smooth_max(current_h, beta)
                grad = compute_grad_smooth_max(current_h, beta)
                
                new_h = current_h - eta * grad
                new_h = np.maximum(new_h, 0)
                s = np.sum(new_h)
                if s < 1e-10:
                    eta *= 0.5
                    continue
                new_h = new_h / s
                
                new_val = compute_autoconv_max(new_h)
                
                if new_val < current_val:
                    current_h = new_h
                    current_val = new_val
                    eta = min(eta * 1.01, 0.02)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
                else:
                    eta *= 0.95
    
    # Stage 3: Local coordinate descent refinement
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_rounds = int(min(10, time_left() // 3))
        
        for round_num in range(max_rounds):
            if time_left() < 5:
                break
            
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break
                
                for delta in [0.05, 0.02, -0.05, -0.02]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] * (1 + delta))
                    s = np.sum(new_h)
                    if s < 1e-10:
                        continue
                    new_h /= s
                    
                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val - 1e-12:
                        current_h = new_h
                        current_val = new_val
                        improved = True
                        break
            
            if not improved:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h.tolist()
