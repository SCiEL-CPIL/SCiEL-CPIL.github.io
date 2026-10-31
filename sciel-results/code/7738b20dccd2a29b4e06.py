import numpy as np
import time
import random

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
    
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    # Parametric families for initialization
    def make_bimodal(sigma, alpha):
        """Bimodal: two Gaussians at positions alpha*N and (1-alpha)*N"""
        pos1 = alpha * N
        pos2 = (1 - alpha) * N
        g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
        g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
        return g1 + g2
    
    def make_poly(c, d, e):
        """Polynomial: c + d*(x-center)^2 + e*(x-center)^4, clipped to nonneg"""
        x = (i - center) / (N / 2)
        val = c + d * x**2 + e * x**4
        return np.maximum(val, 0.01)
    
    # Generate diverse initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal with various parameters
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            inits.append(make_bimodal(sigma, alpha))
    
    # Polynomial shapes
    for c, d, e in [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 0.5, 1), (2, 1, 0), (1, 3, 0)]:
        inits.append(make_poly(c, d, e))
    
    # U-shaped
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)
    
    # W-shape (two bumps)
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # Random smooth
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
    
    # Stage 2: Cosine series parameterization with Remez-like equioscillation
    if time_left() > 30:
        M = 12
        H_freq = np.fft.rfft(best_h)
        coeffs = np.zeros(M + 1)
        coeffs[0] = H_freq[0].real / N
        for m in range(1, M + 1):
            if m < len(H_freq):
                coeffs[m] = 2 * H_freq[m].real / N
        
        def coeffs_to_h(c):
            """Convert cosine coefficients to heights."""
            h = np.zeros(N)
            for m in range(len(c)):
                h += c[m] * np.cos(2 * np.pi * m * i / N)
            return h
        
        def compute_conv_weights(h, beta=30.0):
            """Compute softmax weights on autoconvolution for smooth max."""
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            m = np.max(conv)
            exp_conv = np.exp((conv - m) * beta)
            weights = exp_conv / np.sum(exp_conv)
            return weights, m, H
        
        def compute_grad_coeff(c, beta=30.0):
            """Compute gradient of smooth max autoconvolution w.r.t. coefficients."""
            h = coeffs_to_h(c)
            h = np.maximum(h, 1e-10)
            s = np.sum(h)
            h_norm = h / s
            
            weights, m_val, H = compute_conv_weights(h_norm, beta)
            
            size = 2 * N
            w_full = np.zeros(size)
            w_full[:len(weights)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
            
            grad_c = np.zeros(len(c))
            for m in range(len(c)):
                grad_c[m] = np.dot(grad_h, np.cos(2 * np.pi * m * i / N))
            
            return grad_c, m_val + np.log(np.sum(weights)) / beta
        
        def remez_step(c, beta=30.0):
            """Perform one Remez-like step: find peaks and adjust to equioscillate."""
            h = coeffs_to_h(c)
            h = np.maximum(h, 1e-10)
            s = np.sum(h)
            h_norm = h / s
            
            weights, m_val, H = compute_conv_weights(h_norm, beta)
            
            K = min(8, 2 * N)
            top_indices = np.argsort(weights)[-K:]
            
            peak_mask = np.zeros(2 * N)
            peak_mask[top_indices] = weights[top_indices]
            peak_mask = peak_mask / np.sum(peak_mask)
            
            w_full = np.zeros(2 * N)
            w_full[:len(peak_mask)] = peak_mask
            W = np.fft.fft(w_full, n=2 * N)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
            
            grad_c = np.zeros(len(c))
            for m in range(len(c)):
                grad_c[m] = np.dot(grad_h, np.cos(2 * np.pi * m * i / N))
            
            return grad_c, m_val
        
        current_coeffs = coeffs.copy()
        current_val = best_val
        
        # Phase 1: Gradient descent
        eta = 0.05
        for k in range(150):
            if time_left() < 15:
                break
            
            h_curr = coeffs_to_h(current_coeffs)
            h_curr = np.maximum(h_curr, 1e-10)
            h_curr = h_curr / np.sum(h_curr)
            curr_val = compute_autoconv_max(h_curr)
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()
            
            grad_c, smooth_val = compute_grad_coeff(current_coeffs)
            
            grad_norm = np.linalg.norm(grad_c)
            if grad_norm > 1e-10:
                grad_c = grad_c / grad_norm
            
            new_coeffs = current_coeffs - eta * grad_c
            
            new_h = coeffs_to_h(new_coeffs)
            if np.any(new_h < -1e-6):
                eta *= 0.5
                if eta < 1e-8:
                    break
                continue
            
            new_h = np.maximum(new_h, 0)
            new_h = new_h / np.sum(new_h)
            new_val = compute_autoconv_max(new_h)
            
            if new_val < curr_val:
                current_coeffs = new_coeffs
                eta = min(eta * 1.05, 0.3)
            else:
                eta *= 0.8
        
        # Phase 2: Remez-like refinement
        eta = 0.02
        for k in range(100):
            if time_left() < 10:
                break
            
            h_curr = coeffs_to_h(current_coeffs)
            h_curr = np.maximum(h_curr, 1e-10)
            h_curr = h_curr / np.sum(h_curr)
            curr_val = compute_autoconv_max(h_curr)
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()
            
            grad_c, m_val = remez_step(current_coeffs)
            
            grad_norm = np.linalg.norm(grad_c)
            if grad_norm > 1e-10:
                grad_c = grad_c / grad_norm
            
            new_coeffs = current_coeffs - eta * grad_c
            
            new_h = coeffs_to_h(new_coeffs)
            if np.any(new_h < -1e-6):
                eta *= 0.5
                if eta < 1e-9:
                    break
                continue
            
            new_h = np.maximum(new_h, 0)
            new_h = new_h / np.sum(new_h)
            new_val = compute_autoconv_max(new_h)
            
            if new_val < curr_val:
                current_coeffs = new_coeffs
                eta = min(eta * 1.03, 0.15)
            else:
                eta *= 0.85
        
        final_h = coeffs_to_h(current_coeffs)
        final_h = np.maximum(final_h, 0)
        if np.sum(final_h) > 0:
            final_h = final_h / np.sum(final_h)
            final_val = compute_autoconv_max(final_h)
            if final_val < best_val:
                best_val = final_val
                best_h = final_h.copy()
    
    # Stage 3: Direct coordinate-space optimization with smooth max
    # This is the key structural change: optimize directly in h-space using
    # gradient of the smooth max, avoiding the cosine parameterization bottleneck
    if time_left() > 25:
        current_h = best_h.copy()
        current_val = best_val
        
        def compute_grad_h_direct(h, beta=50.0):
            """Compute gradient of smooth max autoconvolution w.r.t. h directly."""
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            m = np.max(conv)
            exp_conv = np.exp((conv - m) * beta)
            weights = exp_conv / np.sum(exp_conv)
            
            w_full = np.zeros(size)
            w_full[:len(weights)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
            
            return grad_h, m + np.log(np.sum(exp_conv)) / beta
        
        # Gradient descent in h-space with line search
        eta = 0.01
        for k in range(200):
            if time_left() < 10:
                break
            
            curr_val = compute_autoconv_max(current_h)
            if curr_val < best_val:
                best_val = curr_val
                best_h = current_h.copy()
            
            grad_h, smooth_val = compute_grad_h_direct(current_h)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad_h)
            if grad_norm < 1e-12:
                break
            grad_h = grad_h / grad_norm
            
            # Try several step sizes
            best_step = None
            best_new_val = curr_val
            for factor in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                step = eta * factor
                new_h = current_h - step * grad_h
                new_h = np.maximum(new_h, 0)
                s = np.sum(new_h)
                if s < 1e-10:
                    continue
                new_h = new_h / s
                new_val = compute_autoconv_max(new_h)
                if new_val < best_new_val:
                    best_new_val = new_val
                    best_step = new_h
            
            if best_step is not None and best_new_val < curr_val:
                current_h = best_step
                eta = min(eta * 1.1, 0.1)
            else:
                eta *= 0.7
        
        if compute_autoconv_max(current_h) < best_val:
            best_val = compute_autoconv_max(current_h)
            best_h = current_h.copy()
    
    # Stage 4: Simulated Annealing refinement
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        best_h_sa = best_h.copy()
        best_val_sa = best_val
        
        T = 0.005
        T_min = 1e-7
        cooling = 0.997
        n_iter = int(min(3000, time_left() * 40))
        
        for k in range(n_iter):
            if time_left() < 5:
                break
            
            n_perturb = random.randint(1, 3)
            indices = random.sample(range(N), n_perturb)
            
            new_h = current_h.copy()
            for idx in indices:
                noise = random.gauss(0, T * 0.05)
                new_h[idx] = max(0, current_h[idx] + noise)
            
            s = np.sum(new_h)
            if s < 1e-10:
                continue
            new_h /= s
            
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val or random.random() < np.exp(-(new_val - current_val) / T):
                current_h = new_h
                current_val = new_val
                
                if current_val < best_val_sa:
                    best_val_sa = current_val
                    best_h_sa = current_h.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_val_sa < best_val:
            best_val = best_val_sa
            best_h = best_h_sa
    
    # Stage 5: Fine coordinate descent
    if time_left() > 8:
        current_h = best_h.copy()
        current_val = best_val
        
        for round_num in range(3):
            if time_left() < 3:
                break
            
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 2:
                    break
                
                for delta in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] * (1 + delta))
                    s = np.sum(new_h)
                    if s < 1e-10:
                        continue
                    new_h /= s
                    
                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val - 1e-13:
                        current_h = new_h
                        current_val = new_val
                        break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h.tolist()
