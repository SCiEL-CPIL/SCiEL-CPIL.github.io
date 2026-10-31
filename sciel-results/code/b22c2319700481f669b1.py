import numpy as np
import time
import random
from scipy.optimize import minimize

N = 600
DEADLINE = 145

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    k = np.arange(N)
    
    def compute_autoconv(h):
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    def compute_autoconv_max(h):
        conv = compute_autoconv(h)
        return np.max(conv)
    
    # --- Spectral Equalization Strategy (Sinkhorn-Knopp inspired) ---
    # Goal: Flatten the magnitude spectrum |H(f)| to minimize peak autoconvolution.
    # Autoconvolution is inverse FFT of |H(f)|^2.
    # Minimizing max(ifft(|H|^2)) is related to making |H| as flat as possible (flat-top window).
    # We iterate: 
    # 1. Compute current spectrum H
    # 2. Compute target spectrum magnitude T such that ifft(T^2) is flatter
    # 3. Update H magnitude towards T while keeping phase
    # 4. Inverse FFT to get new h
    # 5. Project h to non-negative and normalize
    # 6. Repeat
    
    def spectral_equalization(h, iterations=200, decay=0.9):
        """
        Apply iterative spectral equalization to h.
        Returns optimized h.
        """
        h = h.copy()
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s == 0:
            return np.ones(N) / N
        h = h / s
        
        best_h = h.copy()
        best_val = compute_autoconv_max(h)
        
        for it in range(iterations):
            if time_left() < 5:
                break
            
            # Compute spectrum
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            
            # Current power spectrum
            P = np.abs(H) ** 2
            
            # Current autoconvolution
            conv = np.fft.ifft(P).real
            max_conv = np.max(conv)
            
            # We want to reduce the peak.
            # A simple heuristic: reduce the magnitude of frequency components that contribute most to the peak.
            # Or, aim for a flat spectrum.
            # Let's try to flatten the spectrum towards a target.
            # Target: constant magnitude?
            # If H is constant, h is a delta function (peak is 1).
            # If h is uniform, H is a sinc-like function (peak is 1/N).
            # Wait, for probability distribution (sum=1):
            # Uniform h: h_k = 1/N. H_0 = 1, others ~0. Conv is triangle peak 1/N.
            # Delta h: h_0 = 1. H is constant 1. Conv is delta peak 1.
            # So we want to spread out the spectrum? No, we want to minimize the peak of the inverse transform of |H|^2.
            
            # Let's use the gradient of the smooth max to guide the spectral update.
            # Or simpler: Sinkhorn-like alternation.
            
            # Let's try a direct approach:
            # Minimize max(conv) by adjusting H.
            # grad_H = 2 * H * conj(fft(weights))
            # where weights = softmax(conv)
            
            # Let's implement a simple gradient step in spectral domain.
            beta = 50.0
            exp_conv = np.exp((conv - max_conv) * beta)
            weights = exp_conv / np.sum(exp_conv)
            
            # Gradient of max(conv) w.r.t H
            # conv = ifft(|H|^2)
            # d conv / d H = H * conj(fft(weights)) ?
            # Let's verify.
            # L = <w, ifft(|H|^2)> = <fft(w), |H|^2>
            # dL/dH = H * conj(fft(w))
            
            W = np.fft.fft(weights, n=size)
            grad_H = H * np.conj(W)
            
            # Update H
            # H_new = H - eta * grad_H
            # But we want to keep it real and non-negative after IFFT.
            # So we update the magnitude and phase separately or just do a step and project.
            
            # Let's try adjusting magnitude towards a flatter profile.
            # Target magnitude: average magnitude?
            avg_mag = np.mean(np.abs(H))
            target_mag = avg_mag * np.ones_like(np.abs(H))
            
            # Blend current magnitude with target
            alpha = 0.1
            new_mag = (1 - alpha) * np.abs(H) + alpha * target_mag
            
            # Keep phase
            phase = np.angle(H)
            H_new = new_mag * np.exp(1j * phase)
            
            # Inverse FFT
            h_new = np.fft.ifft(H_new).real[:N]
            
            # Project to non-negative
            h_new = np.maximum(h_new, 0)
            s_new = np.sum(h_new)
            if s_new < 1e-10:
                break
            h_new = h_new / s_new
            
            val_new = compute_autoconv_max(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
            else:
                # If it got worse, reduce alpha
                alpha *= 0.5
                if alpha < 1e-4:
                    break
            
            h = h_new
        
        return best_h

    # --- Initialization ---
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal
    for sigma in [N/10, N/8, N/6]:
        for alpha in [0.1, 0.2, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((k - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((k - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
    
    # W-shape
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * k / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # U-shape
    center = (N - 1) / 2.0
    u_shape = 1.0 + 2.0 * ((k - center) / (N/2))**2
    inits.append(u_shape)
    
    # Random smooth
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * k / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    best_h = None
    best_val = np.inf
    
    # Stage 1: Evaluate inits
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
    
    # Stage 2: Spectral Equalization on best init
    if time_left() > 20:
        h_opt = spectral_equalization(best_h, iterations=300)
        val_opt = compute_autoconv_max(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Stage 3: Gradient refinement on the result
    # Use the cosine series parameterization from the current program for fine-tuning
    if time_left() > 30:
        M = 12
        
        def h_to_coeffs(h):
            H_freq = np.fft.rfft(h)
            c = np.zeros(M + 1)
            c[0] = H_freq[0].real / N
            for m in range(1, M + 1):
                if m < len(H_freq):
                    c[m] = 2 * H_freq[m].real / N
            return c
        
        def coeffs_to_h(c):
            h = np.zeros(N)
            for m in range(len(c)):
                h += c[m] * np.cos(2 * np.pi * m * k / N)
            return h
        
        def smooth_max_obj_and_grad(coeffs, tau=1.0):
            h = coeffs_to_h(coeffs)
            h_clipped = np.maximum(h, 0)
            s = np.sum(h_clipped)
            if s < 1e-10:
                return 1e10, np.zeros(len(coeffs))
            h_norm = h_clipped / s
            
            c = compute_autoconv(h_norm)
            c_max = np.max(c)
            exp_c = np.exp((c - c_max) / tau)
            sum_exp = np.sum(exp_c)
            obj = tau * (np.log(sum_exp) + c_max / tau)
            
            w = exp_c / sum_exp
            
            L = 4 * N
            w_pad = np.zeros(L)
            w_pad[:len(w)] = w
            h_pad = np.zeros(L)
            h_pad[:N] = h_norm
            
            W = np.fft.fft(w_pad, n=L)
            H = np.fft.fft(h_pad, n=L)
            
            corr_full = np.fft.ifft(W * np.conj(H)).real
            G_h = 2.0 * corr_full[0:N]
            
            grad_c = np.zeros(len(coeffs))
            for m in range(len(coeffs)):
                if m == 0:
                    grad_c[m] = np.sum(G_h)
                else:
                    grad_c[m] = np.dot(G_h, np.cos(2 * np.pi * m * k / N))
                    
            return obj, grad_c
        
        c_init = h_to_coeffs(best_h)
        
        def obj_func(c):
            val, grad = smooth_max_obj_and_grad(c, tau=1.0)
            return val
        
        def grad_func(c):
            val, grad = smooth_max_obj_and_grad(c, tau=1.0)
            return grad
            
        try:
            res = minimize(obj_func, c_init, jac=grad_func, method='L-BFGS-B', 
                           options={'maxiter': 100, 'maxfun': 200, 'maxls': 50})
            
            h_res = coeffs_to_h(res.x)
            h_res = np.maximum(h_res, 0)
            h_res = h_res / np.sum(h_res)
            
            c_res = compute_autoconv(h_res)
            val_res = np.max(c_res)
            
            if val_res < best_val:
                best_val = val_res
                best_h = h_res.copy()
                
        except Exception as e:
            pass

    return best_h.tolist()

construct()
