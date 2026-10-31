import numpy as np
import time
import random

DEADLINE = 115.0

def construct():
    start_time = time.time()
    N = 600
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def conv(h):
        """Linear autoconvolution of h (normalized so sum(h)=1)."""
        c = np.fft.fftconvolve(h, h, mode='full')
        return c
    
    def obj(h):
        """Objective: max of linear autoconvolution."""
        h = np.asarray(h)
        if h.size != N or np.any(h < -1e-10):
            return 1e8
        s = h.sum()
        if s <= 1e-12:
            return 1e8
        h = h / s
        return float(np.max(conv(h)))
    
    def smax(c, beta):
        """Smooth max of c via log-sum-exp."""
        m = np.max(c)
        return m + np.log(np.sum(np.exp(beta * (c - m)))) / beta
    
    def grad_conv(c, h, beta):
        """Gradient of soft-max autoconvolution w.r.t. h (length N)."""
        # c[i] = sum_{j=0}^{N-1} h[j] * h[i-j] for i=0..2N-2 (valid autoconv)
        # d c[i]/d h[k] = h[i-k] + (if i-k >= 0 and i-k <= N-1) h[i-k]
        # Actually for linear autocorr c[i] = sum_j h[j] h[i-j] (full), d c[i]/d h[k] = h[i-k] + h[k-i]
        w = np.exp(beta * (c - np.max(c)))
        w /= np.sum(w)
        # w has length 2N-1
        # grad_h[k] = sum_i w[i] * (h[i-k] + h[k-i])
        # Use convolution: grad_h = 2 * (w conv h) evaluated appropriately
        # Let's do it via convolution: grad_h[k] = sum_i w[i]*(h[i-k]+h[k-i])
        # For i-k in [0,N-1]: h[i-k]; for k-i in [0,N-1]: h[k-i]
        # This is 2 * sum_{j} w[k+j] * h[j] where j = i-k, but with range limits
        # More simply: grad_h = 2 * ifft(fft(w_full) * fft(h_full)).real[:N] where padding is handled
        # Since c is full convolution of length 2N-1, and we're computing sum_i w[i]*(h[i-k]+h[k-i]),
        # this equals 2 * (w circ_conv h)[k] if lengths match, but here it's linear.
        # Let's just do it directly for clarity and correctness with a small optimization.
        # For N=600, O(N^2) is 360k ops, fine.
        # But let's use FFT: pad w and h to length L >= 2N-1
        L = 2 * N
        w_pad = np.zeros(L)
        w_pad[:len(w)] = w
        h_pad = np.zeros(L)
        h_pad[:N] = h
        grad_h = 2.0 * np.real(np.fft.ifft(np.fft.fft(w_pad) * np.fft.fft(h_pad)))[:N]
        return grad_h
    
    i = np.arange(N, dtype=float)
    
    # Parametric initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
    
    # U-shaped / W-shaped
    center = (N - 1) / 2.0
    x = (i - center) / (N / 2)
    inits.append(1.0 + 2.0 * x**2)
    inits.append(np.maximum(1.0 + 1.5 * np.cos(4 * np.pi * i / N), 0.01))
    inits.append(np.maximum(1.0 + 3.0 * x**4, 0.01))
    
    # Cosine series random initializations
    for _ in range(10):
        h = np.ones(N)
        for m in range(1, 15):
            h += random.uniform(-0.5, 0.5) * np.cos(2 * np.pi * m * i / N)
        inits.append(np.maximum(h, 0.01))
    
    # Find best initial point
    best_h = None
    best_val = np.inf
    for init in inits:
        if time_left() < 5:
            break
        h = init / np.sum(init)
        val = obj(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = obj(best_h)
    
    # Cosine-series Remez-like optimization
    if time_left() > 30:
        M = 20  # Number of cosine harmonics
        
        def coeffs_to_h(c):
            h = np.zeros(N)
            for m in range(min(len(c), M + 1)):
                h += c[m] * np.cos(2 * np.pi * m * i / N)
            return h
        
        def h_to_coeffs(h):
            c = np.zeros(M + 1)
            H_f = np.fft.rfft(h)
            c[0] = H_f[0].real / N
            for m in range(1, min(M + 1, len(H_f))):
                c[m] = 2.0 * H_f[m].real / N
            return c
        
        # Initial coefficients from best_h
        c_curr = h_to_coeffs(best_h)
        
        # Optimization loop with adaptive beta for smooth max
        beta = 20.0
        eta = 0.1
        
        for iteration in range(200):
            if time_left() < 10:
                break
            
            h_curr = coeffs_to_h(c_curr)
            h_curr = np.maximum(h_curr, 0.0)
            s = h_curr.sum()
            if s < 1e-12:
                break
            h_curr = h_curr / s
            
            val_curr = obj(h_curr)
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
            
            # Compute smooth max objective and gradient
            c_conv = conv(h_curr)
            soft_val = smax(c_conv, beta)
            
            # Gradient w.r.t. h
            grad_h = grad_conv(c_conv, h_curr, beta)
            
            # Project gradient onto cosine coefficient space
            # d soft_val / d c[m] = sum_k grad_h[k] * d h[k]/d c[m] = sum_k grad_h[k] * cos(2*pi*m*k/N)
            grad_c = np.zeros(M + 1)
            for m in range(M + 1):
                grad_c[m] = np.dot(grad_h, np.cos(2 * np.pi * m * i / N))
            
            # Normalize
            gn = np.linalg.norm(grad_c)
            if gn < 1e-12:
                break
            grad_c = grad_c / gn
            
            # Remez-like equioscillation: find active peaks and bias the gradient
            c_curr_conv = conv(coeffs_to_h(c_curr) / np.sum(coeffs_to_h(c_curr)))
            # Find top K peaks
            K = min(16, len(c_curr_conv))
            top_idx = np.argsort(c_curr_conv)[-K:]
            
            # Compute a peak-biased gradient
            # Weight the gradient contribution by proximity to peaks
            peak_w = np.zeros_like(c_curr_conv)
            peak_w[top_idx] = 1.0
            # Smooth peak weights
            from scipy.ndimage import gaussian_filter
            peak_w_smooth = gaussian_filter(peak_w, sigma=3)
            
            # Recompute gradient with peaked weights
            w_peak = np.exp(beta * (c_curr_conv - np.max(c_curr_conv)))
            w_peak = w_peak * (0.5 + peak_w_smooth)
            w_peak /= np.sum(w_peak)
            
            L = 2 * N
            w_pad = np.zeros(L)
            w_pad[:len(w_peak)] = w_peak
            h_pad = np.zeros(L)
            h_pad[:N] = h_curr
            grad_h_peak = 2.0 * np.real(np.fft.ifft(np.fft.fft(w_pad) * np.fft.fft(h_pad)))[:N]
            
            grad_c_peak = np.zeros(M + 1)
            for m in range(M + 1):
                grad_c_peak[m] = np.dot(grad_h_peak, np.cos(2 * np.pi * m * i / N))
            
            # Blend gradients
            mix = 0.7
            grad_combined = mix * (grad_c_peak / (np.linalg.norm(grad_c_peak) + 1e-12)) + (1 - mix) * grad_c
            
            new_c = c_curr - eta * grad_combined
            new_h = coeffs_to_h(new_c)
            
            # Ensure nonnegativity with projection
            if np.min(new_h) < 0:
                new_h = np.maximum(new_h, 0.0)
                new_c = h_to_coeffs(new_h)
            
            s_new = new_h.sum()
            if s_new < 1e-12:
                eta *= 0.5
                continue
            new_h = new_h / s_new
            new_val = obj(new_h)
            
            if new_val < val_curr:
                c_curr = h_to_coeffs(new_h)
                eta = min(eta * 1.03, 0.5)
                beta = min(beta * 1.02, 100.0)  # Sharpen smooth max gradually
            else:
                eta *= 0.7
                # If stuck, try a perturbation
                if random.random() < 0.3:
                    c_curr += random.gauss(0, 0.01 * eta, size=M + 1)
        
        # Update best
        final_h = coeffs_to_h(c_curr)
        final_h = np.maximum(final_h, 0)
        s = final_h.sum()
        if s > 1e-12:
            final_h = final_h / s
            val = obj(final_h)
            if val < best_val:
                best_val = val
                best_h = final_h.copy()
    
    # High-precision L-BFGS-B polish
    if time_left() > 15:
        from scipy.optimize import minimize
        
        def obj_lbfgs(x):
            h = np.maximum(x, 0)
            s = h.sum()
            if s < 1e-12:
                return 1e8
            h = h / s
            return obj(h)
        
        def grad_lbfgs(x):
            h = np.maximum(x, 0)
            s = h.sum()
            if s < 1e-12:
                return np.zeros(N)
            h = h / s
            beta = 50.0
            c_conv = conv(h)
            w = np.exp(beta * (c_conv - np.max(c_conv)))
            w /= np.sum(w)
            grad_h = grad_conv(c_conv, h, beta)
            # Account for normalization constraint
            # grad_f/g(x) where g = sum h - 1... for simplicity, just return grad_h
            return grad_h
        
        res = minimize(obj_lbfgs, best_h, jac=grad_lbfgs, method='L-BFGS-B',
                       bounds=[(0, None)] * N,
                       options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
        
        if res.fun < best_val:
            h_new = np.maximum(res.x, 0)
            s = h_new.sum()
            if s > 1e-12:
                h_new = h_new / s
                val = obj(h_new)
                if val < best_val:
                    best_val = val
                    best_h = h_new.copy()
    
    # Final coordinate descent polish
    if time_left() > 8:
        h_cd = best_h.copy()
        val_cd = best_val
        for rnd in range(5):
            if time_left() < 3:
                break
            improved = False
            for _ in range(N):
                if time_left() < 2:
                    break
                k = random.randint(0, N - 1)
                for delta in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                    new_h = h_cd.copy()
                    new_h[k] = max(0, new_h[k] + delta * new_h[k])
                    s = new_h.sum()
                    if s < 1e-12:
                        continue
                    new_h = new_h / s
                    v = obj(new_h)
                    if v < val_cd - 1e-14:
                        h_cd = new_h
                        val_cd = v
                        improved = True
                        break
            if not improved:
                break
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd
    
    return best_h.tolist()
