import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140
NFFT = 2048

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        return np.max(c) / (s * s)

    def smooth_max(c, temp):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))

    def smooth_metric(h, temp):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)

    def compute_gradient_log(h, temp):
        """
        Compute gradient of smooth_metric(exp(x), temp) w.r.t x.
        h = exp(x).
        Returns gradient w.r.t x.
        """
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(N)
        
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        
        # Weights for smooth max
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        exp_c = np.exp(c_shifted)
        w = exp_c / np.sum(exp_c)
        
        # grad_smooth_max w.r.t h
        # g[i] = sum_k w[k] (h[i+k] + h[i-k])
        # Use FFT for efficiency
        w_ext = np.zeros(NFFT)
        w_ext[:2*N-1] = w
        h_ext = np.zeros(NFFT)
        h_ext[:N] = h
        
        W_fft = np.fft.fft(w_ext)
        H_fft = np.fft.fft(h_ext)
        
        # a[i] = sum_k w[k] h[i+k]
        a = np.fft.ifft(np.conj(W_fft) * H_fft).real[:N]
        
        # b[i] = sum_k w[k] h[i-k]
        b = np.fft.ifft(W_fft * H_fft).real[:N]
        
        grad_h = a + b
        
        # smooth_max_val
        sm_val = smooth_max(c, temp)
        
        # d(smooth_metric)/dh = grad_h/s^2 - 2*sm_val/s^3
        df_dh = grad_h / (s * s) - 2 * sm_val / (s ** 3)
        
        # d/dx = d/dh * h
        df_dx = df_dh * h
        
        return df_dx

    n = np.arange(N)
    
    # Phase 1: Structured candidates
    candidates = []
    candidates.append(np.ones(N))
    
    # Barker-like sequences mapped to non-negative
    # Barker codes of length 11, 13, 17, 29, 31, 35, 63, 127...
    # We can tile them or use them to modulate a window.
    # Let's try a simple pseudo-random balanced sequence mapped to 0/1 or 0/2.
    # But we want smoothness.
    
    # Generate a few random balanced sequences
    np.random.seed(42)
    for _ in range(5):
        # Random walk or random signs
        signs = np.random.choice([-1, 1], N)
        # Smooth it
        kernel = np.ones(10) / 10
        smooth_signs = np.convolve(signs, kernel, mode='same')
        # Map to non-negative: h = 0.5 + 0.5 * smooth_signs (clipped)
        h_rand = 0.5 + 0.5 * smooth_signs
        h_rand = np.clip(h_rand, 0, None)
        if h_rand.sum() > 0:
            candidates.append(h_rand)

    # Standard windows
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    center = N / 2 - 0.5
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)

    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))

    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))

    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))

    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))

    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) +
                0.277263158 * np.cos(4*np.pi*n/(N-1)) +
                0.083578947 * np.cos(6*np.pi*n/(N-1)) +
                0.006947368 * np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))

    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)

    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)

    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)

    best_val = 1e10
    best_h = np.ones(N)
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 2: Gradient Descent in Log-Space
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)
    
    # Anneal temperature
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for temp in temperatures:
        if elapsed() > DEADLINE - 30:
            break
        
        step_size = 0.1
        x_curr = x0.copy()
        
        max_iter = 100
        for it in range(max_iter):
            if elapsed() > DEADLINE - 30:
                break
            
            try:
                grad = compute_gradient_log(np.exp(x_curr), temp)
            except Exception:
                break
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            grad_normed = grad / grad_norm
            
            x_trial = x_curr - step_size * grad_normed
            x_trial = x_trial - np.max(x_trial)
            
            f_trial = smooth_metric(np.exp(x_trial), temp)
            f_curr = smooth_metric(np.exp(x_curr), temp)
            
            if f_trial < f_curr:
                x_curr = x_trial
                step_size = min(step_size * 1.1, 0.5)
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        
        h_opt = np.exp(x_curr - np.max(x_curr))
        v = fast_metric(h_opt)
        if v < best_val:
            best_val = v
            best_h = h_opt.copy()
            x0 = x_curr - np.max(x_curr)

    # Phase 3: Coordinate Descent
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)

    no_improve = 0
    max_no_improve = 5000

    while elapsed() < DEADLINE - 30 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]

        mean_h = np.mean(current_h)
        local_scale = max(orig, 0.1 * mean_h, 1e-6)
        step_factors = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
        step_sizes = step_factors * local_scale
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])

        best_local_val = current_val
        best_local_v = orig

        for cv in candidates_val:
            if abs(cv - orig) < 1e-12:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        if no_improve % 200 == 0:
            for scale in [0.01, 0.03, 0.08, 0.15]:
                pert = current_h.copy()
                pert += scale * np.random.randn(N) * np.mean(current_h)
                pert = np.clip(pert, 0, None)
                if pert.sum() > 0:
                    pv = fast_metric(pert)
                    if pv < current_val:
                        current_h = pert
                        current_val = pv
                        no_improve = 0
                        if current_val < best_val:
                            best_val = current_val
                            best_h = current_h.copy()
                        break

    # Phase 4: Fine polish
    if elapsed() < DEADLINE - 5:
        for trial in range(3000):
            if elapsed() > DEADLINE - 3:
                break
            step_frac = 0.005 * (0.7 ** (trial // 100))
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h

    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
