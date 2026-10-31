import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def get_autoconv(h):
        """Return the raw autoconvolution array."""
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1]
    
    # Log-sum-exp smooth approximation of max
    def smooth_max(c, temp=0.01):
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))
    
    def smooth_metric(h, temp=0.01):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)
    
    # Cosine series parameterization
    K = 16
    i_arr = np.arange(N)
    
    def cheb_to_h(coeffs):
        h = np.full(N, coeffs[0])
        for k in range(1, len(coeffs)):
            h += coeffs[k] * np.cos(2 * np.pi * k * i_arr / N)
        return np.maximum(h, 0.0)
    
    def fit_cosine(h, K_order):
        A = np.zeros((N, K_order + 1))
        A[:, 0] = 1.0
        for k in range(1, K_order + 1):
            A[:, k] = np.cos(2 * np.pi * k * i_arr / N)
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return coeffs
    
    # Phase 1: Evaluate structured candidates
    candidates = []
    center = N / 2 - 0.5
    
    candidates.append(np.ones(N))
    
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    # Sinc-like
    for width in [0.2, 0.3, 0.4, 0.5]:
        sinc_h = np.sinc((i_arr - center) / (N * width))
        sinc_h = np.clip(sinc_h, 0, None)
        if sinc_h.sum() > 0:
            candidates.append(sinc_h)
    
    # Gaussian
    for sigma in [0.2, 0.3, 0.4, 0.5]:
        xg = (i_arr - center) / (N * sigma)
        candidates.append(np.exp(-0.5 * xg**2))
    
    # Standard windows
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i_arr / (N - 1)) + 0.08 * np.cos(4 * np.pi * i_arr / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i_arr / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i_arr / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*i_arr/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*i_arr/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*i_arr/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*i_arr/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Plateau/binary-inspired candidates (contiguous block insight)
    # Two-plateau: high at edges, low in center
    for mid_val in [0.0, 0.2, 0.3, 0.5]:
        two_plat = np.ones(N)
        two_plat[N//4:3*N//4] = mid_val
        candidates.append(two_plat)
    
    # Inverse: low at edges, high in center
    for edge_val in [0.0, 0.2, 0.3]:
        inv_plat = np.ones(N)
        inv_plat[:N//4] = edge_val
        inv_plat[3*N//4:] = edge_val
        candidates.append(inv_plat)
    
    # Triangular
    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)
    
    # Asymmetric candidates
    for skew in [0.2, 0.4, -0.2, -0.4]:
        xg = (i_arr - center - skew * N * 0.1) / (N * 0.3)
        candidates.append(np.exp(-0.5 * xg**2))
    
    for shift in [0.1, -0.1, 0.2, -0.2]:
        x_shifted = np.linspace(0, np.pi * (1 + shift), N)
        candidates.append(0.5 * (1 + np.cos(x_shifted)))
    
    # Step functions
    for bp in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        step_h = np.ones(N)
        step_h[int(N*bp):] = 0.2
        candidates.append(step_h)
        step_h2 = np.ones(N)
        step_h2[:int(N*bp)] = 0.2
        candidates.append(step_h2)
    
    # Binary-like: alternating blocks
    for block_len in [10, 20, 30, 50]:
        binary = np.zeros(N)
        for start in range(0, N, block_len * 2):
            binary[start:start+block_len] = 1.0
        candidates.append(binary)
    
    # Evaluate all candidates
    best_val = 1e10
    best_h = np.ones(N)
    
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Remez-like equioscillation on cosine coefficients
    coeffs = fit_cosine(best_h, K)
    
    def get_conv_from_coeffs(coeffs):
        h = cheb_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(2*N-1), 0
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1], s
    
    for remez_iter in range(50):
        if elapsed() > DEADLINE - 40:
            break
        
        c, s = get_conv_from_coeffs(coeffs)
        if s < 1e-15:
            break
        
        max_idx = np.argmax(c)
        max_val = c[max_idx]
        h = cheb_to_h(coeffs)
        k = max_idx
        if k >= N:
            k = 2*N - 1 - k
        
        grad = np.zeros(K + 1)
        for j in range(K + 1):
            g = 0.0
            for i in range(N):
                if i + k < N:
                    g += np.cos(2*np.pi*j*i_arr/N) * h[i+k] + h[i] * np.cos(2*np.pi*j*(i_arr[i]+k)/N)
                if i - k >= 0 and k > 0:
                    g += np.cos(2*np.pi*j*i_arr/N) * h[i-k] + h[i] * np.cos(2*np.pi*j*(i_arr[i]-k)/N)
            grad[j] = g / (s * s)
        
        lr = 0.01 * max_val / (np.max(np.abs(grad)) + 1e-15)
        lr = min(lr, 0.1)
        
        new_coeffs = coeffs - lr * grad
        new_h = cheb_to_h(new_coeffs)
        
        if np.sum(new_h) > 1e-15:
            new_val = fast_metric(new_h)
            if new_val < best_val:
                best_val = new_val
                best_h = new_h.copy()
                coeffs = new_coeffs
            else:
                new_coeffs = coeffs - lr * 0.1 * grad
                new_h = cheb_to_h(new_coeffs)
                if np.sum(new_h) > 1e-15:
                    new_val = fast_metric(new_h)
                    if new_val < best_val:
                        best_val = new_val
                        best_h = new_h.copy()
                        coeffs = new_coeffs
    
    # Phase 3: Gradient-based optimization with smooth max
    def objective(x, temp=0.01):
        if elapsed() > DEADLINE - 20:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=temp)
    
    np.random.seed(42)
    x0_best = np.log(best_h + 1e-15)
    x0_best = x0_best - np.max(x0_best)
    start_points = [x0_best]
    
    for _ in range(4):
        x_rand = x0_best.copy()
        pert = np.random.randn(N) * 0.1
        pert += np.linspace(-0.2, 0.2, N) * np.random.uniform(0.5, 1.5)
        x_rand += pert
        x_rand = x_rand - np.max(x_rand)
        start_points.append(x_rand)
    
    for x0 in start_points:
        if elapsed() > DEADLINE - 25:
            break
        
        for temp in [0.02, 0.01, 0.005]:
            if elapsed() > DEADLINE - 15:
                break
            try:
                result = minimize(
                    lambda x: objective(x, temp), x0, method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 250}
                )
                if result.success or result.fun < 1e9:
                    h_opt = np.exp(result.x - np.max(result.x))
                    v = fast_metric(h_opt)
                    if v < best_val:
                        best_val = v
                        best_h = h_opt.copy()
                        x0 = result.x - np.max(result.x)
            except Exception:
                pass
    
    # Phase 4: Coordinate descent
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    no_improve = 0
    max_no_improve = 4000
    
    while elapsed() < DEADLINE - 15 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0, orig]])
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
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
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
    
    # Phase 5: Fine-tune with small paired moves
    if elapsed() < DEADLINE - 5:
        step_frac = 0.005
        for _ in range(800):
            if elapsed() > DEADLINE - 3:
                break
            i_idx, j_idx = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i_idx] += delta
            new_h[j_idx] -= delta
            if new_h[i_idx] >= 0 and new_h[j_idx] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
    
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
