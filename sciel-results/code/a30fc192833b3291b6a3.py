import numpy as np
import time

N = 600
DEADLINE = 135  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def compute_conv_max(h):
        """Compute max of linear autoconvolution of h."""
        c = np.convolve(h, h, mode='full')
        return np.max(c)
    
    def metric(h):
        """Normalized metric: max_conv / (sum_h)^2, scale-invariant."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        return compute_conv_max(h) / (s * s)
    
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 2 * N
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    # Chebyshev/cosine parameterization: h[i] = c0 + sum_{k=1}^{K} c_k * cos(2*pi*k*i/N)
    # This gives smooth, low-frequency step functions
    K = 12  # number of cosine harmonics
    
    def cheb_to_h(coeffs):
        """Convert cosine series coefficients to step heights (nonnegative)."""
        i = np.arange(N)
        h = np.full(N, coeffs[0])
        for k in range(1, K+1):
            h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
        return np.maximum(h, 0.0)
    
    def cheb_metric(coeffs):
        """Metric for a cosine-series parameterized function."""
        h = cheb_to_h(coeffs)
        if np.sum(h) < 1e-15:
            return 1e10
        return fast_metric(h)
    
    # Phase 1: Evaluate structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    i = np.arange(N)
    candidates.append(0.5 * (1 + np.cos(np.pi * i / (N - 1))))
    
    # Gaussian
    xg = (i - N/2) / (N * 0.3)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*i/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*i/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*i/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*i/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse plateau
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    best_val = 1e10
    best_h = np.ones(N)
    
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Remez-like iterative equioscillation on cosine coefficients
    # We parameterize h via cosine series and try to make the convolution
    # as flat as possible (equioscillation principle)
    
    # Fit initial cosine coefficients from best candidate
    def fit_cosine(h, K_order):
        """Fit first K_order cosine coefficients to h."""
        i = np.arange(N)
        # Build design matrix
        A = np.zeros((N, K_order + 1))
        A[:, 0] = 1.0
        for k in range(1, K_order + 1):
            A[:, k] = np.cos(2 * np.pi * k * i / N)
        # Least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return coeffs
    
    coeffs = fit_cosine(best_h, K)
    
    # Remez-like iteration: adjust coefficients to equalize peaks
    # The idea: find the lag where convolution is max, then adjust
    # coefficients to reduce that peak while not increasing others
    
    def get_conv_from_coeffs(coeffs):
        h = cheb_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(2*N-1), 0
        nfft = 2 * N
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1], s
    
    for remez_iter in range(30):
        if elapsed() > DEADLINE - 30:
            break
        
        c, s = get_conv_from_coeffs(coeffs)
        if s < 1e-15:
            break
        
        # Find the max convolution value and its location
        max_idx = np.argmax(c)
        max_val = c[max_idx]
        
        # Compute gradient of convolution at max_idx w.r.t. each coefficient
        # c[k] = sum_i h[i]*h[i+k] (with appropriate bounds)
        # h[i] = sum_j coeffs[j]*cos(2*pi*j*i/N)
        # dc[k]/dcoeffs[j] = sum_i (cos(2*pi*j*i/N)*h[i+k] + h[i]*cos(2*pi*j*(i+k)/N))
        
        h = cheb_to_h(coeffs)
        k = max_idx
        if k >= N:
            k = 2*N - 1 - k  # use symmetry
        
        grad = np.zeros(K + 1)
        for j in range(K + 1):
            g = 0.0
            for i in range(N):
                if i + k < N:
                    g += np.cos(2*np.pi*j*i/N) * h[i+k] + h[i] * np.cos(2*np.pi*j*(i+k)/N)
                if i - k >= 0 and k > 0:
                    g += np.cos(2*np.pi*j*i/N) * h[i-k] + h[i] * np.cos(2*np.pi*j*(i-k)/N)
            grad[j] = g / (s * s)
        
        # Simple gradient step to reduce the peak
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
                # Try smaller step
                new_coeffs = coeffs - lr * 0.1 * grad
                new_h = cheb_to_h(new_coeffs)
                if np.sum(new_h) > 1e-15:
                    new_val = fast_metric(new_h)
                    if new_val < best_val:
                        best_val = new_val
                        best_h = new_h.copy()
                        coeffs = new_coeffs
    
    # Phase 3: Direct coordinate descent on the full 600-dim space
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(42)
    
    no_improve = 0
    max_no_improve = 3000
    
    while elapsed() < DEADLINE - 15 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
        step_sizes = step_factors * mean_h
        candidates_val = list(orig + step_sizes) + list(orig - step_sizes) + [0.0, orig]
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-10:
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
        
        # Periodic perturbation
        if no_improve % 200 == 0 and no_improve > 0:
            pert = current_h.copy()
            pert += 0.03 * np.random.randn(N) * np.mean(current_h)
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
    
    # Phase 4: Fine-tune with paired updates
    if elapsed() < DEADLINE - 5:
        step_frac = 0.003
        for _ in range(1000):
            if elapsed() > DEADLINE - 3:
                break
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
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    # Final verification with exact metric
    final_val = metric(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h.tolist()
