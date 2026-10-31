import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 2048  # power of 2 >= 2*N-1 = 1199
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def smooth_max(c, temp=0.01):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))
    
    def smooth_metric(h, temp=0.01):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 2048
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)
    
    # Phase 1: Try structured candidates
    candidates = []
    n = np.arange(N)
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    # Sinc-like
    center = N / 2 - 0.5
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)
    
    # Gaussian
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning window
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top window
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*n/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*n/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau: high at edges, low in center
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse: low at edges, high in center
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    # Triangular (Bartlett)
    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)
    
    # Evaluate candidates
    best_val = 1e10
    best_h = np.ones(N)
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Gradient descent with simplex projection on probability distribution
    # Normalize to probability distribution: sum(h) = 1
    # Metric becomes: max(c) where c is autoconvolution of normalized h
    # Use smooth max as differentiable surrogate
    
    def project_simplex(v, s=1.0):
        """Euclidean projection onto the simplex {x >= 0, sum(x) = s}."""
        v = np.asarray(v)
        if np.sum(v) < 1e-15:
            return np.full(N, s / N)
        
        # Sort v in descending order
        u = np.sort(v)[::-1]
        
        # Cumulative sum
        cssv = np.cumsum(u)
        
        # Find rho
        # Condition: u_i > (cssv[rho-1] - s) / rho
        rhos = np.arange(1, N + 1)
        taus = (cssv - s) / rhos
        cond = u > taus
        
        true_indices = np.where(cond)[0]
        if len(true_indices) == 0:
            rho = N
        else:
            rho = true_indices[-1] + 1
            
        tau = (cssv[rho-1] - s) / rho
        h = np.maximum(v - tau, 0)
        
        # Numerical cleanup
        h = np.clip(h, 0, None)
        total = np.sum(h)
        if total > 0:
            h = h * (s / total)
        else:
            h = np.full(N, s / N)
            
        return h
    
    def compute_gradient(h, temp=0.01):
        """Compute gradient of smooth_metric w.r.t. h (unnormalized)."""
        nfft = 2048
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1].copy()
        
        # Smooth max gradient: softmax weights
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        weights = np.exp(c_shifted) / np.sum(np.exp(c_shifted))
        
        # Gradient of smooth_max(c) w.r.t. c is weights
        # c[k] = sum_i h[i] * h[i+k] (with appropriate indexing for linear convolution)
        # dc/dh[j] = sum_k weights[k] * dc[k]/dh[j]
        
        # For linear convolution c[k] = sum_i h[i]*h[i+k] where 0<=i<N, 0<=i+k<N
        # dc[k]/dh[j] = h[j-k] + h[j+k] (when indices valid)
        
        # Use FFT for efficient gradient computation
        # c = ifft(fft(h)^2), so dc/dh = ifft(2*fft(h)*conj(fft(h)))... 
        # Actually, let's use the chain rule properly.
        
        # Let's compute gradient numerically-friendly way using FFT
        # c[k] = (h * h)[k] (linear convolution)
        # dc/dh[j] = sum_k weights[k] * (dc[k]/dh[j])
        # dc[k]/dh[j] = h[j-k] + h[j+k] if indices valid
        
        # Efficient computation:
        # grad[j] = sum_k weights[k] * (h[j-k] + h[j+k])
        # This is like a weighted correlation
        
        # Use FFT: create extended h and compute
        h_ext = np.zeros(2*N-1)
        h_ext[:N] = h
        
        # For each k, dc[k]/dh[j] = h[j-k] + h[j+k] (with bounds)
        # This is equivalent to: grad = convolve(weights, h) + convolve(weights_flipped, h)
        
        # Let's do it directly with FFT
        # weights is length 2N-1
        # grad[j] = sum_k weights[k] * (h[j-k] + h[j+k])
        
        # First term: sum_k weights[k] * h[j-k] = (weights * h)[j] (cross-correlation)
        # Second term: sum_k weights[k] * h[j+k] = (weights_flipped * h)[j]
        
        # Use FFT for cross-correlation
        w_ext = np.zeros(2*N-1)
        w_ext[:len(weights)] = weights
        
        # Cross-correlation: sum_k w[k] * h[j-k]
        # This is ifft(conj(fft(w)) * fft(h))
        nfft2 = 4096
        W = np.fft.rfft(w_ext, n=nfft2)
        Hh = np.fft.rfft(h_ext, n=nfft2)
        corr1 = np.fft.irfft(np.conj(W) * Hh, n=nfft2)
        
        # Second term: sum_k w[k] * h[j+k]
        # This is ifft(fft(w) * conj(fft(h)))... let me think
        # Actually: sum_k w[k] * h[j+k] = sum_k w[k] * h[k+j]
        # Let m = k+j, then k = m-j: sum_m w[m-j] * h[m]
        # This is convolution of w_flipped with h
        
        w_flip = w_ext[::-1]
        Wf = np.fft.rfft(w_flip, n=nfft2)
        corr2 = np.fft.irfft(Wf * Hh, n=nfft2)
        
        grad_raw = corr1[:N] + corr2[:N]
        
        # Now account for the 1/(s^2) factor and the fact that we're differentiating
        # smooth_metric = smooth_max(c) / s^2
        # d/dh [smooth_max(c) / s^2] = (1/s^2) * d(smooth_max)/dh - 2*smooth_max/s^3 * s * (1/s) * 1
        # = (1/s^2) * grad_raw - 2*smooth_max/(s^3) * (1/N) * N
        # Wait, let me be more careful.
        
        s = np.sum(h)
        sm = smooth_max(c, temp)
        
        # d/dh_j [smooth_max(c)/s^2]
        # = (1/s^2) * d(smooth_max)/dh_j - 2*smooth_max/s^3 * ds/dh_j
        # ds/dh_j = 1
        # d(smooth_max)/dh_j = grad_raw[j]
        
        grad = grad_raw / (s * s) - 2 * sm / (s * s * s)
        
        return grad
    
    # Start from best_h normalized to sum=1
    h_prob = best_h.copy()
    h_prob = h_prob / np.sum(h_prob)
    
    # Gradient descent with simplex projection and line search
    temp_schedule = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temp_schedule:
        if elapsed() > DEADLINE - 20:
            break
        
        h_curr = h_prob.copy()
        best_local = fast_metric(h_curr)
        best_local_h = h_curr.copy()
        
        max_iter = 200
        for it in range(max_iter):
            if elapsed() > DEADLINE - 10:
                break
            
            grad = compute_gradient(h_curr, temp)
            
            # Line Search
            lr = 0.1
            improved = False
            for _ in range(20):
                h_new = h_curr - lr * grad
                h_new = project_simplex(h_new, 1.0)
                
                v_new = fast_metric(h_new)
                
                if v_new < best_local:
                    best_local = v_new
                    best_local_h = h_new.copy()
                    h_curr = h_new.copy()
                    improved = True
                    break
                else:
                    lr *= 0.5
            
            if not improved:
                break
                
        h_prob = best_local_h
        h_eval = h_prob
        v = fast_metric(h_eval)
        if v < best_val:
            best_val = v
            best_h = h_eval.copy()
    
    # Phase 3: Fine-tune with coordinate descent
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    no_improve = 0
    max_no_improve = 3000
    
    while elapsed() < DEADLINE - 25 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
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
        
        # Periodic perturbation
        if no_improve % 100 == 0:
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
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
