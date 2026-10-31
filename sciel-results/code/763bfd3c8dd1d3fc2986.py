"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    i = np.arange(N)
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution."""
        h = np.asarray(h, dtype=np.float64)
        if np.all(h == 0):
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = acorr[0]
        if peak == 0:
            return float('inf')
        # Use peak sidelobe ratio: max of off-peak autocorrelation / peak
        sidelobe_max = np.max(np.abs(acorr[1:]))
        return sidelobe_max / peak
    
    def optimize_from(h_init, max_iters=3000):
        """Optimize using projected gradient descent with adaptive step size."""
        h = h_init.copy()
        best_h = h.copy()
        best_m = compute_metric(h)
        lr = 0.05
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            size = 2 * N
            H = np.fft.rfft(h, n=size)
            acorr = np.fft.irfft(H * np.conj(H), n=size)
            peak = acorr[0]
            if peak == 0:
                break
            
            # Find max sidelobe lag
            sidelobe = np.abs(acorr[1:])
            max_lag = np.argmax(sidelobe) + 1
            
            # Gradient of acorr[k] w.r.t. h[j]
            # acorr[k] = sum_{j} h[j]*h[j+k] + h[j+k]*h[j] for the symmetric part
            # Actually acorr[k] = sum_j h[j] h[j+k] for k>=0 (one-sided)
            # d(acorr[k])/d(h[j]) = h[j+k] + h[j-k] (with bounds)
            
            k = max_lag
            grad_acorr_k = np.zeros(N)
            for j in range(N):
                if j + k < N:
                    grad_acorr_k[j] += h[j + k]
                if j - k >= 0:
                    grad_acorr_k[j] += h[j - k]
            
            # d(acorr[0])/d(h[j]) = 2*h[j]
            grad_peak = 2.0 * h
            
            acorr_k = acorr[k]
            # metric = acorr_k / peak
            # d(metric)/d(h[j]) = (grad_acorr_k[j] * peak - acorr_k * grad_peak[j]) / peak^2
            grad_metric = (grad_acorr_k * peak - acorr_k * grad_peak) / (peak * peak)
            
            # Gradient step
            h_new = h - lr * grad_metric
            h_new = np.maximum(h_new, 0.0)
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            # Adaptive learning rate
            if m_new < best_m * 0.999:
                lr = min(lr * 1.3, 0.5)
            else:
                lr = max(lr * 0.95, 0.001)
            
            h = h_new
        
        return best_h, best_m
    
    # Try multiple initializations
    initializations = [
        0.5 * (1.0 + np.cos(np.pi * (i - N / 2.0) / (N / 2.0))),  # raised cosine
        np.cos(np.pi * (i - N / 2.0) / N),  # pure cosine
        1.0 - 0.3 * np.cos(2 * np.pi * i / N),  # flatter
        np.ones(N),  # uniform
        0.5 * (1.0 + np.cos(np.pi * (i - N / 2.0) / (N / 3.0))),  # broader cosine
        np.exp(-2.0 * ((i - N/2.0) / (N/4.0))**2),  # Gaussian
    ]
    
    best_h = None
    best_m = float('inf')
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        init = np.maximum(init, 0.0)
        h_opt, m_opt = optimize_from(init, max_iters=2000)
        
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    return list(best_h)

# EVOLVE-BLOCK-END
