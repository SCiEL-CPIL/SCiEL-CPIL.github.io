<<<<<<< SEARCH
import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill
=======
import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def spectral_phase_optimize(h, start_time, max_time, n_iters=2000, lr_init=0.05):
    """Optimize phase angles in frequency domain to flatten autoconvolution peak.
    
    Represent h as |FFT^{-1}(A * e^{i*theta})| where A is fixed amplitude spectrum
    and theta are phase angles to optimize. Use gradient descent on the peak of
    the autoconvolution (which equals |FFT(h)|^2).
    """
    best_h = h.copy()
    best_val = compute_score(h)
    
    # Normalize to unit sum
    s = np.sum(h)
    if s < 1e-12:
        return best_h, best_val
    h_norm = h * (N / s)
    
    # Compute FFT
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H_fft = np.fft.rfft(h_norm, n=n_fft)
    
    # Split into amplitude and phase
    amp = np.abs(H_fft)
    phase = np.angle(H_fft)
    
    # Protect against zero amplitudes
    amp_safe = np.maximum(amp, 1e-10)
    
    current_phase = phase.copy()
    current_h = h_norm.copy()
    current_val = best_val
    
    lr = lr_init
    
    for it in range(n_iters):
        if time.time() - start_time > max_time:
            break
        
        # Compute current h from amplitude and phase
        H_current = amp_safe * np.exp(1j * current_phase)
        h_candidate = np.fft.irfft(H_current, n=n_fft)[:N]
        h_candidate = np.maximum(h_candidate, 0.0)
        
        s_cand = np.sum(h_candidate)
        if s_cand < 1e-12:
            break
        h_candidate = h_candidate * (N / s_cand)
        
        # Compute autoconvolution peak
        conv = fft_autoconv(h_candidate)
        val = np.max(conv) / (N * N)
        
        if val < current_val:
            current_val = val
            current_h = h_candidate.copy()
            if val < best_val:
                best_val = val
                best_h = h_candidate.copy()
        
        # Compute gradient of peak w.r.t. phase
        # Peak occurs at some lag m, we want to minimize conv[m]
        peak_idx = np.argmax(conv)
        
        # Gradient: d(conv[m])/d(phase[k]) for k in 0..n_fft//2
        # conv[m] = sum_j h[j]*h[m-j]
        # h = IFFT(amp * exp(i*phase))
        # This is complex; use finite differences or analytical gradient
        
        # Analytical gradient: 
        # h[n] = (1/n_fft) * sum_k amp[k] * exp(i*(phase[k] + 2*pi*k*n/n_fft))
        # d(h[n])/d(phase[k]) = (1/n_fft) * i * amp[k] * exp(i*(phase[k] + 2*pi*k*n/n_fft))
        
        # For peak at m: d(conv[m])/d(phase[k]) = sum_j [d(h[j])/d(phase[k]) * h[m-j] + h[j] * d(h[m-j])/d(phase[k])]
        
        # This is expensive to compute directly. Use a simpler approach:
        # Perturb each phase slightly and estimate gradient
        grad = np.zeros_like(current_phase)
        
        # Sample a subset of phases for gradient estimation to save time
        n_sample = min(50, len(current_phase))
        sample_idx = np.random.choice(len(current_phase), n_sample, replace=False)
        
        eps = 1e-4
        for k in sample_idx:
            phase_plus = current_phase.copy()
            phase_plus[k] += eps
            
            H_plus = amp_safe * np.exp(1j * phase_plus)
            h_plus = np.fft.irfft(H_plus, n=n_fft)[:N]
            h_plus = np.maximum(h_plus, 0.0)
            s_plus = np.sum(h_plus)
            if s_plus > 1e-12:
                h_plus = h_plus * (N / s_plus)
                conv_plus = fft_autoconv(h_plus)
                val_plus = np.max(conv_plus) / (N * N)
                grad[k] = (val_plus - current_val) / eps
        
        # Gradient descent step
        current_phase = current_phase - lr * grad
        
        # Adaptive learning rate
        if it % 10 == 0:
            if current_val < best_val * 0.99:
                lr = min(lr * 1.05, 0.2)
            else:
                lr *= 0.9
        
        if lr < 1e-6:
            break
    
    return best_h, best_val
>>>>>>> REPLACE

