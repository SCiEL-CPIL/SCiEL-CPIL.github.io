import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    def elapsed():
        return time.time() - start_time
    
    def metric(h):
        """Compute the normalized maximum autoconvolution metric."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        # Use FFT for speed
        F = np.fft.fft(hn, n=2*N)
        conv = np.fft.ifft(F * F).real[:2*N-1]
        return np.max(conv)
    
    # Initial guess: uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = metric(best_h)
    
    # Phase 1: Try structural candidates
    candidates = []
    
    # Uniform
    candidates.append(('uniform', np.ones(N)))
    
    # Triangular
    center = N // 2
    tri = np.array([1.0 - abs(i - center) / center if center > 0 else 1.0 for i in range(N)])
    tri = np.clip(tri, 0, None)
    candidates.append(('triangular', tri))
    
    # Gaussian
    x_grid = np.linspace(-3, 3, N)
    gauss = np.exp(-x_grid**2)
    candidates.append(('gaussian', gauss))
    
    # Inverted Gaussian
    inv_gauss = np.ones(N) - 0.5 * np.exp(-3.0 * np.linspace(-1, 1, N)**2)
    inv_gauss = np.clip(inv_gauss, 0, None)
    candidates.append(('inv_gauss', inv_gauss))
    
    # Two peaks
    two_peak = np.zeros(N)
    two_peak[N//4] = 1.0
    two_peak[3*N//4] = 1.0
    candidates.append(('two_peak', two_peak))
    
    # Plateaus
    for width_frac in [0.3, 0.5, 0.7]:
        w = int(N * width_frac)
        plateau = np.zeros(N)
        plateau[N//2 - w//2 : N//2 + w//2 + 1] = 1.0
        candidates.append((f'plateau_{width_frac}', plateau))
    
    # Ramp
    ramp = np.linspace(1, 0, N)
    ramp[ramp < 0] = 0
    candidates.append(('ramp', ramp))
    
    # Evaluate all candidates
    for name, h in candidates:
        if elapsed() > time_limit:
            break
        val = metric(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 2: Spectral equalization via Sinkhorn-Knopp on Fourier domain
    # The autoconvolution peak is related to the L-infinity norm of the autocorrelation
    # which equals the maximum of the inverse FFT of |F|^2.
    # We want to flatten the power spectrum to minimize this peak.
    
    def spectral_equalize(h, n_iters=100, eps=1e-10):
        """
        Apply Sinkhorn-Knopp style iterations to flatten the Fourier magnitude.
        Work in log-domain for stability.
        """
        h = h.copy()
        s = np.sum(h)
        if s < 1e-15:
            return h
        h = h / s
        
        for it in range(n_iters):
            if elapsed() > time_limit:
                break
            
            # Compute FFT
            F = np.fft.fft(h, n=2*N)
            
            # Get magnitude squared (power spectrum)
            power = np.abs(F)**2
            
            # We want to make power as flat as possible
            # Use multiplicative updates in the time domain
            # This is analogous to Sinkhorn but for spectral flattening
            
            # Compute target: inverse FFT of flat spectrum
            # If we want power[k] = C for all k, then |F[k]| = sqrt(C)
            # The inverse FFT of a constant magnitude spectrum gives us the correction
            
            # Alternative approach: iteratively adjust h to reduce spectral peaks
            # Find frequency bins with high power and reduce their contribution
            
            # Use a soft-max style approach: weights inversely proportional to power
            # to down-weight high-power frequencies
            
            # Compute correction in frequency domain
            # We want to multiply F by a factor that reduces high-power bins
            # Let's use: F_new[k] = F[k] * (mean_power / power[k])^alpha
            # This reduces high-power bins and boosts low-power bins
            
            alpha = 0.5  # Damping factor
            
            mean_power = np.mean(power)
            if mean_power < eps:
                break
            
            # Correction factor
            correction = (mean_power / (power + eps)) ** alpha
            correction = np.clip(correction, 0.1, 10.0)  # Clamp for stability
            
            # Apply correction in frequency domain
            F_corrected = F * correction
            
            # Transform back to time domain
            h_new = np.fft.ifft(F_corrected).real[:N]
            
            # Ensure non-negativity
            h_new = np.maximum(h_new, 0)
            
            # Normalize
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                break
            h_new = h_new / s_new
            
            # Check convergence
            if np.max(np.abs(h_new - h)) < 1e-8:
                h = h_new
                break
            
            h = h_new
        
        return h
    
    # Apply spectral equalization to best candidate
    if elapsed() < time_limit - 30:
        h_eq = spectral_equalize(best_h, n_iters=200)
        val_eq = metric(h_eq)
        if val_eq < best_val:
            best_val = val_eq
            best_h = h_eq.copy()
    
    # Phase 3: Gradient descent with spectral guidance
    # Use the spectral flatness as a guide for gradient updates
    
    if elapsed() < time_limit - 10:
        h_opt = best_h.copy()
        lr = 0.01
        momentum = 0.9
        v = np.zeros(N)
        
        for step in range(500):
            if elapsed() > time_limit:
                break
            
            # Compute gradient of metric w.r.t. h
            # Use finite differences or analytical gradient
            # For speed, use a few random directions for gradient estimation
            
            h_norm = h_opt / np.sum(h_opt)
            F = np.fft.fft(h_norm, n=2*N)
            conv = np.fft.ifft(F * F).real[:2*N-1]
            
            # Find the peak location
            peak_idx = np.argmax(conv)
            
            # The gradient of max(conv) w.r.t. h is related to the correlation
            # at the peak lag
            # Use subgradient: direction that reduces the peak
            
            # Compute correlation at peak lag
            if peak_idx < N:
                lag = peak_idx
            else:
                lag = 2*N - 1 - peak_idx
            
            # Gradient estimate: shift h by lag and correlate
            if lag == 0:
                grad = np.ones(N)
            else:
                # Circular shift for correlation
                shifted = np.roll(h_norm, -lag)
                grad = shifted
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Update with momentum
            v = momentum * v - lr * grad
            h_opt = h_opt + v
            
            # Project to non-negative simplex
            h_opt = np.maximum(h_opt, 0)
            s_opt = np.sum(h_opt)
            if s_opt < 1e-15:
                h_opt = np.ones(N) / N
            else:
                h_opt = h_opt / s_opt
            
            # Check improvement
            val_opt = metric(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
            
            # Decay learning rate
            lr *= 0.995
    
    # Phase 4: Fine-tuning with simulated annealing
    if elapsed() < time_limit - 5:
        f = best_h.copy()
        current_score = best_val
        T = 0.05
        cooling_rate = 0.999
        min_T = 1e-5
        
        while T > min_T:
            if elapsed() > time_limit:
                break
            
            # Perturb
            old_f = f.copy()
            idx = np.random.randint(0, N, size=5)
            f[idx] += np.random.normal(0, 0.05, size=5)
            f[idx] = np.maximum(f[idx], 0)
            
            norm = np.sum(f)
            if norm < 1e-10:
                f = np.ones(N) / N
            else:
                f /= norm
            
            new_score = metric(f)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_val:
                    best_val = new_score
                    best_h = f.copy()
            else:
                delta = new_score - current_score
                if np.random.rand() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    f = old_f
            
            T *= cooling_rate
    
    return list(best_h)
