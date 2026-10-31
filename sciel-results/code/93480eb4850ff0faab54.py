import numpy as np
import time

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
        
    def smooth_max_and_grad_h(h, beta=100.0):
        """
        Compute smooth max of autoconvolution and its gradient w.r.t h.
        h is assumed to be non-negative and sum to 1.
        Returns: (smooth_max_value, grad_h)
        """
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        
        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        sum_exp = np.sum(exp_conv)
        s_max = m + np.log(sum_exp) / beta
        
        weights = exp_conv / sum_exp
        
        # Gradient of smooth max w.r.t h
        # dS/dh_j = 2 * sum_k weights[k] * h[k-j]
        # This is the correlation of h and weights.
        W = np.fft.fft(weights, n=size)
        corr = np.fft.ifft(W * np.conj(H)).real
        
        # Extract the relevant N values
        # corr[j] corresponds to the sum for index j
        grad_h = 2.0 * corr[:N]
        
        return s_max, grad_h

    def objective_and_grad(params, beta):
        """
        Objective and gradient for optimizer.
        params: coefficients of the cosine series.
        h_i = exp(sum_m params[m] * cos(2*pi*m*(i-N/2)/N))
        """
        M = len(params)
        i = np.arange(N)
        center = N / 2.0
        
        # Compute h
        h_unnorm = np.zeros(N)
        for m in range(M):
            if m == 0:
                h_unnorm += params[0]
            else:
                h_unnorm += params[m] * np.cos(2 * np.pi * m * (i - center) / N)
        
        h_unnorm = np.exp(h_unnorm)
        Z = np.sum(h_unnorm)
        h = h_unnorm / Z
        
        # Compute smooth max and grad w.r.t h
        s_max, grad_h = smooth_max_and_grad_h(h, beta)
        
        # Compute grad w.r.t params
        grad_params = np.zeros(M)
        
        for m in range(M):
            if m == 0:
                phi = np.ones(N)
            else:
                phi = np.cos(2 * np.pi * m * (i - center) / N)
            
            # bar_phi = sum h_j phi_j
            bar_phi = np.dot(h, phi)
            
            # dh_da = h * (phi - bar_phi)
            dh_da = h * (phi - bar_phi)
            
            # grad_params[m] = sum_i grad_h_i * dh_da_i
            grad_params[m] = np.dot(grad_h, dh_da)
            
        return s_max, grad_params

    # Optimization
    best_h = None
    best_val = np.inf
    
    # Try different numbers of cosine terms
    M_vals = [5, 10, 15]
    
    for M in M_vals:
        if time_left() < 20:
            break
            
        # Random initializations
        for _ in range(10):
            if time_left() < 10:
                break
                
            # Random params
            # a_0 controls the mean, but we normalize, so it's less important.
            # Let's initialize a_0 to 0.
            params_init = np.zeros(M)
            for m in range(1, M):
                params_init[m] = np.random.uniform(-2, 2)
                
            try:
                # Use L-BFGS-B for optimization
                from scipy.optimize import minimize
                res = minimize(
                    lambda p: objective_and_grad(p, 100.0)[0],
                    params_init,
                    jac=lambda p: objective_and_grad(p, 100.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                # Reconstruct h
                i = np.arange(N)
                center = N / 2.0
                h_unnorm = np.zeros(N)
                for m in range(M):
                    if m == 0:
                        h_unnorm += res.x[0]
                    else:
                        h_unnorm += res.x[m] * np.cos(2 * np.pi * m * (i - center) / N)
                
                h_unnorm = np.exp(h_unnorm)
                h = h_unnorm / np.sum(h_unnorm)
                
                val = compute_autoconv_max(h)
                
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                    
            except Exception:
                continue

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
        
    # Refinement with direct gradient descent on h
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        eta = 0.01
        
        for k in range(100):
            if time_left() < 5:
                break
                
            s_max, grad_h = smooth_max_and_grad_h(current_h, 100.0)
            
            # Project gradient to maintain sum=1 and non-negativity?
            # Simple step:
            new_h = current_h - eta * grad_h
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s > 0:
                new_h /= s
            else:
                break
                
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.9
                if eta < 1e-6:
                    break

    # NEW: Spectral Equalization (Sinkhorn-like)
    # Idea: Minimize peak autoconvolution by flattening the power spectrum |H|^2.
    # The autoconvolution is the inverse FFT of |H|^2.
    # If we can make |H|^2 as flat as possible (while maintaining H[0]=1 and non-negativity of h),
    # the inverse FFT (autoconv) will have minimal peaks (ideally a delta function, but we have finite support).
    # We iterate:
    # 1. Compute H = FFT(h)
    # 2. Compute target spectrum: flat where |H| is small, preserve where large?
    #    Actually, we want to minimize max(ifft(|H|^2)).
    #    This is equivalent to minimizing the L-infinity norm of the time-domain signal.
    #    A common heuristic is to iteratively project the spectrum onto a "flat" shape
    #    and the time domain onto non-negativity.
    
    if time_left() > 10:
        h_curr = best_h.copy()
        
        # We will iterate in the frequency domain
        # Let's try to equalize the magnitude of the FFT coefficients.
        # Target: |H[k]|^2 should be constant for k != 0?
        # If |H[k]|^2 = C for all k, then ifft(|H|^2) is a delta function (scaled).
        # But h is real and non-negative, so H is Hermitian.
        # We can't make H constant because H[0] = sum(h) = 1.
        # So we want |H[k]| to be as small as possible for k != 0.
        
        # Algorithm:
        # 1. Compute H = FFT(h)
        # 2. Scale H such that |H[k]| is closer to a target value.
        #    A simple approach: H_new[k] = H[k] * (target_mag / |H[k]|)
        #    What is target_mag?
        #    If we set target_mag = 1/sqrt(N) (uniform energy), we get a noise-like h.
        #    But we need h >= 0.
        
        # Let's use a "water-filling" approach on the spectrum.
        # We want to minimize max(conv).
        # Conv is determined by |H|^2.
        
        # Let's try a simple iterative projection:
        # 1. Start with h_curr
        # 2. Loop:
        #    a. H = FFT(h_curr)
        #    b. Compute conv = ifft(|H|^2)
        #    c. Identify peaks in conv.
        #    d. Adjust H to reduce the energy at frequencies that contribute to these peaks?
        #       This is complex.
        
        # Alternative: Directly optimize the spectrum.
        # Let H be the complex spectrum.
        # Constraint: h = ifft(H) >= 0, sum(h)=1.
        # Objective: min max(ifft(|H|^2))
        
        # We can use gradient descent on the spectrum H (real and imag parts).
        # But H must satisfy H[k] = conj(H[-k]).
        
        # Let's stick to time-domain gradient descent but with a better initialization
        # derived from spectral considerations.
        
        # Initialize h with a shape that has a flatter spectrum.
        # A rectangular window has sinc spectrum (high sidelobes).
        # A Hann window has lower sidelobes.
        # Let's try initializing with a Hann window or similar.
        
        if time_left() > 5:
            # Generate a Hann window
            w_hann = np.hanning(N)
            h_hann = w_hann / np.sum(w_hann)
            val_hann = compute_autoconv_max(h_hann)
            
            if val_hann < best_val:
                best_val = val_hann
                best_h = h_hann.copy()
            
            # Generate a Blackman window
            w_black = np.blackman(N)
            h_black = w_black / np.sum(w_black)
            val_black = compute_autoconv_max(h_black)
            
            if val_black < best_val:
                best_val = val_black
                best_h = h_black.copy()

    return best_h.tolist()
