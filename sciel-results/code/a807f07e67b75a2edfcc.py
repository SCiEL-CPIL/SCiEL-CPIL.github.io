<<<<<<< SEARCH
def freq_domain_refinement(h, start_time, max_time):
    """Frequency-domain projected gradient descent to further reduce peak autoconvolution.
    
    The autoconvolution is the inverse FFT of |F|^2 where F is the FFT of h.
    We work in the frequency domain: minimize max of ifft(|F|^2) subject to
    ifft(F) >= 0 and sum(ifft(F)) = N.
    
    Use projected gradient: step in frequency domain, project back to time domain
    by clipping negative values and renormalizing.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    lr = 0.01
    no_improve = 0
    
    while time.time() - start_time < max_time:
        if no_improve > 50:
            break
        
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Identify peak lags
        threshold = max_c * 0.98
        peak_mask = (conv >= threshold).astype(np.float64)
        
        if np.sum(peak_mask) == 0:
            break
        
        # Gradient of max(conv) w.r.t. h: for each bin j, sum over peak m of h[m-j]
        # This is the same sensitivity as before
        pm_hat = np.fft.rfft(peak_mask, n=N_FFT)
        ch_hat = np.fft.rfft(current_h, n=N_FFT)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=N_FFT)[:N]
        
        # Normalize
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Gradient step in time domain (equivalent to frequency domain for linear ops)
        h_new = current_h - lr * sens_norm
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            lr = min(lr * 1.01, 0.1)
        else:
            lr *= 0.9
            if lr < 1e-8:
                break
    
    return best_h, best_val
=======
def smooth_max_refinement(h, start_time, max_time):
    """Refine using smooth max (log-sum-exp) surrogate with adaptive beta.
    
    Uses the smooth approximation to max to get a differentiable objective,
    enabling more stable gradient-based optimization.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0]
    lr = 0.01
    no_improve = 0
    
    for beta in beta_schedule:
        if time.time() - start_time > max_time:
            break
        
        no_improve = 0
        lr = 0.01
        
        while time.time() - start_time < max_time and no_improve < 30:
            s = np.sum(current_h)
            if s < 1e-12:
                break
            current_h = current_h * (N / s)
            
            conv = fft_convolve(current_h)
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Smooth max gradient: softmax weights
            conv_clipped = np.maximum(conv, 1e-15)
            log_weights = beta * (conv_clipped - max_c) / max_c
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)
            
            # Gradient of smooth_max(conv) w.r.t. h
            # d(smooth_max)/d(conv[m]) = weights[m]
            # d(conv[m])/d(h[j]) = h[m-j]
            w_hat = np.fft.rfft(weights, n=N_FFT)
            ch_hat = np.fft.rfft(current_h, n=N_FFT)
            grad = np.fft.irfft(w_hat * np.conj(ch_hat), n=N_FFT)[:N]
            
            # Center gradient (zero-sum for normalization invariance)
            grad_centered = grad - np.mean(grad)
            
            # Relative step
            h_safe = np.maximum(current_h, 1e-10)
            rel_grad = np.clip(grad_centered / h_safe, -2.0, 2.0)
            
            h_new = current_h * (1.0 - lr * rel_grad)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            conv_new = fft_convolve(h_new)
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < current_val:
                current_h = h_new
                lr = min(lr * 1.02, 0.08)
            else:
                lr *= 0.9
    
    return best_h, best_val
>>>>>>> REPLACE

