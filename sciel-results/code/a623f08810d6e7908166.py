"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def linear_autoconv(h):
        """Compute linear autoconvolution using FFT."""
        n = len(h)
        size = 2 * n
        H = np.fft.rfft(h, n=size)
        conv = np.fft.irfft(H * H, n=size)[:2*n-1]
        return conv
    
    def score_from_h(h):
        """Compute normalized maximum of autoconvolution."""
        conv = linear_autoconv(h)
        s = np.sum(h**2)
        if s < 1e-12:
            return 1e10
        return np.max(conv) / s
    
    def h_from_magnitudes(mags):
        """Reconstruct h from DFT magnitudes with zero phase, ensure nonnegativity."""
        # mags are for the full N-point DFT (real, nonnegative)
        # Zero phase means h is symmetric around 0
        # We need to handle the DC component and Nyquist if N is even
        H = np.zeros(N, dtype=complex)
        H[0] = mags[0]  # DC
        for k in range(1, N//2 + 1):
            if k < N:
                H[k] = mags[k]
                H[N-k] = mags[k]  # Conjugate symmetry for real h
        h = np.fft.ifft(H).real
        h = np.maximum(h, 0.0)
        return h
    
    def mags_from_h(h):
        """Extract DFT magnitudes from h."""
        H = np.fft.fft(h)
        mags = np.abs(H)
        return mags
    
    # Initialize with all ones
    h_init = np.ones(N)
    mags_init = mags_from_h(h_init)
    
    best_h = h_init.copy()
    best_score = score_from_h(best_h)
    
    # Try some simple initializations in magnitude space
    # 1. All ones (current best)
    # 2. Low-pass filtered magnitudes
    if time.time() < deadline - 10:
        # Create a low-pass magnitude spectrum
        freqs = np.fft.fftfreq(N)
        lowpass_mags = mags_init * np.exp(-10 * freqs**2)
        h_lp = h_from_magnitudes(lowpass_mags)
        s_lp = score_from_h(h_lp)
        if s_lp < best_score:
            best_score = s_lp
            best_h = h_lp.copy()
    
    # 3. High-pass filtered magnitudes
    if time.time() < deadline - 10:
        highpass_mags = mags_init * (1 - np.exp(-10 * freqs**2))
        h_hp = h_from_magnitudes(highpass_mags)
        s_hp = score_from_h(h_hp)
        if s_hp < best_score:
            best_score = s_hp
            best_h = h_hp.copy()
    
    # Spectral gradient descent on magnitudes
    # We want to minimize max(autoconv(h)) / sum(h^2)
    # Work in log-magnitude space for better conditioning
    if time.time() < deadline - 5:
        # Start from best_h found so far
        mags = mags_from_h(best_h)
        log_mags = np.log(mags + 1e-10)
        
        lr = 0.01
        prev_score = best_score
        
        for it in range(200):
            if time.time() > deadline - 2:
                break
            
            # Reconstruct h from current log_mags
            mags = np.exp(log_mags)
            h = h_from_magnitudes(mags)
            
            # Compute score
            score = score_from_h(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
            
            # Compute gradient with respect to log_mags
            # This is complex, so use finite differences for a few coordinates
            # or use a simpler heuristic: reduce magnitudes at frequencies
            # that contribute to the peak of autoconvolution
            
            # Simple heuristic: if score increased, reduce learning rate
            if score > prev_score:
                lr *= 0.5
            else:
                # Occasionally increase lr if stuck
                if it > 50 and score < prev_score * 0.99:
                    lr = min(lr * 1.1, 0.05)
            
            # Gradient step: perturb log_mags slightly
            # Use random perturbations to explore
            if it % 10 == 0:
                # Try a few random perturbations
                for trial in range(3):
                    if time.time() > deadline - 2:
                        break
                    perturbation = np.random.randn(N) * lr
                    new_log_mags = log_mags + perturbation
                    new_mags = np.exp(new_log_mags)
                    new_h = h_from_magnitudes(new_mags)
                    new_score = score_from_h(new_h)
                    if new_score < score:
                        log_mags = new_log_mags
                        score = new_score
                        if score < best_score:
                            best_score = score
                            best_h = new_h.copy()
                        break
            
            prev_score = score
    
    # Final local refinement in time domain
    if time.time() < deadline - 3:
        # Simple smoothing step
        for smooth_iter in range(20):
            if time.time() > deadline - 2:
                break
            h_smooth = np.convolve(best_h, np.ones(5)/5, mode='same')
            h_smooth = np.maximum(h_smooth, 0.0)
            s_smooth = score_from_h(h_smooth)
            if s_smooth < best_score:
                best_score = s_smooth
                best_h = h_smooth.copy()
            else:
                break
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
