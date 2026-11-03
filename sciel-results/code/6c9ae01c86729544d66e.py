"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_optimize(h0, time_budget=60.0):
    """
    Spectral optimization: minimize max|autoconv| by flattening |H[k]|^2.
    
    Approach: The autoconvolution a[n] = sum_k |H[k]|^2 exp(2πikn/N) / N.
    We want to minimize max_n |a[n]|. This is equivalent to minimizing the
    peak of the inverse DFT of |H[k]|^2.
    
    We optimize the phases of H[k] (subject to conjugate symmetry for real h)
    to minimize the peak of the resulting autoconvolution.
    """
    n = len(h0)
    t_start = time.time()
    deadline = t_start + time_budget
    
    # Work with full DFT
    H = np.fft.fft(h0)
    
    # For real h, H[k] = conj(H[n-k]). We optimize phases of H[1..n/2-1] and magnitudes.
    # Strategy: fix magnitudes, optimize phases to minimize peak of |H|^2 inverse DFT.
    
    best_h = h0.copy()
    best_val = _score(h0)
    
    # Extract phases and magnitudes
    mags = np.abs(H)
    phases = np.angle(H)
    
    # For real signal: H[0] is real, H[n/2] is real, H[k] = conj(H[n-k])
    # We'll optimize the phases of H[1..n/2-1]
    
    # Use gradient descent on phases in frequency domain
    lr = 0.05
    for step in range(500):
        if time.time() > deadline - 5:
            break
        
        # Compute current |H|^2
        H_mag2 = mags * mags
        
        # Inverse DFT to get autoconvolution (up to scaling)
        ac = np.fft.ifft(H_mag2).real
        
        # Peak of |ac|
        peak_idx = np.argmax(np.abs(ac))
        peak_val = ac[peak_idx]
        
        # Gradient of peak w.r.t. phases
        # ac[n] = (1/N) sum_k |H[k]|^2 exp(2πikn/N)
        # d(ac[n])/d(phase_k) = (1/N) |H[k]|^2 * i * exp(2πikn/N)
        # We want to minimize |ac[peak_idx]|, so gradient direction is -sign(ac[peak_idx]) * conj(conj(exp(2πi*peak_idx*k/N)))
        
        # Compute gradient for each k
        grad = np.zeros(n, dtype=complex)
        k_vals = np.arange(n)
        exp_term = np.exp(2j * np.pi * peak_idx * k_vals / n)
        grad = (1.0 / n) * H_mag2 * 1j * exp_term
        
        # For real constraint: phase_k and phase_{n-k} are coupled
        # d(phase_k) = -d(phase_{n-k})
        # Effective gradient for independent phases (k=1..n/2-1):
        # grad_eff[k] = grad[k] - conj(grad[n-k]) (imaginary part matters)
        
        # Project gradient to maintain conjugate symmetry
        # Only update phases for k=1..n/2-1
        half = n // 2
        for k in range(1, half):
            if time.time() > deadline - 3:
                break
            # Gradient contribution for phase_k
            g_k = np.imag(grad[k] * np.exp(-1j * phases[k]))
            g_nk = np.imag(grad[n-k] * np.exp(-1j * phases[n-k]))
            # Combined gradient for phase_k (since phase_{n-k} = -phase_k)
            g_eff = g_k + g_nk
            
            # Update phase
            phases[k] -= lr * g_eff
            phases[n-k] = -phases[k]
        
        # Reconstruct H
        H_new = mags * np.exp(1j * phases)
        # Enforce conjugate symmetry
        H_new[0] = mags[0] * (1 if phases[0] >= 0 else -1)
        if n % 2 == 0:
            H_new[half] = mags[half] * (1 if phases[half] >= 0 else -1)
        
        h_new = np.fft.ifft(H_new).real
        
        # Normalize to keep sum fixed (scale invariant, but keep consistent)
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-9:
            h_new = h_new * (np.sum(h0) / s_new)
        
        val = _score(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            # Update H, mags, phases
            H = np.fft.fft(h_new)
            mags = np.abs(H)
            phases = np.angle(H)
            lr = min(lr * 1.05, 0.2)
        else:
            lr = max(lr * 0.9, 0.005)
            # Revert
            H = np.fft.fft(h0)
            mags = np.abs(H)
            phases = np.angle(H)
            break
    
    return best_h, best_val

def _spectral_flat_init():
    """
    Initialize with a sequence that has flat |H[k]|^2 spectrum.
    Use a chirp sequence which has approximately constant magnitude DFT.
    """
    i = np.arange(N)
    # Quadratic chirp
    h = np.cos(2.0 * np.pi * i**2 / N)
    return h

def _spectral_anneal(h0, time_budget=40.0):
    """
    Simulated annealing on sign flips, guided by spectral peak.
    """
    t_start = time.time()
    deadline = t_start + time_budget
    
    h = h0.copy()
    best_h = h.copy()
    best_val = _score(h)
    
    # Compute |H|^2 for quick evaluation
    H = np.fft.fft(h)
    H_mag2 = np.abs(H)**2
    ac = np.fft.ifft(H_mag2).real
    peak = np.max(np.abs(ac))
    
    T = 1.0
    cooling = 0.999
    
    for step in range(10000):
        if time.time() > deadline - 5:
            break
        
        # Pick random index to flip
        idx = np.random.randint(N)
        h[idx] = -h[idx]
        
        # Update H incrementally
        delta = 2 * h[idx] * (-1)  # change in h[idx]
        # Actually: h[idx] changed by 2*h_old[idx]
        # H[k] += delta * exp(-2πik idx/N)
        k_vals = np.arange(N)
        delta_H = 2.0 * h[idx] * np.exp(-2j * np.pi * k_vals * idx / N)
        H += delta_H
        H_mag2 = np.abs(H)**2
        ac = np.fft.ifft(H_mag2).real
        new_peak = np.max(np.abs(ac))
        
        new_val = _score(h)
        
        if new_val < best_val:
            best_val = new_val
            best_h = h.copy()
        elif np.random.rand() < np.exp(-(new_val - best_val) / max(T, 1e-10)):
            pass  # accept
        else:
            # Revert
            h[idx] = -h[idx]
            H -= delta_H
            H_mag2 = np.abs(H)**2
        
        T *= cooling
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    # Initialize with chirp sequence (flat spectrum)
    i = np.arange(N)
    h_init = np.cos(2.0 * np.pi * i**2 / N)
    
    # Ensure non-zero sum
    s = np.sum(h_init)
    if abs(s) < 1e-9:
        h_init = np.cos(2.0 * np.pi * i**2 / (N + 1))
        s = np.sum(h_init)
    
    best_h = h_init.copy()
    best_score = _score(h_init)
    
    # Spectral phase optimization
    if time.time() < deadline - 30:
        h_opt, val_opt = _spectral_optimize(h_init, time_budget=50.0)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Spectral annealing on best
    if time.time() < deadline - 15:
        h_anneal, val_anneal = _spectral_anneal(best_h, time_budget=40.0)
        if val_anneal < best_score:
            best_score = val_anneal
            best_h = h_anneal.copy()
    
    # Final local search: sign flips
    if time.time() < deadline - 5:
        h_local = best_h.copy()
        best_local = best_score
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(N):
                if time.time() > deadline - 2:
                    break
                h_local[idx] = -h_local[idx]
                new_val = _score(h_local)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_local[idx] = -h_local[idx]
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_local.copy()
    
    return best_h

# EVOLVE-BLOCK-END
