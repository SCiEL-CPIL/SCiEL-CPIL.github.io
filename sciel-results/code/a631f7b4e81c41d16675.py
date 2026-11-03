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

def _spectral_flatten(h, max_iters=200, tol=1e-10):
    """
    Iteratively flatten the magnitude spectrum of h while preserving sum(h).
    
    The key insight: autoconv max ≈ max_k |H[k]|^2 / N_fft, and (sum h)^2 = |H[0]|^2.
    We want to minimize max_k |H[k]|^2 / |H[0]|^2.
    
    Strategy: Use iterative spectral shaping. At each step:
    1. Compute H = DFT(h)
    2. Compute target spectrum: flat for k≠0, preserve H[0]
    3. Update h via gradient descent on the spectral flatness objective
    """
    t_start = time.time()
    deadline = t_start + 90.0
    
    h = h.copy()
    n = len(h)
    s0 = np.sum(h)
    
    if abs(s0) < 1e-9:
        return h, 1e10
    
    # Normalize so sum = 1
    h = h / s0
    best_score = _score(h)
    best_h = h.copy()
    
    # Use FFT size for autoconv
    ac_size = 2 * n - 1
    fft_size = 1
    while fft_size < ac_size:
        fft_size *= 2
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Compute current spectrum
        H = np.fft.rfft(h, fft_size)
        H0 = H[0]  # This is sum(h) = 1 (approximately)
        
        # Compute max magnitude (excluding DC)
        mag = np.abs(H)
        mag_no_dc = mag.copy()
        mag_no_dc[0] = 0
        max_mag = np.max(mag_no_dc)
        
        if max_mag < 1e-12:
            break
        
        # Gradient descent: minimize max_k |H[k]|^2 for k≠0
        # d|H[k]|^2/dh = 2*Re(H[k] * conj(DFT_basis))
        # We use a soft-max approximation: minimize sum_k exp(beta * |H[k]|^2)
        beta = 50.0
        weights = np.exp(beta * mag_no_dc**2 / (max_mag**2 + 1e-15))
        weights[0] = 0  # Don't touch DC
        
        # Gradient of the soft-max objective w.r.t. h
        # d/dh_j sum_k w_k |H[k]|^2 = 2 * ifft(w * H * conj(conj(DFT_basis)))
        # More simply: gradient = 2 * ifft(weights * H * np.conj(H) * ... )
        # Actually: d|H[k]|^2/dh = 2*Re(H[k] * exp(2πijk/n))
        # So gradient = 2 * ifft(weights * H * conj(exp(-2πijk/n)))... 
        # Let's use: grad_j = 2 * sum_k weights[k] * Re(H[k] * exp(2πijk/n))
        # = 2 * Re(ifft(weights * H * exp(2πijk/n)))... 
        
        # Simpler: use the fact that ifft of (weights * H * conj(H_conj)) 
        # Actually, let's just compute gradient numerically via FFT
        # grad = 2 * ifft(weights * H * np.conj(np.fft.rfft(np.ones_like(h), fft_size)))
        # Hmm, let me think more carefully.
        
        # H[k] = sum_j h[j] exp(-2πijk/fft_size)
        # |H[k]|^2 = H[k] * conj(H[k])
        # d|H[k]|^2/dh[j] = 2*Re(conj(H[k]) * exp(-2πijk/fft_size))
        # So gradient_j = sum_k weights[k] * 2*Re(conj(H[k]) * exp(-2πijk/fft_size))
        # = 2 * Re(ifft(weights * conj(H) * exp(-2πijk/fft_size)))
        # But ifft already includes the exp(2πijk/fft_size), so:
        # gradient = 2 * Re(ifft(weights * conj(H)))
        
        grad = 2.0 * np.fft.irfft(weights * np.conj(H), fft_size)[:n]
        
        # Subtract mean to preserve sum(h) = 1
        grad -= np.mean(grad)
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Use a small step
        lr = 0.01
        h_new = h - lr * grad / grad_norm
        
        # Renormalize to sum = 1
        h_new_sum = np.sum(h_new)
        if abs(h_new_sum) > 1e-9:
            h_new = h_new / h_new_sum
        
        new_score = _score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.5
            h_new = h - lr * grad / grad_norm
            h_new_sum = np.sum(h_new)
            if abs(h_new_sum) > 1e-9:
                h_new = h_new / h_new_sum
            new_score = _score(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
                h = h_new
            else:
                # Try even smaller step
                lr *= 0.5
                h_new = h - lr * grad / grad_norm
                h_new_sum = np.sum(h_new)
                if abs(h_new_sum) > 1e-9:
                    h_new = h_new / h_new_sum
                new_score = _score(h_new)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new.copy()
                    h = h_new
                else:
                    break
    
    return best_h, best_score

def _chirp_initializations():
    """Generate diverse chirp-like initializations."""
    i = np.arange(N)
    inits = []
    
    # Standard chirps
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        h = np.exp(1j * 2.0 * np.pi * alpha * i**2 / N)
        h_real = np.real(h)
        s = np.sum(h_real)
        if abs(s) > 1e-6:
            inits.append(h_real / s)
    
    # Cosine chirps
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        h = np.cos(2.0 * np.pi * alpha * i**2 / N)
        s = np.sum(h)
        if abs(s) > 1e-6:
            inits.append(h / s)
    
    # Quadratic phase with different scales
    for scale in [N/4, N/2, N, 2*N]:
        h = np.cos(2.0 * np.pi * i**2 / scale)
        s = np.sum(h)
        if abs(s) > 1e-6:
            inits.append(h / s)
    
    # Random with spectral shaping potential
    np.random.seed(42)
    for _ in range(5):
        h = np.random.randn(N)
        s = np.sum(h)
        if abs(s) > 1e-6:
            inits.append(h / s)
    
    # All-ones baseline
    inits.append(np.ones(N) / N)
    
    return inits

def _local_search_sign_flip(h, max_rounds=5, deadline=None):
    """Local search by sign flipping individual elements."""
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for idx in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            h_refined[idx] = -h_refined[idx]
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] = -h_refined[idx]
        if not improved:
            break
    
    return h_refined, best_local

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    # Generate initializations
    inits = _chirp_initializations()
    
    best_score = np.inf
    best_h = inits[0].copy()
    
    # Score all initializations
    for h_init in inits:
        if time.time() > deadline - 60:
            break
        score = _score(h_init)
        if score < best_score:
            best_score = score
            best_h = h_init.copy()
    
    # Apply spectral flattening to top candidates
    candidates = []
    for h_init in inits[:8]:  # Take top 8 by initial score
        score = _score(h_init)
        candidates.append((score, h_init.copy()))
    
    candidates.sort(key=lambda x: x[0])
    
    for _, h_cand in candidates:
        if time.time() > deadline - 20:
            break
        h_opt, score_opt = _spectral_flatten(h_cand, max_iters=150)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    
    # Final local search
    if time.time() < deadline - 5:
        h_refined, best_local = _local_search_sign_flip(best_h, max_rounds=4, deadline=deadline)
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h

# EVOLVE-BLOCK-END
