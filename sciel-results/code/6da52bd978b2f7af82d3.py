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

def _spectral_flatten(h, iterations=50):
    """
    Iteratively flatten the magnitude spectrum |H[k]| using projection methods.
    
    The autoconvolution spectrum is |H[k]|^2. We want to minimize max_k |H[k]|^2
    subject to sum(h) = 1. This is equivalent to making |H[k]| as flat as possible.
    """
    h = h.copy().astype(float)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, 1e10
    
    for it in range(iterations):
        H = np.fft.rfft(h, 2*N-1)
        mag = np.abs(H)
        
        peak = np.max(mag)
        if peak < 1e-12:
            break
        
        peak_idx = np.argmax(mag)
        
        n = np.arange(N)
        phase = np.exp(2j * np.pi * peak_idx * n / N)
        grad = 2.0 * np.real(H[peak_idx] * phase)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        step_size = 0.1 / (1.0 + it * 0.01)
        h_new = h - step_size * grad
        
        s_new = np.sum(h_new)
        h_new = h_new + (1.0 - s_new) / N
        
        h_new_score = _score(h_new)
        h_score = _score(h)
        
        if h_new_score < h_score:
            h = h_new
        else:
            step_size = step_size * 0.5
            h_new = h - step_size * grad
            s_new = np.sum(h_new)
            h_new = h_new + (1.0 - s_new) / N
            h_new_score = _score(h_new)
            if h_new_score < h_score:
                h = h_new
    
    return h, _score(h)

def _chirp_construction():
    """
    Construct chirp sequences with flat magnitude spectra.
    """
    n = np.arange(N)
    best_score = np.inf
    best_h = np.ones(N) / N
    
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 0.25, 0.75, 1.25, 2.5]:
        h = np.cos(np.pi * alpha * n**2 / N)
        s = np.sum(h)
        if abs(s) > 1e-9:
            h = h / s
            score = _score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        for beta in [0, 0.5, 1.0]:
            h = np.cos(np.pi * alpha * n**2 / N + beta * np.pi * n / N)
            s = np.sum(h)
            if abs(s) > 1e-9:
                h = h / s
                score = _score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    return best_h, best_score

def _zadoff_chu_like():
    """
    Construct Zadoff-Chu like sequences.
    """
    n = np.arange(N)
    best_score = np.inf
    best_h = np.ones(N) / N
    
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        h = np.cos(np.pi * alpha * n * (n + 1) / N)
        s = np.sum(h)
        if abs(s) > 1e-9:
            h = h / s
            score = _score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    return best_h, best_score

def _local_search(h, max_iterations=3):
    """Local search via sign flips."""
    h = h.copy()
    best_score = _score(h)
    
    for _ in range(max_iterations):
        improved = False
        for idx in range(N):
            h[idx] = -h[idx]
            new_val = _score(h)
            if new_val < best_score:
                best_score = new_val
                improved = True
            else:
                h[idx] = -h[idx]
        if not improved:
            break
    
    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    if time.time() < deadline - 60:
        h_chirp, score_chirp = _chirp_construction()
        if score_chirp < best_score:
            best_score = score_chirp
            best_h = h_chirp.copy()
    
    if time.time() < deadline - 50:
        h_zc, score_zc = _zadoff_chu_like()
        if score_zc < best_score:
            best_score = score_zc
            best_h = h_zc.copy()
    
    if time.time() < deadline - 20:
        h_flat, score_flat = _spectral_flatten(best_h, iterations=30)
        if score_flat < best_score:
            best_score = score_flat
            best_h = h_flat.copy()
    
    if time.time() < deadline - 5:
        h_local, score_local = _local_search(best_h, max_iterations=2)
        if score_local < best_score:
            best_score = score_local
            best_h = h_local.copy()
    
    return best_h

# EVOLVE-BLOCK-END
