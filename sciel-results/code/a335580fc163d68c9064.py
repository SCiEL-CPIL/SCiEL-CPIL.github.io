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

def _construct_flat_spectrum_sequence():
    """
    Construct a real sequence with nearly flat non-DC spectrum.
    Uses quadratic phase (chirp) which has flat magnitude spectrum.
    """
    i = np.arange(N)
    # Quadratic phase chirp: h[i] = cos(2*pi*i^2/N)
    # This has approximately flat magnitude spectrum for prime N
    h = np.cos(2.0 * np.pi * i**2 / N)
    return h

def _construct_zadoff_chu_like():
    """
    Construct a Zadoff-Chu-like sequence for real-valued output.
    ZC sequences have constant amplitude and zero autocorrelation sidelobes.
    """
    i = np.arange(N)
    # For real sequences, use cos of quadratic phase
    # ZC: exp(-j*pi*k*i*(i+1)/N) for complex, take real part
    h = np.cos(np.pi * i * (i + 1) / N)
    return h

def _optimize_flat_spectrum():
    """
    Optimize the phase of the non-DC DFT components to minimize peak |H[k]|^2.
    Start from a chirp and refine.
    """
    i = np.arange(N)
    # Start with quadratic phase
    phases = 2.0 * np.pi * i**2 / N
    h = np.cos(phases)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Local search: try small phase perturbations
    t_start = time.time()
    deadline = t_start + 100.0
    
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try flipping signs of individual elements
        for idx in range(0, N, 2):  # Sample every other for speed
            if time.time() > deadline - 5:
                break
            h[idx] = -h[idx]
            new_score = _score(h)
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                h[idx] = -h[idx]
        
        if not improved:
            break
    
    return best_h, best_score

def _gradient_descent_on_phases():
    """
    Gradient descent on the phase of the DFT to flatten the spectrum.
    """
    i = np.arange(N)
    phases = 2.0 * np.pi * i**2 / N
    h = np.cos(phases)
    
    best_h = h.copy()
    best_score = _score(h)
    
    t_start = time.time()
    deadline = t_start + 80.0
    
    lr = 0.1
    for step in range(200):
        if time.time() > deadline - 5:
            break
        
        # Compute current score
        current_score = _score(h)
        
        # Estimate gradient via finite differences on a few elements
        grad = np.zeros(N)
        eps = 0.01
        for idx in range(0, N, 4):  # Sample for speed
            if time.time() > deadline - 3:
                break
            h_old = h[idx]
            h[idx] += eps
            score_plus = _score(h)
            h[idx] -= 2 * eps
            score_minus = _score(h)
            h[idx] = h_old
            grad[idx] = (score_plus - score_minus) / (2 * eps)
        
        # Update
        h = h - lr * grad
        # Keep signs: project to nearest sign pattern
        h = np.sign(h)
        
        new_score = _score(h)
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            lr = min(lr * 1.1, 0.5)
        else:
            h = best_h.copy()
            lr = max(lr * 0.8, 0.01)
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Strategy 1: Flat spectrum chirp sequences
    candidates = []
    
    # Quadratic phase chirp
    i = np.arange(N)
    candidates.append(np.cos(2.0 * np.pi * i**2 / N))
    candidates.append(np.cos(np.pi * i**2 / N))
    candidates.append(np.cos(2.0 * np.pi * i**2 / (N+1)))
    
    # Zadoff-Chu like
    candidates.append(np.cos(np.pi * i * (i + 1) / N))
    candidates.append(np.cos(np.pi * i * (i + 1) / (N+1)))
    
    # Barker-like patterns
    for period in [4, 6, 8, 12]:
        pattern = np.array([1, -1, 1, -1])[:period]
        h = np.tile(pattern, N // period + 1)[:N]
        candidates.append(h)
    
    # Random smoothed
    rng = np.random.default_rng(42)
    for seed in range(3):
        rng_local = np.random.default_rng(seed)
        noise = rng_local.standard_normal(N)
        kernel = np.ones(5) / 5
        smooth = np.convolve(noise, kernel, mode='same')
        candidates.append(np.sign(smooth))
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline - 30:
            break
        s = np.sum(h_cand)
        if abs(s) < 1e-6:
            continue
        # Normalize to have sum = 1 for fair comparison, but score is scale-invariant
        val = _score(h_cand)
        if val < best_score:
            best_score = val
            best_h = h_cand.copy()
    
    # Local search refinement on best candidate
    if time.time() < deadline - 10:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5:
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
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
