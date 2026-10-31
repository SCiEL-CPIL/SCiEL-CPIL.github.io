"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_cubic_phase(N, T):
    """
    Generate a sequence based on the real part of a cubic phase profile.
    h[i] = cos(pi * (i/T)^3)
    Cubic phase codes have better sidelobe properties than quadratic.
    """
    i = np.arange(N)
    t = i / T
    phase = np.pi * t**3
    return np.cos(phase)

def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def _triple_flip(h, max_passes=2, deadline=None):
    """Local optimization: try flipping triples of elements, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Sample a subset of triples to avoid O(N^3) complexity
        rng = np.random.default_rng(12345 + int(time.time() * 1000) % 1000)
        n_triples = 5000
        for _ in range(n_triples):
            if deadline is not None and time.time() > deadline:
                break
            idx = rng.choice(N, 3, replace=False)
            i, j, k = idx
            h[i] = -h[i]
            h[j] = -h[j]
            h[k] = -h[k]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
                h[j] = -h[j]
                h[k] = -h[k]
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_h = np.ones(N)
    best_val = _score(best_h)

    # List of initializations
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cubic phase with various T values
    # Try to make T irrational relative to N
    x = np.arange(N)
    for T in [N / np.sqrt(2), N / np.pi, N / np.e, N / 1.414, N / 1.732, 300.0, 350.0, 450.0]:
        inits.append(_gen_cubic_phase(N, T))
        
    # 3. Quadratic phase (chirp) for comparison
    for T in [N / np.sqrt(2), N / np.pi, N / np.e]:
        phase = np.pi * (x / T)**2
        inits.append(np.cos(phase))
        
    # 4. Random with continuous values
    rng = np.random.default_rng(42)
    for _ in range(5):
        h = rng.uniform(-1, 1, size=N)
        inits.append(h)
        
    # 5. Random sign flips
    for _ in range(5):
        h = rng.choice([-1.0, 1.0], size=N)
        inits.append(h)
        
    # 6. Block patterns
    for _ in range(5):
        h = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.randint(1, N//10 + 1)
            sign = rng.choice([-1.0, 1.0])
            h[pos:pos+block_len] = sign
            pos += block_len
        inits.append(h)

    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization
        try:
            # Start with moderate beta, increase for refinement
            for beta in [20, 50, 100]:
                if time.time() > deadline - 10:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
                h_opt = res.x
                
                # Check validity
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_val:
                        best_val = val
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Final local search refinement
    if time.time() < deadline - 5:
        # Single flip polish
        h_refined, val_refined = _local_sign_flip(best_h, max_passes=3, deadline=deadline-2)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()
            
        # Triple flip polish
        if time.time() < deadline - 2:
            h_refined2, val_refined2 = _triple_flip(best_h, max_passes=1, deadline=deadline-1)
            if val_refined2 < best_val:
                best_val = val_refined2
                best_h = h_refined2.copy()

    # Final check against baseline
    if best_val > _score(np.ones(N)):
        best_val = _score(np.ones(N))
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
