"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """Smooth objective for gradient-based optimization."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gradient_step_smooth(h, beta=100.0, lr=0.01):
    """
    One gradient descent step on the smooth objective.
    Minimize soft_max_abs(autoconv(h)) / (sum h)^2
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    # Compute gradient of soft_max_abs
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient of soft_max_abs w.r.t. ac_k
    # d(soft_max_abs)/d(ac_k) = beta * weights[k] * sign(ac_k)
    grad_sm = np.zeros(size)
    for k in range(size):
        if abs_ac[k] > 1e-12:
            grad_sm[k] = beta * weights[k] * np.sign(ac[k])
        else:
            grad_sm[k] = beta * weights[k]
    
    # Gradient of autoconv w.r.t. h
    # ac[k] = sum_i h[i] h[i+k] (for k >= 0)
    # grad_h[j] = sum_k grad_sm[k] * d(ac[k])/d(h[j])
    
    grad_h = np.zeros(n)
    
    for k in range(size):
        if abs(grad_sm[k]) < 1e-12:
            continue
        gk = grad_sm[k]
        
        if k == 0:
            # ac[0] = sum_i h[i]^2
            # d(ac[0])/d(h[j]) = 2*h[j]
            grad_h += gk * 2.0 * h
        else:
            if k < n:
                # ac[k] = sum_{i=0}^{n-1-k} h[i] h[i+k]
                # d(ac[k])/d(h[j]) = h[j+k] (if j < n-k) + h[j-k] (if j >= k)
                idx1 = np.arange(n - k)
                grad_h[idx1] += gk * h[idx1 + k]
                idx2 = np.arange(k, n)
                grad_h[idx2] += gk * h[idx2 - k]
            else:
                # k >= n: ac[k] = sum_{i=k-n+1}^{n-1} h[i] h[i-(k-n+1)]
                # This is the same as ac[n-k] by symmetry of autoconv
                k_sym = n - k
                if k_sym > 0 and k_sym < n:
                    idx1 = np.arange(n - k_sym)
                    grad_h[idx1] += gk * h[idx1 + k_sym]
                    idx2 = np.arange(k_sym, n)
                    grad_h[idx2] += gk * h[idx2 - k_sym]
    
    # Account for the denominator (sum h)^2
    # f = sm / s^2
    # df/dh = (grad_sm * s^2 - sm * 2*s * ones) / s^4
    #       = grad_sm / s^2 - 2*sm/s^3
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h
    
    sm_val = _soft_max_abs(ac, beta)
    grad_obj = grad_h / (s * s) - 2.0 * sm_val / (s * s * s) * np.ones(n)
    
    h_new = h - lr * grad_obj
    
    # Normalize to keep sum positive and reasonable
    s_new = np.sum(h_new)
    if abs(s_new) > 1e-9:
        # Scale to maintain similar magnitude
        target_sum = s
        h_new = h_new * (target_sum / s_new)
    
    return h_new

def _smooth_optimization(h, deadline, max_iters=200):
    """Run smooth gradient optimization with adaptive learning rate."""
    h = h.copy()
    best_val = _score(h)
    best_h = h.copy()
    
    lr = 0.01
    beta = 50.0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        h_new = _gradient_step_smooth(h, beta=beta, lr=lr)
        
        val_new = _score(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new
            lr *= 0.995
            # Gradually increase beta for sharper approximation
            if it % 20 == 0:
                beta = min(beta * 1.2, 200.0)
        else:
            lr *= 0.9
            # If stuck, try a larger beta
            if lr < 0.001:
                beta = min(beta * 1.5, 300.0)
                lr = 0.005
    
    return best_h, best_val

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
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _generate_sidon_like(n, seed=42):
    """
    Generate a sequence with Sidon-like properties using a deterministic construction.
    Sidon sets have low additive energy, which should minimize autoconvolution peaks.
    """
    rng = np.random.RandomState(seed)
    h = np.zeros(n)
    
    # Use a modular construction inspired by finite fields
    # For prime p, the set {x^2 mod p} has nice additive properties
    p = 401  # Prime > n
    
    # Generate quadratic residues mod p
    residues = set()
    for x in range(p):
        residues.add((x * x) % p)
    
    # Create a balanced sequence based on quadratic character
    for i in range(n):
        if i in residues:
            h[i] = 1.0
        else:
            h[i] = -1.0
    
    # Add some randomness to break perfect symmetry
    flip_mask = rng.random(n) < 0.05
    h[flip_mask] *= -1
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_val = 1e10
    
    np.random.seed(42)
    
    # Phase 1: Multiple diverse initializations
    inits = []
    
    # 1. Thue-Morse
    n_tm = N
    h_tm = np.zeros(n_tm)
    for i in range(n_tm):
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        h_tm[i] = 1.0 if count % 2 == 0 else -1.0
    inits.append(h_tm)
    
    # 2. Sidon-like (quadratic residues)
    inits.append(_generate_sidon_like(N, seed=42))
    inits.append(_generate_sidon_like(N, seed=123))
    
    # 3. Chirp sequence
    x = np.arange(N)
    inits.append(np.cos(np.pi * x**2 / N))
    
    # 4. Random smooth
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(N)
    kernel = np.ones(10) / 10
    smooth = np.convolve(noise, kernel, mode='same')
    inits.append(smooth)
    
    # 5. Cosine window
    inits.append(np.cos(np.pi * x / N))
    
    # 6. Random +/-1
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # 7. Random uniform
    inits.append(rng.uniform(-1, 1, size=N))
    
    # Optimize each initialization
    for h_init in inits:
        if time.time() > deadline - 10:
            break
        
        # Ensure sum is not too small
        s = np.sum(h_init)
        if abs(s) < N * 0.1:
            if s < 0:
                h_init = -h_init
            h_init = h_init + 0.5
            h_init = np.sign(h_init)
        
        # Smooth optimization
        h_opt, val_opt = _smooth_optimization(h_init, deadline, max_iters=150)
        
        # Local sign flips for final refinement
        h_refined, val_refined = _local_sign_flip(h_opt, max_passes=3, deadline=deadline)
        
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()
    
    # Fallback if nothing found
    if best_h is None:
        best_h = np.ones(N)
        best_val = _score(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
