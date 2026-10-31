"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
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

def _gen_legendre(n, p):
    """Generate Legendre sequence of length n using prime p."""
    h = np.zeros(n)
    for i in range(n):
        # Legendre symbol (i/p)
        if i % p == 0:
            h[i] = 0.0
        else:
            h[i] = 1.0 if pow(i, (p-1)//2, p) == 1 else -1.0
    return h

def _gen_m_sequence(n, poly):
    """Generate M-sequence (m-sequence) of length n using a primitive polynomial."""
    # poly is a list of exponents for the primitive polynomial
    # e.g., for GF(2^k), primitive polynomial x^k + ... + 1
    # This is a simplified generator. For robustness, we'll use a standard known one or random if fails.
    # Actually, for N=400, we can just use a random binary sequence or a known good one.
    # Let's stick to Legendre and simple windows as they are easier to implement correctly for arbitrary N.
    pass

def _local_search_sign_flip(h, max_passes=3, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves score."""
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

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence, score ~ 1/N

    # Update baseline score
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Legendre sequences (Algebraic, good autocorrelation properties)
    # Find a prime >= N
    p = N
    while True:
        is_prime = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 1
    leg_seq = _gen_legendre(N, p)
    # Normalize to have non-zero sum if possible, or just use as is
    if abs(np.sum(leg_seq)) < 1e-6:
        leg_seq[N//2] = 1.0 # Break symmetry to ensure non-zero sum for score calculation
    inits.append(leg_seq)
    
    # 6. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        # Low pass filter via simple moving average
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 15:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Define objective for constrained problem: sum(h)=1
        # We minimize max|autoconv| + penalty for sum != 1
        def obj_constrained(x):
            s = np.sum(x)
            penalty = 50.0 * (s - 1.0)**2
            return _autoconv_max_abs(x) + penalty
        
        # Use BFGS for local optimization
        try:
            res = opt.minimize(obj_constrained, h_norm, method='BFGS', 
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-5})
            h_opt = res.x
            
            # Check validity
            if abs(np.sum(h_opt) - 1.0) < 0.5:
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                    
                    # Refine with local sign flips
                    if time.time() < deadline - 10:
                        h_refined, val_refined = _local_search_sign_flip(h_opt, max_passes=2, deadline=deadline-5)
                        if val_refined < best_score:
                            best_score = val_refined
                            best_h = h_refined.copy()
        except Exception:
            pass

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
