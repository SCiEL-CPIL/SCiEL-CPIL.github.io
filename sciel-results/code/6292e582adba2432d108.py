"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize, size

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    fsize, size = _get_fft_size(n)
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

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n, beta=5.0):
    """Generate a sinc-like sequence."""
    i = np.arange(n)
    x = (i - n/2) / (n * beta)
    h = np.sinc(x)
    return h

def _gen_gaussian(n, sigma_frac=0.25):
    """Generate a Gaussian window."""
    i = np.arange(n)
    mu = n / 2
    sigma = n * sigma_frac
    return np.exp(-(i - mu)**2 / (2 * sigma**2))

def _gen_barker_like(n):
    """Generate a sequence with alternating pattern."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 in [0, 2]:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_msequence_like(n, period=7):
    """Generate an m-sequence-like pattern."""
    h = np.zeros(n)
    for i in range(n):
        # Simple LFSR-like pattern
        val = ((i // period) % 2) ^ ((i % period) % 2)
        h[i] = 1.0 if val == 0 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _gen_binary_random(n, seed=42):
    """Generate a random binary (+1/-1) sequence."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < 0.5, 1.0, -1.0)

def _simulated_annealing(h, t_start, deadline, max_iter=50000):
    """
    Simulated annealing on sign flips.
    """
    h = h.copy()
    best_h = h.copy()
    best_val = _score(h)
    cur_val = best_val
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.9995
    
    rng = np.random.default_rng(999)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick random index to flip
        i = rng.integers(0, N)
        h[i] = -h[i]
        new_val = _score(h)
        
        delta = new_val - cur_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-15)):
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_h = h.copy()
        else:
            h[i] = -h[i]  # revert
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Multiple passes until no improvement.
    """
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

def _local_search_perturb(h, max_passes=3, deadline=None, step_size=0.1):
    """
    Local optimization with small perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(123)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = old_val + step_size * rng.standard_normal()
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies - diverse set
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian windows
    inits.append(_gen_gaussian(N, sigma_frac=0.25))
    inits.append(_gen_gaussian(N, sigma_frac=0.15))
    inits.append(_gen_gaussian(N, sigma_frac=0.35))
    
    # 4. Triangle window
    mu = N/2
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Sinc sequences
    inits.append(_gen_sinc(N, beta=3.0))
    inits.append(_gen_sinc(N, beta=5.0))
    
    # 7. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 8. M-sequence-like
    inits.append(_gen_msequence_like(N, period=7))
    
    # 9. Random smoothed sequences
    for seed in [42, 123, 456, 789, 101, 202, 303]:
        inits.append(_gen_random_smoothed(N, seed=seed))
    
    # 10. Random binary sequences
    for seed in [7, 17, 27, 37]:
        inits.append(_gen_binary_random(N, seed=seed))
    
    # Phase 1: Quick continuous optimization to find good basins
    # Use smooth surrogate with L-BFGS-B
    phase1_deadline = t_start + 40.0  # 40s for continuous optimization
    
    for h_init in inits:
        if time.time() > phase1_deadline:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1
        h_norm = h_init / s0
        
        try:
            # Use smooth objective with moderate beta for initial optimization
            for beta in [20, 40]:
                if time.time() > phase1_deadline - 5:
                    break
                
                def obj(x, beta=beta):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-5})
                h_opt = res.x
                
                # Check validity and normalize
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Intensive discrete search - this is where the real gains come from
    # Start with the best continuous solution
    phase2_deadline = t_start + 135.0  # Leave 5s buffer
    
    # First, do a quick local search on the best solution
    if time.time() < phase2_deadline - 10:
        h_refined, val_refined = _local_search_flip(best_h, max_passes=3, deadline=phase2_deadline - 5)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Simulated annealing - the main workhorse
    if time.time() < phase2_deadline - 2:
        h_sa, val_sa = _simulated_annealing(best_h, t_start, phase2_deadline - 1, max_iter=100000)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final local search polish
    if time.time() < phase2_deadline:
        h_final, val_final = _local_search_flip(best_h, max_passes=5, deadline=phase2_deadline)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    # Also try local search on some other good initializations if time permits
    remaining_time = phase2_deadline - time.time()
    if remaining_time > 10:
        # Try a few more initializations with quick local search
        extra_inits = [
            _gen_binary_random(N, seed=100),
            _gen_binary_random(N, seed=200),
            np.ones(N) * np.sign(np.cos(np.pi * x / N)),
        ]
        for h_init in extra_inits:
            if time.time() > phase2_deadline - 3:
                break
            s0 = np.sum(h_init)
            if abs(s0) < 1e-6:
                continue
            h_norm = h_init / s0
            h_ls, val_ls = _local_search_flip(h_norm, max_passes=2, deadline=phase2_deadline - 2)
            if val_ls < best_score:
                best_score = val_ls
                best_h = h_ls.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
