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

def _compute_ac_delta(h, i, sign):
    """
    Compute the change in autoconvolution when h[i] changes by delta.
    If we flip h[i] to -h[i], delta = -2*h[i].
    The change in autoconv is delta * (h shifted by i) + h * (delta shifted by i)
    Actually, ac_new[k] = ac_old[k] + delta * (h[k-i] + h[k-i] for the two terms)
    More precisely: (h+dh)*(h+dh) = h*h + dh*h + h*dh + dh*dh
    So ac_new = ac_old + dh * h + h * dh + dh * dh
    where dh is a vector with delta at position i and 0 elsewhere.
    dh * h is delta * h shifted right by i (i.e., (dh*h)[k] = delta * h[k-i] if 0<=k-i<N)
    h * dh is delta * h shifted left by i (i.e., (h*dh)[k] = delta * h[k-i] if 0<=k-i<N)
    Wait, let me be more careful.
    (h * h)[k] = sum_j h[j] h[k-j]
    If we change h[i] by delta, then:
    (h_new * h_new)[k] = sum_j h_new[j] h_new[k-j]
    = sum_j (h[j] + delta*1[j=i]) (h[k-j] + delta*1[k-j=i])
    = sum_j h[j]h[k-j] + delta * sum_j h[j] 1[k-j=i] + delta * sum_j h[k-j] 1[j=i] + delta^2 * 1[j=i]1[k-j=i]
    = ac_old[k] + delta * h[k-i] + delta * h[k-i] + delta^2 * 1[k=2i]
    = ac_old[k] + 2*delta*h[k-i] + delta^2 * (1 if k==2i else 0)
    
    So the change is:
    - For all k where 0 <= k-i < N: ac[k] += 2*delta*h[k-i]
    - Additionally, ac[2*i] += delta^2 (if 2*i < 2*N-1)
    """
    n = len(h)
    size = 2 * n - 1
    ac_change = np.zeros(size)
    
    # 2*delta*h[k-i] for k in [i, i+n-1]
    # h[k-i] for k-i in [0, n-1]
    ac_change[i:i+n] += 2 * sign * h
    
    # delta^2 at position 2*i
    if 2 * i < size:
        ac_change[2 * i] += sign * sign
    
    return ac_change

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses incremental autoconvolution update for efficiency.
    Multiple passes until no improvement.
    """
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    cur_val = best_val
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try flipping h[i]
            delta = -2 * h[i]
            ac_change = _compute_ac_delta(h, i, delta)
            ac_new = ac + ac_change
            s_new = s + delta
            
            if abs(s_new) < 1e-9:
                continue
            
            new_val = np.max(np.abs(ac_new)) / (s_new * s_new)
            
            if new_val < cur_val:
                # Accept the flip
                ac = ac_new
                s = s_new
                h[i] = -h[i]
                cur_val = new_val
                improved = True
                if cur_val < best_val:
                    best_val = cur_val
            # else: reject, no change needed
        
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

def _simulated_annealing(h, t_start, deadline, max_iter=50000):
    """
    Simulated annealing on sign flips with incremental AC update.
    """
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_h = h.copy()
    best_ac = ac.copy()
    best_s = s
    best_val = np.max(np.abs(ac)) / (s * s)
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
        delta = -2 * h[i]
        ac_change = _compute_ac_delta(h, i, delta)
        ac_new = ac + ac_change
        s_new = s + delta
        
        if abs(s_new) < 1e-9:
            T *= cooling
            continue
        
        new_val = np.max(np.abs(ac_new)) / (s_new * s_new)
        
        delta_val = new_val - cur_val
        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(T, 1e-15)):
            # Accept
            ac = ac_new
            s = s_new
            h[i] = -h[i]
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_h = h.copy()
                best_ac = ac.copy()
                best_s = s
        else:
            # Reject - no change needed
            pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

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
    phase1_deadline = t_start + 35.0  # 35s for continuous optimization
    
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
                if time.time() > phase1_deadline - 3:
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
    
    # Phase 2: Intensive discrete search
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
