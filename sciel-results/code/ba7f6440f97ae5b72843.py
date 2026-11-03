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

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses vectorized precomputation to speed up evaluation.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, 1e10
    best_val = _score(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Precompute autoconvolution once per pass
        ac = _autoconv_fft(h)
        s_sq = s * s
        current_max_abs = np.max(np.abs(ac))
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flipping h[i] to -h[i] changes sum by -2*h[i]
            # New sum: s - 2*h[i]
            new_s = s - 2.0 * h[i]
            if abs(new_s) < 1e-9:
                continue
            new_s_sq = new_s * new_s
            
            # The autoconvolution changes: ac_new = ac - 2*h[i]*h[i] at lag 0
            # Actually, flipping sign: h[i] -> -h[i]
            # ac_new[k] = ac[k] - 2*h[i]*h[i-k] for all k... this is complex
            # Let's just do the direct flip and recompute
            h[i] = -h[i]
            new_val = _autoconv_max_abs(h) / new_s_sq
            if new_val < best_val:
                best_val = new_val
                s = new_s
                improved = True
                # Update ac for next iteration
                ac = _autoconv_fft(h)
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

def _discrete_polish(h, deadline=None):
    """
    Final polish: try quantizing each element to ±1 and see if it improves.
    Also try small neighborhoods of the current solution.
    """
    h = h.copy()
    best_val = _score(h)
    
    if deadline is not None and time.time() > deadline:
        return h, best_val
    
    # Try rounding to ±1
    h_round = np.sign(h)
    # Handle zeros
    h_round[h_round == 0] = 1.0
    s_round = np.sum(h_round)
    if abs(s_round) > 1e-9:
        val_round = _score(h_round)
        if val_round < best_val:
            best_val = val_round
            h = h_round.copy()
    
    if deadline is not None and time.time() > deadline:
        return h, best_val
    
    # Try local perturbations around best
    rng = np.random.default_rng(999)
    for _ in range(50):
        if deadline is not None and time.time() > deadline:
            break
        h_trial = h.copy()
        # Perturb a random subset
        n_perturb = rng.integers(5, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            h_trial[idx] += 0.1 * rng.standard_normal()
        s_trial = np.sum(h_trial)
        if abs(s_trial) < 1e-9:
            continue
        val_trial = _score(h_trial)
        if val_trial < best_val:
            best_val = val_trial
            h = h_trial.copy()
    
    return h, best_val

def _cosine_basis_matrix(K, N):
    """
    Construct the cosine basis matrix for the first K modes.
    Modes are 0..K-1.
    """
    i = np.arange(N)
    basis = np.zeros((K, N))
    for k in range(K):
        if k == 0:
            basis[k, :] = 1.0
        else:
            basis[k, :] = np.cos(np.pi * k * i / (N - 1))
    return basis

def _optimize_in_cosine_space(K, beta_schedule, deadline, initial_coeffs=None):
    """
    Optimize coefficients of a cosine basis expansion.
    """
    if initial_coeffs is None:
        initial_coeffs = np.zeros(K)
        # Small random initialization to avoid symmetry
        rng = np.random.default_rng(42)
        initial_coeffs += rng.standard_normal(K) * 0.1

    B = _cosine_basis_matrix(K, N)
    
    def obj(coeffs):
        h = coeffs @ B
        return _objective_smooth(h, beta=beta_schedule)

    best_coeffs = initial_coeffs.copy()
    best_val = obj(best_coeffs)

    for beta in beta_schedule:
        if time.time() > deadline - 10:
            break
        
        def obj_current(c):
            return obj(c)
            
        try:
            res = minimize(obj_current, best_coeffs, method='L-BFGS-B', 
                           options={'maxiter': 100, 'disp': False, 'gtol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    h_opt = best_coeffs @ B
    # Normalize to sum 1? The objective is scale invariant in numerator/denominator?
    # max|ac| scales with h^2, (sum h)^2 scales with h^2. So it is scale invariant.
    # But numerical stability is better if sum is not tiny.
    s = np.sum(h_opt)
    if abs(s) > 1e-6:
        h_opt = h_opt / s
        val = _score(h_opt)
    else:
        val = 1e10
        
    return h_opt, val

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # 1. Cosine Subspace Optimization
    # This is the new strategy: optimize in low-dim cosine basis
    K_values = [10, 20, 50, 100]
    beta_schedule = [10, 30, 60, 120]
    
    for K in K_values:
        if time.time() > deadline - 40:
            break
            
        # Multiple random starts for each K
        for start_idx in range(3):
            if time.time() > deadline - 35:
                break
            
            rng = np.random.default_rng(100 + start_idx)
            init_coeffs = rng.standard_normal(K) * 0.5
            
            h_opt, val_opt = _optimize_in_cosine_space(K, beta_schedule, deadline, init_coeffs)
            
            if val_opt < best_score:
                best_score = val_opt
                best_h = h_opt.copy()

    # 2. Traditional Full-Space Optimization (if time permits)
    # We keep this as a fallback or secondary refinement if the subspace didn't find anything great
    # But we skip the heavy multi-start full-space L-BFGS if we already have a good result from subspace
    # to save time for local search.
    
    # Let's try a few full-space starts only if we haven't beaten a threshold
    if best_score > 1.1: # Arbitrary threshold, current best is ~1.27
        inits = []
        # 1. All ones
        inits.append(np.ones(N))
        # 2. Gaussian window
        inits.append(_gen_gaussian(N, sigma_frac=0.25))
        # 3. Chirp
        inits.append(_gen_chirp(N))
        
        x = np.arange(N)
        # 4. Cosine windows
        inits.append(np.cos(np.pi * x / N))
        
        for h_init in inits:
            if time.time() > deadline - 30:
                break
            s0 = np.sum(h_init)
            if abs(s0) < 1e-6:
                continue
            h_norm = h_init / s0
            try:
                for beta in [20, 50, 100]:
                    if time.time() > deadline - 25:
                        break
                    def obj(x):
                        return _objective_smooth(x, beta=beta)
                    res = minimize(obj, h_norm, method='L-BFGS-B', 
                                   options={'maxiter': 50, 'disp': False, 'gtol': 1e-5})
                    h_opt = res.x
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-6:
                        h_opt = h_opt / s_opt
                        val = _score(h_opt)
                        if val < best_score:
                            best_score = val
                            best_h = h_opt.copy()
            except Exception:
                pass

    # 3. Local Search Refinement
    # This is crucial for the final squeeze
    if time.time() < deadline - 15:
        h_refined, val_refined = _local_search_flip(best_h, max_passes=6, deadline=deadline-8)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    if time.time() < deadline - 8:
        h_pert, val_pert = _local_search_perturb(best_h, max_passes=3, deadline=deadline-4, step_size=0.05)
        if val_pert < best_score:
            best_score = val_pert
            best_h = h_pert.copy()
    
    if time.time() < deadline - 3:
        h_polished, val_polished = _discrete_polish(best_h, deadline=deadline-1)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
