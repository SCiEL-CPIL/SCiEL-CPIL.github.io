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
    return fsize

def _autoconv_fft(h):
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_gaussian(n, sigma_ratio=0.25):
    x = np.arange(n)
    mu = n / 2
    sigma = n * sigma_ratio
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    h = np.ones(n)
    for i in range(1, n):
        h[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _gen_sparse_support(n, k, seed=42, spacing='uniform'):
    """Generate a sparse support pattern with k active positions."""
    rng = np.random.default_rng(seed)
    if spacing == 'uniform':
        # Uniformly spaced with jitter
        base = np.linspace(0, n-1, k, endpoint=False).astype(int)
        jitter = rng.integers(-n//k//2, n//k//2 + 1, size=k)
        positions = np.sort((base + jitter) % n)
    elif spacing == 'clustered':
        # Clustered in a few regions
        n_clusters = max(2, k // 10)
        centers = rng.choice(n, size=n_clusters, replace=False)
        positions = np.sort(rng.choice(n, size=k, replace=False))
        # Bias towards clusters
        for c in centers:
            idx = np.argmin(np.abs(positions - c))
            positions[idx] = c + rng.integers(-5, 5) % n
        positions = np.sort(positions % n)
    else:  # random
        positions = np.sort(rng.choice(n, size=k, replace=False))
    return positions

def _local_search_adaptive(h, max_passes=5, deadline=None):
    """
    Adaptive local search: sign flips + multiplicative perturbations.
    Uses decreasing step sizes.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    step_sizes = [0.3, 0.15, 0.07, 0.03, 0.01]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # 1. Try sign flips
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
        
        # 2. Try multiplicative perturbations on active elements
        active = np.where(np.abs(h) > 1e-8)[0]
        if len(active) > 0:
            for i in active:
                if deadline is not None and time.time() > deadline:
                    break
                orig = h[i]
                for factor in [1 + step, 1 - step]:
                    h[i] = orig * factor
                    new_val = _score(h)
                    if new_val < best_val:
                        best_val = new_val
                        improved = True
                        break
                else:
                    h[i] = orig
        
        if not improved:
            break
            
    return h, best_val

def _gradient_sign_search(h, beta=50.0, max_iters=200, deadline=None):
    """
    Discrete gradient descent on binary sign representation.
    Decompose h = signs * |h|, then update signs using the gradient
    of the soft-max autoconvolution surrogate w.r.t. each sign.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, _score(h)
    h = h / s  # normalize sum to 1
    
    signs = np.sign(h)
    signs[signs == 0] = 1.0
    mags = np.abs(h)
    
    best_h = h.copy()
    best_val = _score(h)
    
    # Precompute FFT size
    fsize = _get_fft_size(N)
    
    def _compute_ac(h_vec):
        H = np.fft.rfft(h_vec, fsize)
        ac = np.fft.irfft(H * H, fsize)
        return ac[:2*N-1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution of current h
        h_curr = signs * mags
        s_curr = np.sum(h_curr)
        if abs(s_curr) < 1e-9:
            break
        h_curr = h_curr / s_curr
        
        ac = _compute_ac(h_curr)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        if m < 1e-15:
            break
        
        # Soft-max weight: w_j = exp(beta*(|ac_j| - m)) / sum(exp(beta*(|ac_j| - m)))
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        w = exp_scaled / sum_exp
        
        # Gradient of soft-max w.r.t. h_i:
        # d(softmax)/dh_i = sum_j w_j * d|ac_j|/dh_i
        # ac_j = sum_k h_k * h_{j-k} (linear autoconv)
        # d|ac_j|/dh_i = sign(ac_j) * (h_{j-i} + h_{j-i+N-1}) approximately
        # For simplicity, use the fact that autoconv is quadratic:
        # d(ac_j)/dh_i = h_{j-i} + h_{j-i} (symmetric)
        # Actually: ac_j = sum_i h_i * h_{j-i}
        # d(ac_j)/dh_i = h_{j-i} + (if j-i != i) h_{j-i} ... 
        # More carefully: ac_j = sum_k h_k h_{j-k}
        # d(ac_j)/dh_i = h_{j-i} + (j-i != i ? h_{j-i} : 0) = 2*h_{j-i} if j!=2i, else h_i
        # For the gradient of the soft-max objective:
        # grad_i = sum_j w_j * sign(ac_j) * (h_{j-i} + h_{j-i}) 
        # This is complex. Use a simpler approach: compute gradient via FFT.
        
        # Alternative: use the gradient of the soft-max autoconv directly.
        # The soft-max is: S = m + (1/beta)*log(sum_j exp(beta*(|ac_j|-m)))
        # dS/dh_i = sum_j w_j * sign(ac_j) * d(ac_j)/dh_i
        # where w_j = exp(beta*(|ac_j|-m)) / sum_k exp(beta*(|ac_k|-m))
        
        # d(ac_j)/dh_i: ac_j = sum_k h_k * h_{j-k}
        # = h_i * h_{j-i} + h_{j-i} * h_i + sum_{k != i, k != j-i} h_k * h_{j-k}
        # If j-i != i: d(ac_j)/dh_i = 2 * h_{j-i}
        # If j-i == i (j=2i): d(ac_j)/dh_i = h_i
        
        # This is O(N^2) to compute directly. Use FFT-based approach:
        # The gradient of the autoconvolution at lag j w.r.t. h_i is:
        # g_{i,j} = h_{j-i} + h_{j-i} (with appropriate bounds)
        # Actually, the Jacobian of autoconv is related to circulant-like structure.
        
        # Simpler: just do coordinate descent on signs using the full score.
        # For each position, try flipping the sign and see if it improves.
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            signs[i] = -signs[i]
            h_test = signs * mags
            s_test = np.sum(h_test)
            if abs(s_test) < 1e-9:
                signs[i] = -signs[i]
                continue
            h_test = h_test / s_test
            val_test = _score(h_test)
            if val_test < best_val:
                best_val = val_test
                best_h = h_test.copy()
                improved = True
            else:
                signs[i] = -signs[i]
        
        if not improved:
            break
    
    return best_h, best_val

def _high_precision_polish(h, deadline=None, max_iters=500):
    """
    High-precision polish: fine-tune amplitudes with very tight tolerance.
    Uses L-BFGS-B with very small gtol and more iterations.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, _score(h)
    h = h / s
    
    best_h = h.copy()
    best_val = _score(h)
    
    try:
        # Use very high beta for near-exact max
        def obj(x, beta=500.0):
            sx = np.sum(x)
            if abs(sx) < 1e-9:
                return 1e10
            h_norm = x / sx
            ac = _autoconv_fft(h_norm)
            sm = _soft_max_abs(ac, beta)
            return sm
        
        res = minimize(obj, h, method='L-BFGS-B',
                       options={'maxiter': max_iters, 'disp': False, 'gtol': 1e-8, 'ftol': 1e-12})
        
        h_opt = res.x
        s_opt = np.sum(h_opt)
        if abs(s_opt) > 1e-9:
            h_opt = h_opt / s_opt
            val_opt = _score(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Additional local search with very small steps
    if deadline is not None and time.time() < deadline - 1:
        for i in range(N):
            if deadline is not None and time.time() > deadline - 1:
                break
            orig = best_h[i]
            for factor in [1.01, 0.99, 1.001, 0.999]:
                best_h[i] = orig * factor
                val = _score(best_h)
                if val < best_val:
                    best_val = val
                else:
                    best_h[i] = orig
                    break
    
    return best_h, best_val

def _optimize_sparse_amplitudes(support, init_amplitudes, beta=100.0, deadline=None):
    """
    Optimize amplitudes for a fixed sparse support pattern.
    Uses gradient descent on the smooth surrogate.
    """
    h = np.zeros(N)
    h[support] = init_amplitudes
    
    s0 = np.sum(h)
    if abs(s0) < 1e-9:
        return h, 1e10
    h = h / s0  # Normalize sum to 1
    
    def obj(x):
        h_x = np.zeros(N)
        h_x[support] = x
        sx = np.sum(h_x)
        if abs(sx) < 1e-9:
            return 1e10
        h_norm = h_x / sx
        ac = _autoconv_fft(h_norm)
        sm = _soft_max_abs(ac, beta)
        return sm
    
    x0 = h[support].copy()
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
        x_opt = res.x
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        s_opt = np.sum(h_opt)
        if abs(s_opt) > 1e-9:
            h_opt = h_opt / s_opt
            val = _score(h_opt)
            return h_opt, val
    except Exception:
        pass
    
    return h, _score(h)

def _simulated_annealing_sparse(h, deadline, max_iter=50000):
    """
    Simulated annealing on sign flips for sparse sequences.
    """
    h = h.copy()
    best_h = h.copy()
    best_val = _score(h)
    cur_val = best_val
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.9997
    
    rng = np.random.default_rng(999)
    active = np.where(np.abs(h) > 1e-8)[0]
    
    if len(active) == 0:
        return best_h, best_val
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick random active index to flip
        i = active[rng.integers(0, len(active))]
        h[i] = -h[i]
        new_val = _score(h)
        
        delta = new_val - cur_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-15)):
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_h = h.copy()
        else:
            h[i] = -h[i]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

def _enforce_symmetry(h):
    """Enforce time-reversal symmetry: h[i] = h[N-1-i]."""
    n = len(h)
    h_sym = (h + h[::-1]) / 2.0
    return h_sym

def _score_symmetric(h_sym):
    """Score a symmetric half-vector by expanding to full and computing score."""
    n = len(h_sym)
    # h_sym has N elements (full vector already symmetric)
    s = np.sum(h_sym)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h_sym)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _objective_symmetric(h_sym, beta=100.0):
    """Smooth surrogate for symmetric vectors."""
    s = np.sum(h_sym)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h_sym)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _optimize_symmetric(h_init_sym, beta=100.0, max_iters=200, deadline=None):
    """
    Optimize a time-reversal symmetric vector using L-BFGS-B on the smooth surrogate.
    The symmetry is enforced by construction: we optimize the full vector but 
    project onto the symmetric subspace after each evaluation.
    """
    h_sym = _enforce_symmetry(h_init_sym.copy())
    s0 = np.sum(h_sym)
    if abs(s0) < 1e-9:
        return h_sym, 1e10
    h_sym = h_sym / s0  # Normalize sum to 1
    
    def obj(x):
        h_test = _enforce_symmetry(x)
        sx = np.sum(h_test)
        if abs(sx) < 1e-9:
            return 1e10
        h_test = h_test / sx
        ac = _autoconv_fft(h_test)
        sm = _soft_max_abs(ac, beta)
        return sm
    
    best_h = h_sym.copy()
    best_val = _score_symmetric(h_sym)
    
    try:
        res = minimize(obj, h_sym, method='L-BFGS-B',
                       options={'maxiter': max_iters, 'disp': False, 'gtol': 1e-6})
        
        h_opt = _enforce_symmetry(res.x)
        s_opt = np.sum(h_opt)
        if abs(s_opt) > 1e-9:
            h_opt = h_opt / s_opt
            val_opt = _score_symmetric(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Local search on symmetric vector: flip pairs (i, N-1-i) together
    if deadline is not None and time.time() < deadline - 2:
        for _pass in range(3):
            if deadline is not None and time.time() > deadline - 2:
                break
            improved = False
            for i in range(N // 2):
                if deadline is not None and time.time() > deadline - 2:
                    break
                j = N - 1 - i
                # Flip both h[i] and h[j] (they should be equal due to symmetry)
                best_h[i] = -best_h[i]
                best_h[j] = -best_h[j]
                new_val = _score_symmetric(best_h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    best_h[i] = -best_h[i]
                    best_h[j] = -best_h[j]
            if not improved:
                break
    
    return best_h, best_val

def _local_search_symmetric(h_sym, max_passes=5, deadline=None):
    """
    Local search on symmetric vector: try flipping symmetric pairs and 
    multiplicative perturbations on pairs.
    """
    h = _enforce_symmetry(h_sym.copy())
    best_val = _score_symmetric(h)
    rng = np.random.default_rng(42)
    
    step_sizes = [0.3, 0.15, 0.07, 0.03, 0.01]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # Try flipping symmetric pairs
        for i in range(N // 2):
            if deadline is not None and time.time() > deadline:
                break
            j = N - 1 - i
            h[i] = -h[i]
            h[j] = -h[j]
            new_val = _score_symmetric(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
                h[j] = -h[j]
        
        # Try multiplicative perturbations on symmetric pairs
        for i in range(N // 2):
            if deadline is not None and time.time() > deadline:
                break
            j = N - 1 - i
            orig_i = h[i]
            orig_j = h[j]
            for factor in [1 + step, 1 - step]:
                h[i] = orig_i * factor
                h[j] = orig_j * factor
                new_val = _score_symmetric(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
                else:
                    h[i] = orig_i
                    h[j] = orig_j
                    h[i] = orig_i * factor
                    h[j] = orig_j * factor
                    # Already restored above if no improvement
        
        if not improved:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = base_score
    
    # Phase 1: Symmetric optimization with diverse initializations
    phase1_deadline = t_start + 90.0
    
    x = np.arange(N)
    
    # Generate symmetric initializations
    sym_inits = []
    
    # Uniform
    sym_inits.append(np.ones(N))
    
    # Gaussian envelope (symmetric by construction)
    sym_inits.append(_gen_gaussian(N, sigma_ratio=0.25))
    sym_inits.append(_gen_gaussian(N, sigma_ratio=0.35))
    sym_inits.append(_gen_gaussian(N, sigma_ratio=0.45))
    
    # Cosine envelope (symmetric)
    sym_inits.append(np.cos(np.pi * x / N))
    sym_inits.append(np.cos(2 * np.pi * x / N))
    
    # Chirp made symmetric
    chirp = _gen_chirp(N)
    sym_inits.append(_enforce_symmetry(chirp))
    
    # Barker-like made symmetric
    barker = _gen_barker_like(N)
    sym_inits.append(_enforce_symmetry(barker))
    
    # Random smoothed made symmetric
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rand_smooth = _gen_random_smoothed(N, seed=seed)
        sym_inits.append(_enforce_symmetry(rand_smooth))
    
    # Alternating signs with Gaussian envelope
    gauss_env = _gen_gaussian(N, sigma_ratio=0.3)
    alt_signs = np.where(x % 2 == 0, 1.0, -1.0)
    sym_inits.append(_enforce_symmetry(gauss_env * alt_signs))
    
    # Sparse-like symmetric patterns
    for k in [N//4, N//3, N//2]:
        rng = np.random.default_rng(k * 100)
        sparse = np.zeros(N)
        positions = rng.choice(N, size=k, replace=False)
        sparse[positions] = rng.choice([-1.0, 1.0], size=k)
        sym_inits.append(_enforce_symmetry(sparse))
    
    # Optimize each symmetric initialization
    for h_init in sym_inits:
        if time.time() > phase1_deadline - 10:
            break
        
        h_sym = _enforce_symmetry(h_init.copy())
        s0 = np.sum(h_sym)
        if abs(s0) < 1e-6:
            continue
        
        # Try multiple beta values
        for beta in [50, 100, 200]:
            if time.time() > phase1_deadline - 8:
                break
            
            h_opt, val_opt = _optimize_symmetric(h_sym, beta=beta, max_iters=150, 
                                                  deadline=phase1_deadline - 5)
            if val_opt < best_score:
                best_score = val_opt
                best_h = h_opt.copy()
    
    # Phase 2: Local search on the best symmetric solution
    if time.time() < deadline - 15:
        h_refined, val_refined = _local_search_symmetric(best_h, max_passes=8, 
                                                          deadline=deadline - 10)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Phase 3: More gradient restarts from perturbed versions of best
    if time.time() < deadline - 12:
        rng = np.random.default_rng(12345)
        for restart in range(4):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            noise = rng.standard_normal(N) * 0.1
            h_pert = _enforce_symmetry(best_h + noise)
            s_pert = np.sum(h_pert)
            if abs(s_pert) < 1e-6:
                continue
            h_pert = h_pert / s_pert
            
            for beta in [100, 200]:
                if time.time() > deadline - 8:
                    break
                
                h_opt, val_opt = _optimize_symmetric(h_pert, beta=beta, max_iters=100,
                                                      deadline=deadline - 5)
                if val_opt < best_score:
                    best_score = val_opt
                    best_h = h_opt.copy()
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 8:
        h_final, val_final = _local_search_symmetric(best_h, max_passes=5,
                                                      deadline=deadline - 5)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    # Phase 5: High-precision polish with L-BFGS on symmetric vector
    if time.time() < deadline - 5:
        h_sym = _enforce_symmetry(best_h.copy())
        s0 = np.sum(h_sym)
        if abs(s0) > 1e-9:
            h_sym = h_sym / s0
            
            def obj_high(x):
                h_test = _enforce_symmetry(x)
                sx = np.sum(h_test)
                if abs(sx) < 1e-9:
                    return 1e10
                h_test = h_test / sx
                ac = _autoconv_fft(h_test)
                sm = _soft_max_abs(ac, 500.0)
                return sm
            
            try:
                res = minimize(obj_high, h_sym, method='L-BFGS-B',
                               options={'maxiter': 300, 'disp': False, 
                                        'gtol': 1e-8, 'ftol': 1e-12})
                h_opt = _enforce_symmetry(res.x)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-9:
                    h_opt = h_opt / s_opt
                    val_opt = _score_symmetric(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Ultra-fine final local search
    if time.time() < deadline - 2:
        h_uf, val_uf = _local_search_symmetric(best_h, max_passes=2,
                                                deadline=deadline - 1)
        if val_uf < best_score:
            best_score = val_uf
            best_h = h_uf.copy()
    
    # Ensure valid output
    if not np.all(np.isfinite(best_h)):
        return np.ones(N) / N
    
    # Final symmetry enforcement
    best_h = _enforce_symmetry(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
